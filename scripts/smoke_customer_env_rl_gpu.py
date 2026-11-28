"""GPU smoke: an RL lease on a customer env, end to end across the processes that disagree when settings drift.

1. ``contract``: a training order naming the customer env (max_new_tokens, thinking) becomes its task's contract.
2. ``controller`` (under that contract): serves the env's rows (``/rl/customer-envs``) and writes a lease, whose
   ``env_settings`` come from the contract.
3. ``miner`` (the compiled profile, NOT the contract): reads the rows from the controller and generates the lease's
   16 rollouts on vLLM with the lease's settings.
4. ``validator`` (under the contract): grades the group with ``GroupGrader`` (prompt render, length cap,
   termination) and scores it through the customer's reward worker.

    python scripts/smoke_customer_env_rl_gpu.py all --model /root/models/qwen35-4b --work /root/smoke \\
        --worker-manifest /opt/reliquary/manifest.json

The reward worker runs as a subprocess here (a pod has no Docker): the sealed container is covered by
``tests/unit/test_customer_env_docker.py``; this checks the RL path around it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROFILE = "qwen35-4b-reliquary-suite-v10-gumbel-dev1"
ENV = "customer:smoke/toy-add@" + "5" * 64
MAX_NEW_TOKENS, THINKING = 256, False
PORT = 18733
# The reward worker's command: its own interpreter (verifiers and the customer's package; a reward host's venv).
WORKER = os.environ.get("SMOKE_WORKER", f"{sys.executable} -I -m reliquary.customer_env.worker").split()


def contract(args) -> None:
    from reliquary.admin.training import CreateTraining, training_entry
    from reliquary.validator.corpus_periods import usd_price_params

    config = json.loads((Path(args.model) / "config.json").read_text())
    body = CreateTraining(training_id="smoke", profile_id=PROFILE, account="smoke",
                          environments=[{"name": "reliquary_dapo_math_v1", "share_bps": 5000},
                                        {"name": ENV, "share_bps": 5000, "max_new_tokens": MAX_NEW_TOKENS,
                                         "thinking": THINKING}],
                          limits={"max_steps": 1, "deadline_seconds": 3600}, max_price_usd_per_mtoken=1.0,
                          batch_groups=1)
    prices = usd_price_params(usd_per_mtoken_input=0., usd_per_mtoken_output=1., alpha_usd=1., tao_usd=1.,
                              miner_alpha_per_period=1., set_at=1000.)
    entry = training_entry(body, task_prefix="order-", created_at=1000, price_params=prices,
                           customer_env=lambda env, account: None, model_architecture=config["architectures"][0])
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "contract.json").write_text(json.dumps(entry.contract, sort_keys=True))
    print(json.dumps({"task": entry.task_id, "profile_sha256": entry.profile_sha256,
                      "env": entry.contract["environments"][ENV]}))


def freeze(args) -> None:
    """The toy env's rows, frozen by its own worker (what qualification publishes)."""
    from reliquary.customer_env.runner import RewardClient, WorkerProcess

    client = RewardClient(WorkerProcess(WORKER, start_timeout_s=600))
    rows, refused = client.freeze_all()
    client.close()
    body = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in rows)
    (Path(args.work) / "prompts.jsonl").write_bytes(body)
    print(json.dumps({"rows": len(rows), "refused": len(refused)}))


def controller(args) -> None:
    import hashlib

    import uvicorn
    from fastapi import APIRouter, FastAPI

    from reliquary.customer_env import rl
    from reliquary.validator.rl_leases import Lease

    assert ENV in rl.task_environments(), "the controller's contract does not name the customer env"
    body = (Path(args.work) / "prompts.jsonl").read_bytes()
    rl._ROWS_CACHE[ENV] = (body, hashlib.sha256(body).hexdigest())
    fields = {f: None for f in Lease.__dataclass_fields__}
    fields.update(lease_id="smoke-1", environment=ENV, prompt_idx=args.prompt_idx, checkpoint=0, nonce="ab" * 32,
                  issued_at=time.time(), expires_at=time.time() + 3600, state="open")
    (Path(args.work) / "lease.json").write_text(json.dumps(Lease(**fields).to_public()))
    router = APIRouter()
    rl.mount_rows_routes(router)
    app = FastAPI()
    app.include_router(router)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def miner(args) -> None:
    from reliquary.miner.rl_async import HeldLease, VllmGroupGenerator
    from reliquary.shared.ckpt_delta import load_anchor_shards
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE

    assert ENV not in ACTIVE_PROTOCOL_PROFILE.environments, "the miner must run its compiled profile"
    lease = HeldLease.from_wire(json.loads((Path(args.work) / "lease.json").read_text()))
    generator = VllmGroupGenerator(base_model_dir=args.model, work_dir=str(Path(args.work) / "miner"), swap="reload",
                                   engine_kwargs={"gpu_memory_utilization": 0.6})
    generator.load_weights(load_anchor_shards(sorted(Path(args.model).glob("*.safetensors"))), 0)
    prompt = generator._prompt(lease)
    (group,) = generator.generate([lease])
    (Path(args.work) / "rollouts.json").write_text(json.dumps({"prompt": prompt, "rollouts": group}))
    print(json.dumps({"settings": lease.env_settings, "prompt_tokens": len(prompt),
                      "lengths": [len(r) for r in group],
                      "first": generator.tokenizer.decode(group[0])[:200]}))


def validator(args) -> None:
    from transformers import AutoTokenizer

    from reliquary.customer_env import rl
    from reliquary.customer_env.runner import RewardClient, WorkerProcess
    from reliquary.customer_env.scoring import CustomerScorer
    from reliquary.shared.modeling import resolve_eos_token_ids
    from reliquary.validator.rl_submit import GroupGrader, Rollout

    from transformers import GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:      # as the miner reads them (``VllmGroupGenerator._load_model_files``)
        meta = type("Meta", (), {"generation_config": GenerationConfig.from_pretrained(args.model)})()
    except OSError:
        meta = None
    rows = [json.loads(line) for line in (Path(args.work) / "prompts.jsonl").read_text().splitlines() if line]
    rl._SCORER["default"] = CustomerScorer(lambda env: "local@sha256:" + "0" * 64, lambda image: RewardClient(
        WorkerProcess(WORKER, start_timeout_s=600)))
    env = rl.CustomerEnvironment(ENV, rows=rows)
    grader = GroupGrader({ENV: env}, tokenizer, eos_token_ids=sorted(resolve_eos_token_ids(meta, tokenizer)),
                         vocab_size=len(tokenizer))
    sent = json.loads((Path(args.work) / "rollouts.json").read_text())
    graded = grader.grade(ENV, args.prompt_idx, [Rollout(tuple(r)) for r in sent["rollouts"]])
    same_prompt = list(graded.prompt_tokens) == list(sent["prompt"])
    print(json.dumps({"reject": graded.reject, "rewards": graded.rewards, "truncated": graded.truncated,
                      "same_prompt_tokens": same_prompt, "problem": env.get_problem(args.prompt_idx)["prompt"]}))
    if graded.reject is not None or not same_prompt or not graded.rewards:
        raise SystemExit(1)


def run_all(args) -> None:
    me = [sys.executable, __file__]
    common = ["--model", args.model, "--work", args.work, "--prompt-idx", str(args.prompt_idx)]
    task = {**os.environ, "RELIQUARY_TASK_CONTRACT": str(Path(args.work) / "contract.json")}
    plain = {k: v for k, v in os.environ.items() if k != "RELIQUARY_TASK_CONTRACT"}
    subprocess.run([*me, "contract", *common], env=plain, check=True)
    subprocess.run([*me, "freeze", *common], env=plain, check=True)
    served = subprocess.Popen([*me, "controller", *common], env=task)
    try:
        time.sleep(8)
        subprocess.run([*me, "miner", *common], check=True,
                       env={**plain, "RELIQUARY_CUSTOMER_ENV_ROWS_URL": f"http://127.0.0.1:{PORT}/rl"})
        subprocess.run([*me, "validator", *common], env=task, check=True)
    finally:
        served.terminate()
    print("SMOKE PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["contract", "freeze", "controller", "miner", "validator", "all"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--prompt-idx", type=int, default=3)
    args = parser.parse_args()
    {"contract": contract, "freeze": freeze, "controller": controller, "miner": miner, "validator": validator,
     "all": run_all}[args.stage](args)


if __name__ == "__main__":
    main()
