"""Plan 2C end to end on CPU (Task 17): two real episode miners (the ``reliquary mine-episodes`` loop:
``run_episode_miner`` -> ``EpisodeGroupMiner``) against this validator's real RL side, in one window.

Real: reliquary-sandbox's gateway over ``FakeBoxes`` on 127.0.0.1 (real routes, signing, grading); the RL
services (``build_rl_episode_services``: machine directory and fleet over a fake bucket, session book and
issuer, the ``/rl`` precommit / open / close routes, the episode intake); the miner loop, its stack
(forced ``GenerateEngine`` and its generate app, ``ForcedDraws``, ``HttpRlEpisodes``,
``RlSignedEpisodeRunner``, ``hf_prover`` over a tiny CPU model, the verdict poll); the validator's admission
worker (``ValidatorServer._process_auction_submission``: the admission child's ``prepare_submission`` run
in-process, the parent's episode intake, the submitted records, the batcher's acceptance and the
settlement of the claims); the batcher, its arrival proof and exploration audit on a real
``GlobalProofScheduler`` (``ValidationService._execute_scheduled_proof``); the seal drain, the settlement,
the final verdicts, the training payload, weight replay and the public events.

Fakes: the checkpoint is a tiny CPU model whose logits follow a script (``Checkpoint``: two equally likely
words of prose, then a bash call whose command is the answer or not; every party loads the same weights);
vLLM's core runs it with the miner's real ``ForcedSeedVLLMProcessor`` (``FakeCore``); the validator's
forward is the real ``verify_commitment_proofs`` over its own copy (GRAIL, TOPLOC, forced-seed CDF,
logprobs, stop picks). The verifiers harness (``Harness``: the same steps over the real gateway and
generate endpoint); the admission child's process boundary; the HTTP hops (ASGI, in this event loop);
the ``/submit`` envelope protocol's HTTP part; drand and ``/miner-state``.

Honest miner: plays the 2M seeds (the forced draw decides which ones find the answer), submits an in-zone
group of M (training lane), proven on arrival, selected, paid; the others withdrawn. A replay of its group
is refused. Forger: its engine writes the answer whatever command the draw picked (every seed solves: a
uniform group, exploration lane); the probation audit's forward finds a model token outside the forced
draw's CDF interval: a deterministic failure (R31): banned for a day, its window forfeited, nothing paid."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

pytest.importorskip("reliquary_sandbox_service.episodes.testing")
bt = pytest.importorskip("bittensor")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from reliquary_sandbox.episode_client import EpisodeClient  # noqa: E402
from reliquary_sandbox.tool_routing import ToolRouter  # noqa: E402
from reliquary_sandbox_service.episodes import registry, testing  # noqa: E402
from reliquary_sandbox_verifiers.runner import SandboxEpisodeResult  # noqa: E402

from reliquary import constants  # noqa: E402
from reliquary.constants import M_ROLLOUTS  # noqa: E402
from reliquary.infrastructure import sandbox_store  # noqa: E402
from reliquary.miner import episode_group_miner as group_miner_module  # noqa: E402
from reliquary.miner import episode_mining as em  # noqa: E402
from reliquary.miner.corpus_generate_server import SESSION_HEADER, Finished  # noqa: E402
from reliquary.miner.vllm_generation import FORCED_SEED_KEY  # noqa: E402
from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, TOPLOC_DEPLOYED_DEFAULTS  # noqa: E402
from reliquary.protocol.service_episode import parse_rl_engagement  # noqa: E402
from reliquary.protocol.signatures import sign_envelope  # noqa: E402
from reliquary.protocol.submission import BatchSubmissionResponse, RejectReason  # noqa: E402
from reliquary.sandbox.sessions import CLOSED, SUBMITTED  # noqa: E402
from reliquary.services import runtime as runtime_module  # noqa: E402
from reliquary.services.runtime import ServiceRuntime, protocol_slot_geometry  # noqa: E402
from reliquary.services.settlement import service_archive_rewards  # noqa: E402
from reliquary.shared.task_registry import MECHANISM_SERVICE_RL  # noqa: E402
from reliquary.shared.training_payload import decode_training_payload, encode_training_payload  # noqa: E402
from reliquary.validator import batcher as batcher_module  # noqa: E402
from reliquary.validator.batcher import audit_scheduler_environment  # noqa: E402
from reliquary.validator.fill_window import FillState  # noqa: E402
from reliquary.validator.observability import DrandRoundObservation, SubmitTelemetry  # noqa: E402
from reliquary.validator.proof_scheduler import GlobalProofScheduler  # noqa: E402
from reliquary.validator.rl_sandbox_wiring import build_rl_episode_services  # noqa: E402
from reliquary.validator.sandbox_wiring import SandboxValidatorConfig  # noqa: E402
from reliquary.validator.server import ValidatorServer, _QueuedAuctionSubmission, _UploadPrecommitReceipt  # noqa: E402
from reliquary.validator.service import ValidationService  # noqa: E402
from reliquary.validator.weight_only import WeightOnlyValidator  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    EPISODE, POOL, REVISION, TASK, WINDOW_BEACON, episode_contract, make_test_episode_env,
    register_episode_env,
)
from tests.unit.service_v2_fixtures import qualification_v2  # noqa: E402
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: E402, F401
from tests.unit.test_episode_mining import miner_state  # noqa: E402
from tests.unit.test_grpo_window_batcher import _make_batcher  # noqa: E402
from tests.unit.test_signed_episode_e2e import (  # noqa: E402, F401
    CMD_CLOSE, CMD_OPEN, JOB, RIGHT, VALIDATOR, WRONG, ScriptRenderer, r2,
)
from tests.unit.test_trajectory_parse import CHAR, EOT, GEN, TERM, TEXT, TR  # noqa: E402

HONEST = bt.Keypair.create_from_uri("//Alice")
FORGER = bt.Keypair.create_from_uri("//Charlie")
URL = "http://validator.test"
RANDOMNESS = "cd" * 32                     # the window's randomness (``test_episode_mining.miner_state``)
FORGED_TASK = TASK + 1
PICKS, SLOTS = protocol_slot_geometry()
TOPLOC = TOPLOC_DEPLOYED_DEFAULTS
CAPACITY = 4 * M_ROLLOUTS                  # both pools live at once on the one machine
CLOCK = {"round": 1_000}
SHADOW_TOPLOC_PROFILE = dataclasses.replace(
    ACTIVE_PROTOCOL_PROFILE, proofs=(*ACTIVE_PROTOCOL_PROFILE.proofs, dataclasses.replace(TOPLOC, mode="shadow")))


# -- the checkpoint: a tiny CPU model whose turns follow a script ---------------------------------------------


PROSE = (TEXT, TEXT + 1)            # two words, equally likely: a sampled draw of the checkpoint
PROSE_WORDS = 20      # each turn's prose: an episode holds >= CHALLENGE_K model tokens (the logprob challenge)


def command_ids(command: str) -> list[int]:
    return [CMD_OPEN] + [CHAR + ord(c) for c in command] + [CMD_CLOSE]


def script_options(prefix) -> tuple[int, ...] | None:
    """The tokens the checkpoint may write next after ``prefix`` (equally likely; the forced draw picks),
    or None outside a model turn. A turn starts after the renderer's generation prompt (GEN): PROSE_WORDS words
    of prose; on the first turn a bash call, whose command is the answer or not (one draw decides), then
    the turn's end; on the next turn (an observation was rendered) the turn's end."""
    prefix = list(prefix)
    if GEN not in prefix:
        return None
    start = len(prefix) - prefix[::-1].index(GEN)
    generated, before = prefix[start:], prefix[:start]
    k = len(generated)
    if k < PROSE_WORDS:
        return PROSE
    if TR in before:                                       # the second turn
        return (TERM,) if k == PROSE_WORDS else None
    if k == PROSE_WORDS:
        return (CMD_OPEN,)
    branches = {CHAR + ord(RIGHT[0]): RIGHT, CHAR + ord(WRONG[0]): WRONG}
    at = k - PROSE_WORDS - 1                               # the command's own position
    if at == 0:
        return tuple(branches)
    chosen = generated[PROSE_WORDS + 1]
    command = command_ids(branches[chosen])[1:] + [TERM] if chosen in branches else []
    return (command[at],) if at < len(command) else None


class ScriptedHead(torch.nn.Module):
    """``lm_head``: the weights' logits, replaced where the script decides (rows matched to positions)."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.last = inner, None

    def forward(self, hidden):
        logits = self.inner(hidden)
        stored, ids = self.last
        if hidden.dim() == 3:
            rows = [(0, t, t) for t in range(hidden.shape[1])]
            out = logits.clone()
            view = lambda r: out[0, r]                      # noqa: E731
        else:
            matches = [(stored[0] == row).all(-1).nonzero().flatten() for row in hidden]
            rows = [(None, int(m[0]), i) for i, m in enumerate(matches) if len(m)]
            out = logits.clone()
            view = lambda r: out[r]                         # noqa: E731
        for _, position, row in rows:
            options = script_options(ids[:position + 1])
            if options is not None:
                target = view(row)
                target.fill_(-30.0)
                target[list(options)] = 30.0
        return out


class ScriptedBase(torch.nn.Module):
    def __init__(self, inner, head):
        super().__init__()
        self.inner, self.head = inner, head

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.inner(input_ids, attention_mask=attention_mask, use_cache=False)
        self.head.last = (out.last_hidden_state.detach(), [int(t) for t in input_ids[0]])
        return out


class Checkpoint(torch.nn.Module):
    """The announced checkpoint on CPU (each party loads its own copy: same weights, same script)."""

    base_model_prefix = "model"

    _weights = None
    _weights_lock = threading.Lock()

    def __init__(self):
        import copy

        super().__init__()
        with Checkpoint._weights_lock:          # built once: every party loads the same weights
            if Checkpoint._weights is None:
                config = transformers.AutoConfig.for_model(
                    "qwen3", vocab_size=2048, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                    eos_token_id=EOT, tie_word_embeddings=True)
                torch.manual_seed(0)
                Checkpoint._weights = transformers.AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
            inner = copy.deepcopy(Checkpoint._weights)
        self.config, self.name_or_path = inner.config, "models/test"
        self.lm_head = ScriptedHead(inner.lm_head)
        self.model = ScriptedBase(inner.model, self.lm_head)
        self.eval()

    def get_input_embeddings(self):
        return self.model.inner.embed_tokens

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        hidden = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,) if output_hidden_states else None)


class FakeCore:
    """vLLM's place behind the forced ``GenerateEngine``: the checkpoint's logits, then the miner's real
    ``ForcedSeedVLLMProcessor`` one-hots the seed's pick (the engine takes the argmax). ``cheat(prompt,
    generated, pick)`` lets an engine write another token than the draw's."""

    def __init__(self, model, cheat=None):
        self.model, self.cheat, self.queue, self.draws = model, cheat, [], []
        self.turns, self.cheated = [], set()          # (seed, offset, completion); seeds written off-draw
        self._lock = threading.Lock()

    def add(self, request_id, prompt_ids, max_tokens, draw=None):
        if draw is None:
            raise ValueError("a forced core takes every request with its draw")
        forced = draw[FORCED_SEED_KEY]
        with self._lock:
            self.draws.append((int(forced["seed_index"]), int(forced["base_offset"])))
            self.queue.append((request_id, list(prompt_ids), dict(draw), int(max_tokens)))

    def has_unfinished(self):
        with self._lock:
            return bool(self.queue)

    def step(self):
        from reliquary.miner.vllm_generation import ForcedSeedVLLMProcessor

        with self._lock:
            queue, self.queue = self.queue, []
        done = []
        for request_id, prompt, draw, cap in queue:
            seed, offset = int(draw[FORCED_SEED_KEY]["seed_index"]), int(draw[FORCED_SEED_KEY]["base_offset"])
            processor = ForcedSeedVLLMProcessor()
            processor.update_state(SimpleNamespace(removed=(), moved=(),
                                                   added=[(0, SimpleNamespace(extra_args=draw))]))
            generated, logprobs = [], []
            while len(generated) < cap:
                with torch.no_grad():
                    logits = self.model(torch.tensor([prompt + generated])).logits[0, -1].float()
                pick = int(processor.apply(logits.unsqueeze(0))[0].argmax())
                if self.cheat is not None:
                    written = self.cheat(prompt, generated, pick)
                    if written != pick:
                        self.cheated.add(seed)
                    pick = written
                logprobs.append(float(torch.log_softmax(logits, -1)[pick]))
                generated.append(pick)
                if pick in (TERM, EOT):
                    break
            self.turns.append((seed, offset, tuple(generated)))
            done.append(Finished(request_id, tuple(generated), tuple(logprobs), "stop", ()))
        return done


def forger_cheat(prompt, generated, pick):
    """The forger's engine: the answer, whatever command the draw picked."""
    if len(generated) == PROSE_WORDS + 1 and pick == CHAR + ord(WRONG[0]):
        return CHAR + ord(RIGHT[0])
    return pick


FORWARDS: list[dict] = []


def cpu_forward(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
    """The validator's real verifier over its copy of the checkpoint (recorded for the assertions)."""
    from reliquary.validator.verifier import verify_commitment_proofs

    result = verify_commitment_proofs(commit, model, randomness, tokenizer=tokenizer, seed_u_values=seed_u_values)
    episode = commit["rollout"]["episode"]
    FORWARDS.append({"seed": episode["seed_index"], "uniforms": list(seed_u_values or ()),
                     "transcript": "transcript" in episode, "hard": result.seed_n_hard_mismatch,
                     "toploc": (result.toploc_checked, result.toploc_passed), "grail": result.all_passed,
                     "positions": result.seed_n_positions})
    return result


# -- the harness: verifiers' steps over the real gateway and the miner's real generate endpoint --------------


class Harness:
    def __init__(self, url, world, runner_kwargs):
        self.url, self.world, self.kwargs = url, world, runner_kwargs
        self.renderer = ScriptRenderer()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def run(self, *, token, prompt, on_trace=None):
        import uuid

        trace = SimpleNamespace(id=uuid.uuid4().hex, stop_condition=None, ok=True, errors=[])
        on_trace(trace)                    # the trace is bound to its seed's draw before its first turn
        app = self.world.generate_apps[self.kwargs["hotkey"]]
        async with EpisodeClient(self.url) as client:
            episode = await client.open(token)
            router = ToolRouter(episode, offered=tuple(episode.record0["body"]["tools"]))
            ids, error = self.renderer.initial_ids(prompt), None
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url=self.kwargs["generate_url"]) as generate:
                for turn in range(self.kwargs["policy"].max_turns):
                    answer = await generate.post("/inference/v1/generate", headers={SESSION_HEADER: trace.id},
                                                 json={"token_ids": ids, "model": self.kwargs["model_name"]})
                    if answer.status_code != 200:
                        error = f"generate {answer.status_code}: {answer.text}"
                        break
                    completion = answer.json()["choices"][0]["token_ids"]
                    calls = [(f"call_{j}", name, arguments)
                             for j, (name, arguments) in enumerate(self.renderer.tool_calls(completion))]
                    if not calls:
                        trace.stop_condition = "agent_completed"
                        break
                    observations = [await router.answer(turn, calls, call_id) for call_id, _, _ in calls]
                    ids = self.renderer.next_prompt(ids, completion, observations)
                else:
                    trace.stop_condition = "max_turns"
            finished = await episode.finish()
            transcript = (await episode.transcript())["transcript"]
        return SandboxEpisodeResult(episode.id, trace, finished.record, finished.state, transcript,
                                    tuple(router.refusals), error)


class AppBridge(httpx.BaseTransport):
    """The miner's session client is synchronous (it runs in worker threads): each request is served by
    the validator's ASGI app on the test's event loop."""

    def __init__(self, app, loop):
        self.app, self.loop = app, loop

    def handle_request(self, request):
        body = request.read()

        async def send():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url=URL) as client:
                answer = await client.request(request.method, str(request.url), headers=dict(request.headers),
                                              content=body)
                return answer.status_code, dict(answer.headers), answer.content

        status, headers, content = asyncio.run_coroutine_threadsafe(send(), self.loop).result(120)
        headers.pop("content-length", None)
        return httpx.Response(status, headers=headers, content=content)


# -- the validator ----------------------------------------------------------------------------------------------


class Tokenizer:
    eos_token_id = TERM


def make_runtime(path) -> ServiceRuntime:
    contract = episode_contract()
    path.mkdir(parents=True, exist_ok=True)
    rt = ServiceRuntime(path / "runtime.sqlite3", contract, qualification_v2(contract), now=time.time() - 10,
                        drand_round_at=lambda instant: CLOCK["round"])
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=REVISION)
    rt.open_window(1, pools={name: POOL for name in contract.environments}, picks_target=PICKS,
                   batch_slots=SLOTS, now=time.time() - 5)
    rt.announcement(window=1, randomness=WINDOW_BEACON)
    return rt


class Validator:
    """This validator in one window: the RL services, the admission worker, the episode batcher and its
    proof plane. ``submit`` is ``/submit``'s worker for one group (the HTTP envelope protocol aside)."""

    def __init__(self, gateway, rt, tmp_path):
        self.rt, self.opened = rt, time.time()
        config = SandboxValidatorConfig(key_file=tmp_path / "unused", key_id=gateway.validator.key_id,
                                        retired_keys={})
        model, tokenizer = Checkpoint(), Tokenizer()        # the validator's own copy of the checkpoint
        from reliquary.shared.modeling import resolve_eos_token_ids

        self.services = build_rl_episode_services(
            config, validator_hotkey=VALIDATOR.ss58_address, runtime=rt,
            environments={EPISODE: make_test_episode_env()}, renderer_for=lambda policy: ScriptRenderer(),
            current_window=lambda: 1, chunk_tokens=TOPLOC.chunk_tokens,
            window_started_at=lambda window: self.opened if window == 1 else None,
            proof_stop_ids=resolve_eos_token_ids(model, tokenizer), signer=gateway.validator)
        self.service = ValidationService.__new__(ValidationService)
        self.service._service_runtime = rt
        self.service._proof_models = {"d0": model}
        self.service._proof_measurements = None
        self.scheduler = GlobalProofScheduler(
            devices=("d0",), environments=(EPISODE, audit_scheduler_environment(EPISODE)),
            proof_callable=self.service._execute_scheduled_proof, checkpoint_revision=REVISION)
        b = _make_batcher(window_start=1, env=make_test_episode_env(), model=model, tokenizer=tokenizer,
                          verify_commitment_proofs_fn=cpu_forward, verify_signature_fn=None,
                          proof_scheduler=self.scheduler,
                          operator_by_hotkey={HONEST.ss58_address: "operator-a", FORGER.ss58_address: "operator-b"})
        b.randomness = RANDOMNESS
        b.service_runtime, b.service_environment = rt, EPISODE
        b.service_policy = rt.announcement(window=1, randomness=WINDOW_BEACON, environment=EPISODE)
        b.current_checkpoint_hash = REVISION
        b.fill_state = FillState(budgets={EPISODE: 64}, picks_target=PICKS)
        # A profile whose environments name the episode env (auction, final verdicts), as the order's own.
        b.difficulty_auction_enabled = True
        b.episode_proof_inconclusive = self.services.batcher_hook(EPISODE, 1)
        self.batcher = b
        self.server = ValidatorServer()
        self.server.set_active_batchers({EPISODE: b})
        self.server._episode_intake = self.services.intake
        self.server._admission_materialization_pool = ThreadPoolExecutor(max_workers=2)

        async def in_process(environment, function, *args, wall_seconds):
            # The admission child, without its process: on the main thread (its deadline is a SIGALRM).
            return function(*args)

        self.server._run_admission_process = in_process
        self.service.server = self.server
        self.service._active_batchers = {EPISODE: b}
        self.app = FastAPI()
        for router in self.services.routers:
            self.app.include_router(router)
        self.receipts, self.workers, self.requests = [], [], {}

    async def start(self):
        await self.services.start()
        await self.services.fleet.refresh_directory()
        await self.services.fleet.poll_once()

    async def submit(self, url, request, *, client, wallet, randomness):
        """The miner's ``submit_batch_v2`` and the validator's ``/submit`` in one: the envelope is finalized
        and signed as the submitter does, the body is queued for the real admission worker, and the miner
        gets its queue receipt (the verdict follows on ``/miner-verdicts``)."""
        assert url == URL and randomness == RANDOMNESS
        nonce = os.urandom(16).hex()
        signature = sign_envelope(
            wallet=wallet, miner_hotkey=request.miner_hotkey, window_start=request.window_start,
            prompt_idx=request.prompt_idx, merkle_root=request.merkle_root, checkpoint_hash=request.checkpoint_hash,
            drand_round=0, randomness=randomness, nonce=nonce, protocol_version=request.protocol_version,
            generation_profile_id=request.generation_profile_id, pool_selection=request.pool_selection,
            service_binding=request.service_binding).hex()
        request = request.model_copy(update={"drand_round": 0, "nonce": nonce, "envelope_signature": signature})
        self.requests.setdefault(request.miner_hotkey, []).append(request)
        self.enqueue(request)
        return BatchSubmissionResponse(accepted=True, reason=RejectReason.SUBMITTED)

    def enqueue(self, request):
        import hashlib

        exclude = {"generation_profile_id"} if not request.generation_profile_id else None
        raw = request.model_dump_json(exclude=exclude).encode("utf-8")
        receipt_id, now = os.urandom(8).hex(), time.time()
        # The upload protocol's bookkeeping (``/submit/precommit``, then the body's bytes), on the batcher.
        b = self.batcher
        accepted, why, _ = b.try_register_upload_precommit(receipt_id, request.miner_hotkey, t_arrival_wall=now,
                                                           payload_bytes=len(raw))
        assert accepted, why
        assert b.account_upload_precommit_bytes(receipt_id, t_arrival_wall=now, chunk_bytes=len(raw))[0]
        assert b.mark_upload_precommit_transport_complete(receipt_id, completed_at_wall=now)[0]
        assert b.mark_upload_precommit_revealed(receipt_id)
        receipt = _UploadPrecommitReceipt(
            receipt_id=receipt_id, precommit_signature="signed", miner_hotkey=request.miner_hotkey,
            prompt_idx=request.prompt_idx, window_start=request.window_start, merkle_root=request.merkle_root,
            checkpoint_hash=request.checkpoint_hash, environment=EPISODE, payload_bytes=len(raw),
            payload_sha256=hashlib.sha256(raw).hexdigest(), drand_round=request.drand_round,
            protocol_version=request.protocol_version, nonce=request.nonce,
            generation_profile_id=request.generation_profile_id, expires_at_wall=time.time() + 60.0,
            precommit_arrival_ts=time.time(),
            drand_observation=DrandRoundObservation(
                submitted_drand_round=request.drand_round, arrival_drand_round=request.drand_round, drand_delta=0,
                drand_tolerance=0, drand_status="current", reject_reason=None),
            batcher=self.batcher, consumed=True, operator=b._operator_for_hotkey(request.miner_hotkey),
            body_completed_at_wall=now)
        item = _QueuedAuctionSubmission(
            raw_body=raw, receipt=receipt, batcher=self.batcher,
            telemetry=SubmitTelemetry.from_request(request, t_arrival=time.time()), enqueued_monotonic=time.monotonic())
        self.receipts.append(receipt)
        self.workers.append(asyncio.ensure_future(self.server._process_auction_submission(item, asyncio.Queue())))
        return receipt

    async def close(self):
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.scheduler.close()
        self.server._admission_materialization_pool.shutdown(wait=True)
        await self.services.stop()


# -- the miners --------------------------------------------------------------------------------------------------


class Miners:
    """``deps`` of each miner's real stack: its own fake vLLM core and tiny proof model, the validator over
    ASGI. ``generate_apps[hotkey]``: the generate endpoint each stack serves (the harness plays on it)."""

    def __init__(self, validator: Validator, loop):
        self.validator, self.loop = validator, loop
        self.generate_apps, self.cores, self.runners = {}, {}, []

    def deps(self, keypair, cheat):
        hotkey = keypair.ss58_address

        def make_core(directory, **kwargs):
            assert kwargs["max_total_tokens"] == self.validator.rt.contract.episode_policy(EPISODE).max_episode_tokens
            self.cores[hotkey] = FakeCore(Checkpoint(), cheat)
            return self.cores[hotkey]

        async def serve(app, port):
            self.generate_apps[hotkey] = app
            serving = asyncio.get_running_loop().create_future()
            serving.set_result(None)
            return SimpleNamespace(should_exit=False), serving

        def make_runner(**kwargs):
            from reliquary.miner.rl_episode_client import RlSignedEpisodeRunner

            runner = RlSignedEpisodeRunner(**kwargs, runner_factory=lambda url: Harness(url, self, kwargs))
            self.runners.append(runner)
            return runner

        return em.EpisodeStackDeps(
            download=lambda repo, revision, cache_dir: f"/checkpoints/{revision[:8]}",
            load_renderer=lambda directory, tools: ScriptRenderer(), make_core=make_core,
            max_model_len=lambda core, default: default, release_core=lambda core: None,
            load_proof_model=lambda directory: Checkpoint(), release_model=lambda model: None,
            toploc=lambda: TOPLOC, serve=serve, make_runner=make_runner, submit=self.validator.submit,
            sync_http=lambda url: httpx.Client(base_url=url, transport=AppBridge(self.validator.app, self.loop)),
            async_http=lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=self.validator.server.app)),
            package_refusal=lambda policy: None)

    def run(self, keypair, cheat, task):
        """``run_episode_miner`` for ``keypair``, offered the one task ``task`` (all others in cooldown)."""
        stop = asyncio.Event()
        state = miner_state(self.validator.rt, cooldown=set(range(1000)) - {task}, prompt_range=(0, 1000))
        config = em.EpisodeMinerConfig(environments=(EPISODE,), validator_url=URL,
                                       validator_hotkey=VALIDATOR.ss58_address, groups_in_flight=1)

        async def read_state():
            return state

        async def poll(seconds):
            await asyncio.sleep(0.05)

        task_ = asyncio.ensure_future(em.run_episode_miner(
            config=config, wallet=SimpleNamespace(hotkey=keypair), stop=stop, deps=self.deps(keypair, cheat),
            task_prompts=em.EpisodeTaskPrompts([EPISODE]), read_state=read_state, sleep=poll))
        return stop, task_


# -- helpers ------------------------------------------------------------------------------------------------------


async def until(predicate, seconds=120.0, what="", explain=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        await asyncio.sleep(0.05)
    assert predicate(), f"timed out waiting for {what}: {explain() if explain else ''}"


def proof_state(b):
    return {"rows": list(b.difficulty_auction_metadata_by_id.values()), "forwards": FORWARDS,
            "pending": [(p.hotkey[:8], p.proof_reject_stage, getattr(p, "proof_reject_scope", None))
                        for p in b.pending_submissions()]}


def sessions_of(services, hotkey):
    """{seed: record} of the hotkey's RL sessions (one precommit each here)."""
    out = {}
    for record in services.book.records():
        if record.hotkey == hotkey:
            out[parse_rl_engagement(record.engagement)[2]] = record
    return out


@pytest.fixture
def wide_gateway(tmp_path, monkeypatch):
    """The signed corpus e2e's gateway with room for two pools at once."""
    monkeypatch.setattr(registry, "_package_of", lambda entry: JOB.episode.sandbox.env_package)
    with testing.fake_gateway(tmp_path / "gw", envs={"reliquary-swe": testing.FAKE_ENV_ENTRY},
                              episode_capacity=CAPACITY) as (served, boxes):
        served.boxes = boxes
        yield served


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    CLOCK["round"] = 1_000
    FORWARDS.clear()
    monkeypatch.setattr(runtime_module, "_verified_beacon", lambda round_id: "ef" * 32)
    # The validator's profile names TOPLOC (as the order's must for signed episodes).
    monkeypatch.setattr(batcher_module, "_active_profile", lambda: SHADOW_TOPLOC_PROFILE)
    # The v6 fill-closed window (arrival proofs, FIFO picks), as the order's validator runs it.
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(group_miner_module, "VERDICT_POLL_S", 0.1)
    monkeypatch.setattr(group_miner_module, "VERDICT_POLL_MAX_S", 0.3)
    monkeypatch.setattr(constants, "SERVICE_EXPLORATION_DRAIN_SECONDS", 20.0)
    monkeypatch.setattr(constants, "SERVICE_EXPLORATION_DRAIN_PROBATION_SECONDS", 20.0)


# -- the test --------------------------------------------------------------------------------------------------


async def test_two_miners_one_window_precommit_sessions_choice_proof_audit_settlement_replay(
        wide_gateway, r2, tmp_path, monkeypatch):  # noqa: F811
    register_episode_env(monkeypatch)
    await sandbox_store.register_machine(
        machine_id=wide_gateway.settings.machine_id, address=wide_gateway.url, provider="local",
        capacity=CAPACITY, key_id=wide_gateway.machine.key_id, public_key_b64=wide_gateway.machine.public_key_b64,
        valid_from=0, now=time.time())
    rt = make_runtime(tmp_path / "rt")
    v = Validator(wide_gateway, rt, tmp_path)
    await v.start()
    miners = Miners(v, asyncio.get_running_loop())
    b, services, server = v.batcher, v.services, v.server
    two_m = 2 * M_ROLLOUTS
    try:
        honest_stop, honest = miners.run(HONEST, None, TASK)
        forger_stop, forger = miners.run(FORGER, forger_cheat, FORGED_TASK)

        # 1. Both groups reach the batcher: every seed of each 2M pool opened once, played and closed final.
        await until(lambda: len(v.receipts) >= 2 and all(r.terminal for r in v.receipts[:2]), 300,
                    "both groups admitted")
        for receipt in v.receipts[:2]:
            # Admitted into the pool (the miner reads it on ``/miner-verdicts``): its kept episodes are its group.
            assert receipt.outcome.accepted and receipt.outcome.reason is RejectReason.ACCEPTED, receipt.outcome
        for hotkey, task in ((HONEST.ss58_address, TASK), (FORGER.ss58_address, FORGED_TASK)):
            precommit = rt.episode_precommit_sha(window=1, environment=EPISODE, task_index=task, hotkey=hotkey)
            sessions = sessions_of(services, hotkey)
            assert sorted(sessions) == list(range(two_m))
            assert {parse_rl_engagement(r.engagement)[1] for r in sessions.values()} == {precommit}
        (honest_request,) = v.requests[HONEST.ss58_address]
        (forged_request,) = v.requests[FORGER.ss58_address]
        # Every model turn of every seed was generated under that seed's forced draw (the miner's own
        # processor), each turn's offset counting the episode's model tokens so far; the draw decided
        # which seeds found the answer.
        solved = {}
        for hotkey in (HONEST.ss58_address, FORGER.ss58_address):
            turns = miners.cores[hotkey].turns
            assert {seed for seed, _, _ in turns} == set(range(two_m))
            for seed in range(two_m):
                (first, offset0), (second, offset1) = sorted(((c, o) for s_, o, c in turns if s_ == seed),
                                                             key=lambda item: item[1])
                assert (offset0, offset1) == (0, len(first)) and second[-1] == TERM
                solved[hotkey, seed] = command_ids(RIGHT)[1] in first
        honest_seeds = honest_request.pool_selection["seeds"]
        honest_rewards = [r.reward for r in honest_request.rollouts]
        assert honest_rewards == [float(solved[HONEST.ss58_address, s]) for s in honest_seeds]
        assert 0.0 in honest_rewards and 1.0 in honest_rewards              # in zone: the training lane
        assert honest_request.service_binding["purpose"] == "training"
        # The forger solved every seed: a uniform group, sent as exploration; some of its seeds' draws had
        # picked the wrong command.
        forged_seeds = forged_request.pool_selection["seeds"]
        assert all(solved[FORGER.ss58_address, s] for s in range(two_m))
        assert forged_request.service_binding["purpose"] == "exploration"
        assert set(forged_seeds) & miners.cores[FORGER.ss58_address].cheated
        # The chosen sessions are paid (submitted) the moment the batcher takes the group.
        honest_sessions = sessions_of(services, HONEST.ss58_address)
        assert {honest_sessions[s].state for s in honest_seeds} == {SUBMITTED}

        # 2. The training group is proven on arrival (the real scheduler and verifier, the batcher's own
        # checks, the checkpoint's forward on CPU): proven, its training observation recorded.
        await until(lambda: (b._reconcile_fill_state_decisions(EPISODE),
                             len(b._proven_groups.get(EPISODE, [])) == 1)[1], 60, "the training proof",
                    lambda: proof_state(b))
        (proven,) = [group.value for group in b._proven_groups[EPISODE]]
        assert proven.hotkey == HONEST.ss58_address and proven.prompt_idx == TASK
        assert [r.reward for r in proven.rollouts] == honest_rewards
        honest_forwards = FORWARDS[:M_ROLLOUTS]
        assert [f["seed"] for f in honest_forwards] == honest_seeds and not any(f["transcript"] for f in FORWARDS)
        assert all(f["hard"] == 0 and f["toploc"] == (True, True) and f["grail"] for f in honest_forwards)
        pool = rt.seed_pool(environment=EPISODE, prompt_idx=TASK, window=1)
        for forward in honest_forwards:
            # The validator checks each model token against the seed's pool uniforms, by model-token offset.
            assert forward["uniforms"] == [pool.uniform(forward["seed"], j) for j in range(forward["positions"])]

        # 3. A replay of the honest group (same sessions) is refused before any claim.
        real_admit, intake_answers = services.intake.admit, []

        async def admit(**kwargs):
            prepared, claim = await real_admit(**kwargs)
            intake_answers.append((prepared.reject_reason, prepared.reject_stage, claim))
            return prepared, claim

        services.intake.admit = admit
        replay = v.enqueue(honest_request)
        await until(lambda: replay.terminal, 60, "the replay's verdict")
        assert (replay.outcome.accepted, replay.outcome.reason) == (False, RejectReason.HASH_DUPLICATE)
        assert intake_answers == [(RejectReason.HASH_DUPLICATE, "episode_session_reused", None)]
        # (the miner reads the server's identity refusal, checked after the intake: same reason)
        assert server._verdicts[HONEST.ss58_address][-1]["reject_stage"] == "logical_dedup"
        assert {honest_sessions[s].state for s in honest_seeds} == {SUBMITTED}

        # 4. The forged group: exploration, forced to audit (probation); the audit's forward finds model
        # tokens the forced draw never picked: deterministic (R31), banned, its window forfeited.
        CLOCK["round"] += 100                                  # its draw round is due
        await until(lambda: (b.service_exploration_tick(), rt.exploration_banned(FORGER.ss58_address))[1], 60,
                    "the forger's audit")
        (row,) = rt.ledger.rows(1, environment=EPISODE)
        assert row["status"] == "forfeited" and row["audit"] == "failed"
        forged_pending = [p for p in b.pending_submissions() if p.hotkey == FORGER.ss58_address]
        assert forged_pending and forged_pending[0].proof_reject_stage == "forced_seed"
        assert forged_pending[0].proof_reject_scope == "cdf_hard_mismatch"
        assert b.difficulty_auction_metadata_by_id[id(forged_pending[0])]["status"] == "exploration_forfeited"
        assert not rt.exploration_banned(HONEST.ss58_address)
        ban = rt.contract.reward_policy["ban_seconds"]
        assert ban == 86_400 and not rt.exploration_banned(FORGER.ss58_address, now=time.time() + ban + 1)

        # 5. The seal: the audits drained, every env finalized, the window settled from its batch.
        await v.service._drain_service_exploration([b])
        paid = [proven]
        archive = {"window_start": 1, "window_status": "complete", "rewards_by_hotkey": {},
                   "batch": [{"hotkey": g.hotkey, "env_name": EPISODE, "prompt_idx": g.prompt_idx} for g in paid]}
        result = await v.service._settle_service_archive(rt, archive)
        assert set(result["rewards_by_hotkey"]) == {HONEST.ss58_address}
        assert result["rewards_by_hotkey"][HONEST.ss58_address] > 0
        v.service._auction_final_verdict_records(b, paid_groups=paid, finalize=True)

        # 6. The miners read their verdicts: the honest group is submitted, every other episode withdrawn.
        def settled(hotkey, chosen):
            states = {seed: record.state for seed, record in sessions_of(services, hotkey).items()}
            return all(states[s] == (SUBMITTED if s in chosen else CLOSED) for s in range(two_m))

        await until(lambda: settled(HONEST.ss58_address, honest_seeds) and settled(FORGER.ss58_address, forged_seeds),
                    60, "the withdrawals")
        (honest_final,) = [x for x in server._verdicts[HONEST.ss58_address] if x.get("selected_for_batch") is not None]
        assert honest_final["merkle_root"] == honest_request.merkle_root
        assert honest_final["accepted"] and honest_final["selected_for_batch"] and honest_final["rewarded"]
        (forger_final,) = [x for x in server._verdicts[FORGER.ss58_address] if x.get("selected_for_batch") is not None]
        assert forger_final["merkle_root"] == forged_request.merkle_root
        assert not forger_final["rewarded"] and forger_final["outcome_code"] == "exploration_unpaid"
    finally:
        for stop in (locals().get("honest_stop"), locals().get("forger_stop")):
            if stop is not None:
                stop.set()
        for task in (locals().get("honest"), locals().get("forger")):
            if task is not None:
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 60)
        await v.close()

    # 7. Weight replay pays exactly what the seal settled.
    record = {**json.loads(json.dumps(result)), "task_id": "next-rl"}
    replayed = service_archive_rewards(record, rt.contract, cap=0.5, picks_target=PICKS, batch_slots=SLOTS)
    assert replayed == pytest.approx(result["rewards_by_hotkey"], abs=1e-12)
    declared = {"next-rl": SimpleNamespace(mechanism=MECHANISM_SERVICE_RL, service_contract=rt.contract.to_dict(),
                                           params={"cap": 0.5})}
    (out,) = WeightOnlyValidator._validated_service_archives([record], declared)
    assert out["rewards_by_hotkey"] == pytest.approx(result["rewards_by_hotkey"], abs=1e-12)

    # 8. The training payload of the proven group, encoded as an episode protocol (v5+) writes it: tokens,
    # model-token spans, rewards and the generation checkpoint; no transcript. (No pi_old here: the
    # window was proven at this profile's T_PROTO, where the verify forward gives none.)
    monkeypatch.setattr(constants, "PROTOCOL_VERSION", 7)
    monkeypatch.setattr(constants, "T_PROTO", 1.0)
    monkeypatch.setattr(constants, "PI_OLD_FROM_VERIFY_LOGPROBS", True)
    monkeypatch.setattr(constants, "RECOMPUTE_PI_OLD_FROM_VERIFY", True)
    data = encode_training_payload({EPISODE: paid}, window_start=1, checkpoint_revision=REVISION,
                                   env_order=[EPISODE], env_targets={EPISODE: len(paid)},
                                   window_quarantine={"quarantined": False, "reasons": []})
    assert b"transcript" not in data and HONEST.ss58_address.encode() not in data
    (group,) = decode_training_payload(data).batches()[EPISODE]
    assert [r.reward for r in group.rollouts] == honest_rewards
    for trained_rollout, sent in zip(group.rollouts, proven.rollouts):
        assert trained_rollout.commit["tokens"] == sent.commit["tokens"]
        assert [list(span) for span in trained_rollout._validated_assistant_spans] == \
            sent.commit["rollout"]["episode"]["assistant_spans"]
        assert trained_rollout.checkpoint_revision == REVISION
        assert "transcript" not in trained_rollout.commit["rollout"]["episode"]
        assert "transcript" in sent.commit["rollout"]["episode"]          # the miner's commit is untouched

    # 9. Publication: the training observation like a single-turn one; no tokens, transcript or hotkey.
    events = [event for _, event in rt.events(limit=10_000)]
    published = json.dumps(events)
    assert "transcript" not in published and "tokens" not in published
    assert HONEST.ss58_address not in published and FORGER.ss58_address not in published
    trained = [e for e in events if e.get("type") == "observation" and e.get("lane") == "training"]
    assert trained and trained[0]["candidate"]["seeds"] == honest_seeds
    assert trained[0]["rewards_bps"] == [int(r * 10000) for r in honest_rewards]
    assert any(key.startswith(sandbox_store.RL_SESSION_PREFIX) for key in r2.client.objects)
