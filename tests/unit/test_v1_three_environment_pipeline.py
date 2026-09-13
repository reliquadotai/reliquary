"""CPU wiring proof: real V1 assembly/codec/journal/accumulator, no optimizer.

The admitted rollout fixtures are synthetic; GRAIL, model quality and the
external checker's own conformance have separate qualification gates.
"""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys


def _run_full_window():
    import pytest

    from reliquary import constants as c
    from reliquary.infrastructure.training_payload_queue import (
        encoded_window_journal_key, payload_key,
    )
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, render_active_prompt
    from reliquary.shared.training_payload import active_training_identity, decode_training_payload
    from reliquary.trainer.journal import WindowJournal
    from reliquary.trainer.train_runner import TrainRunner
    from reliquary.trainer.worker import TrainerWorker
    from reliquary.validator.fill_closed_batch_assembler import FillClosedBatchAssembler
    from reliquary.validator.training import _plan_from_batches
    from tests.unit.test_training_payload_codec import _group, _roll

    env_order = [name for name, _ in c.ENVIRONMENT_MIX]
    targets = dict(c.ENVIRONMENT_MIX)
    assert c.PROTOCOL_PROFILE_ID == "qwen3-4b-base-dapo-reliquary-v1"
    assert c.PROTOCOL_VERSION == 6 and c.FILL_CLOSED_ENABLED
    assert env_order == ["openmathinstruct", "opencodeinstruct", "reliquary_logic_v2"]
    assert targets == dict.fromkeys(env_order, 16)
    assert c.B_BATCH == c.M_ROLLOUTS == c.FILL_CLOSED_EMISSIONS_PER_WINDOW == 16
    assert c.FILL_CLOSED_TARGET_GROUPS_PER_ENV == 256
    logic = ACTIVE_PROTOCOL_PROFILE.environments["reliquary_logic_v2"]
    assert logic.environment_contract_id == "reliquary/answer-json/v1"
    assert logic.answer_format == "last_json_object_v1"
    native_prompt = (
        'Solve the following problem step by step.\n\nFixture task.\n\n'
        'After your reasoning, give the final answer in the last fenced JSON code block.'
    )
    assert render_active_prompt("reliquary_logic_v2", problem=native_prompt) == native_prompt

    window, revision = 42, "a" * 40
    blobs, writes, tombstones = {}, [], []
    originals, expected_rewards = {}, defaultdict(float)

    def enqueue(key, data):
        assert key not in writes
        writes.append(key)
        blobs[payload_key(key)] = data

    assembler = FillClosedBatchAssembler(
        window_start=window, env_order=env_order, enqueue_fn=enqueue,
        tombstone_fn=lambda key, data: tombstones.append(key), window_pool=1.0,
    )
    chunks = {}
    for env_index, environment in enumerate(env_order):
        chunks[environment] = []
        for emission in range(16):
            groups = []
            for group_index in range(16):
                # Reuse the codec's validator-logprob fixture; every group has
                # mixed rewards, and environment/group-specific token lengths.
                group = _group([
                    _roll(float((rollout_index + group_index) % 2),
                          4 + env_index + group_index % 2 + rollout_index % 3,
                          env=environment)
                    for rollout_index in range(16)
                ], prompt_idx=env_index * 10000 + emission * 16 + group_index)
                group.hotkey = f"{environment}-miner-{group_index % 2}"
                group.eos_tokens = sum(r.commit["rollout"]["completion_length"] for r in group.rollouts)
                originals[group.prompt_idx] = group
                groups.append(group)
            for group in groups:
                expected_rewards[group.hotkey] += 1 / 3 / 16 / 16
            chunks[environment].append(groups)

    # Two fast environments cannot produce any emission before Logic arrives.
    for environment in env_order[:2]:
        for chunk in chunks[environment]:
            assembler.accept(environment, chunk, window, revision)
    assert writes == []
    for emission, chunk in enumerate(chunks[env_order[2]]):
        assembler.accept(env_order[2], chunk, window, revision)
        assert len(writes) == emission + 1
    assembler.close()
    assembler.close()  # no duplicate tail after a full window
    expected_keys = [encoded_window_journal_key(window, i) for i in range(16)]
    assert writes == expected_keys
    assert tombstones == [] and assembler.durable_payload_count == 16
    assert assembler.reward_map() == pytest.approx(dict(expected_rewards))
    assert sum(assembler.reward_map().values()) == pytest.approx(1.0)
    paid = assembler.paid_groups()
    assert set(paid) == set(env_order)
    for environment in env_order:
        assert len(paid[environment]) == 256
        assert Counter(index for index, _ in paid[environment]) == dict.fromkeys(range(16), 16)
        assert sum(value for key, value in assembler.reward_map().items()
                   if key.startswith(environment + "-miner-")) == pytest.approx(1 / 3)

    # Inspect the exact bytes the journal will later decode, with all 48 groups.
    for emission, key in enumerate(writes):
        decoded = decode_training_payload(blobs[payload_key(key)])
        assert decoded.training_identity == active_training_identity()
        assert decoded.env_order == env_order
        assert decoded.window_start == window and decoded.checkpoint_revision == revision
        assert decoded.window_quarantine["quarantined"] is False
        batches = decoded.batches()
        assert {name: len(groups) for name, groups in batches.items()} == targets
        for environment in env_order:
            assert [g.prompt_idx for g in batches[environment]] == [g.prompt_idx for g in chunks[environment][emission]]

    consumed, step_sizes = Counter(), []

    def cpu_step_spy(model, batches, **kwargs):
        # Real metadata/advantage planning, but deliberately no tensor forward,
        # optimizer, publication or simulated GPU-capacity claim.
        assert kwargs["window_index"] == window
        assert kwargs["global_step_hint"] == 0 and kwargs["ref_model"] is None
        assert len(batches) == 3 and all(len(groups) == 16 for groups in batches)
        plan, skipped = _plan_from_batches(batches)
        assert skipped == 0 and len(plan) == 48
        assert all(scale > 0 for _, _, scale in plan)
        for environment, groups in zip(env_order, batches):
            for group in groups:
                original = originals[group.prompt_idx]
                assert len(group.rollouts) == 16
                consumed[group.prompt_idx] += 1
                for actual, expected in zip(group.rollouts, original.rollouts):
                    assert actual.env_name == environment
                    assert actual.reward == expected.reward
                    assert actual.commit["tokens"] == expected.commit["tokens"]
                    assert actual._validated_completion_logprobs == pytest.approx(
                        expected._validated_completion_logprobs, abs=1e-6,
                    )
        step_sizes.append(sum(len(group.rollouts) for groups in batches for group in groups))
        return model

    runner = TrainRunner(model=object(), env_targets=targets, env_order=env_order,
                         train_step_fn=cpu_step_spy, global_step_hint=0)
    journal = WindowJournal(fetch_fn=blobs.get, expected_identity=active_training_identity())
    cursors = []

    def no_publication(reason):
        raise AssertionError("this CPU wiring test must never publish")

    worker = TrainerWorker(journal=journal, train_fn=runner.step,
                           publish_fn=no_publication, head_revision_fn=lambda: revision,
                           cursor=expected_keys[0] - 1, stride=1, publish_every=16,
                           last_published_revision=revision, cursor_writer=cursors.append)
    for key in expected_keys:
        assert worker.run_once() == "trained"
        assert worker.cursor == key
    assert cursors == expected_keys and worker.trained_since_publish == 16
    assert worker.cursor - (expected_keys[0] - 1) == 16
    assert journal.next_entry(worker.cursor, stride=1) is None
    assert runner.groups_dropped_missing_pi_old == 0
    assert runner.snapshot()["accumulator"]["counts"] == dict.fromkeys(env_order, 0)
    assert consumed == Counter({prompt: 1 for prompt in originals})
    assert step_sizes == [768] * 16 and sum(step_sizes) == 12288
    print(json.dumps({"emissions": 16, "environments": env_order, "groups_per_environment": 256,
                      "rollouts_per_step": 768, "total_rollouts": 12288, "cursor_advances": 16,
                      "reward_pool": sum(assembler.reward_map().values()), "optimizer_executed": False}))


def test_full_v1_window_reaches_three_environment_trainer_on_cpu(tmp_path):
    # Profile constants must be imported in a fresh process; never mutate the
    # default profile of the rest of the unit suite.
    import reliquary

    core = Path(reliquary.__file__).resolve().parents[1]
    env = {name: value for name, value in os.environ.items()
           if name in {"PATH", "HOME", "TMPDIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"}}
    env.update(PYTHONPATH=str(core), RELIQUARY_PROTOCOL_PROFILE="qwen3-4b-base-dapo-reliquary-v1",
               RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED="1",
               RELIQUARY_TRAINING_RUN_ID="v1-three-environment-cpu-test",
               RELIQUARY_STATE_DIR=str(tmp_path), HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            cwd=core, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary["total_rollouts"] == 12288 and summary["cursor_advances"] == 16
    assert summary["optimizer_executed"] is False


if __name__ == "__main__":
    _run_full_window()
