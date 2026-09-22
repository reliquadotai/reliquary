"""A synthetic fleet against the pure package. These pin the design claims:
a job terminates, collision waste is negligible, and deviating does not pay."""

from reliquary.corpus.admission import admit
from reliquary.corpus.job import JOB_SCHEMA, parse_job
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.walk import CursorLedger, walk_index

SHA = "c" * 64
EOS = 151645


def _job(prompt_count, slots_per_prompt):
    return parse_job(
        {
            "schema": JOB_SCHEMA,
            "job_id": "sim-v1",
            "checkpoint_repo": "ReliquaryForge/Reliquary-4B",
            "checkpoint_revision": "abc123",
            "checkpoint_sha256": SHA,
            "prompt_source": "reliquarylogic",
            "prompt_count": prompt_count,
            "renderer_id": "reliquary/render/v5",
            "eos_token_id": EOS,
            "sampling": {
                "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                "min_new_tokens": 1, "max_new_tokens": 100, "n": 1,
            },
            "slots_per_prompt": slots_per_prompt,
            "filter": None,
            "prompt_order": "miner_walk",
            "deadline_round": None,
        }
    )


def _run(job, hotkeys, attempts_each):
    """Every miner walks its own order until it runs out of attempts."""
    slots = SlotLedger(job.prompt_count, job.slots_per_prompt)
    cursors = CursorLedger()
    seen: set[str] = set()
    tally = {"accepted": 0, "prompt_full": 0, "job_complete": 0}
    for attempt in range(attempts_each):
        for hotkey in hotkeys:
            cursor = cursors.expected(hotkey)
            index = walk_index(job.job_id, hotkey, cursor, job.prompt_count)
            digest = f"{hotkey}:{attempt}"
            verdict = admit(
                job,
                hotkey=hotkey,
                cursor=cursor,
                prompt_index=index,
                checkpoint_sha256=SHA,
                token_counts=[10],
                terminations=["eos"],
                last_token_ids=[EOS],
                digests=[digest],
                slots=slots,
                cursors=cursors,
                seen=seen,
            )
            if verdict.accepted:
                seen.add(digest)
            tally[verdict.reason] = tally.get(verdict.reason, 0) + 1
    return slots, cursors, tally


def test_a_job_terminates():
    job = _job(prompt_count=50, slots_per_prompt=2)
    slots, _, tally = _run(job, [f"5M{i}" for i in range(10)], attempts_each=60)
    assert slots.is_complete is True
    assert tally["accepted"] == job.total_slots
    assert tally["job_complete"] > 0


def test_waste_is_negligible_while_the_source_is_large():
    # Waste is roughly miners-in-flight over open slots, so a large source
    # makes it vanish. The RL system's refusals came from a rationed space.
    job = _job(prompt_count=20_000, slots_per_prompt=4)
    _, _, tally = _run(job, [f"5M{i}" for i in range(20)], attempts_each=50)
    attempted = tally["accepted"] + tally.get("prompt_full", 0)
    assert tally.get("prompt_full", 0) / attempted < 0.01


def test_grinding_the_cursor_gains_nothing():
    # A miner that tries a cursor other than its own is refused outright, so
    # it cannot hunt for a prompt it prefers.
    job = _job(prompt_count=1000, slots_per_prompt=4)
    slots = SlotLedger(job.prompt_count, job.slots_per_prompt)
    cursors = CursorLedger()
    refused = 0
    for candidate in range(1, 40):
        verdict = admit(
            job,
            hotkey="5Greedy",
            cursor=candidate,
            prompt_index=walk_index(job.job_id, "5Greedy", candidate, job.prompt_count),
            checkpoint_sha256=SHA,
            token_counts=[10],
            terminations=["eos"],
            last_token_ids=[EOS],
            digests=[f"g{candidate}"],
            slots=slots,
            cursors=cursors,
            seen=frozenset(),
        )
        assert verdict.accepted is False
        assert verdict.reason == "bad_cursor"
        refused += 1
    assert refused == 39
    assert slots.filled == 0


def test_the_state_replays_from_its_snapshots():
    # A weight-only node rebuilds from the archive and must agree.
    job = _job(prompt_count=500, slots_per_prompt=2)
    slots, cursors, _ = _run(job, ["5Ma", "5Mb", "5Mc"], attempts_each=40)

    revived_slots = SlotLedger.from_snapshot(
        job.prompt_count, job.slots_per_prompt, slots.snapshot()
    )
    revived_cursors = CursorLedger.from_snapshot(cursors.snapshot())
    assert revived_slots.filled == slots.filled
    assert revived_slots.is_complete == slots.is_complete
    for hotkey in ("5Ma", "5Mb", "5Mc"):
        assert revived_cursors.expected(hotkey) == cursors.expected(hotkey)
