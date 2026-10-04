"""Stand-ins for an episode job on the split harness (tests/unit/corpus_split_harness):
the episode intake over the fake turn renderer, a span-aware GPU score
function that logs what crossed the GPU process's wire, the grade executor
registry on the shared on-disk bucket, and two fake grade executors speaking
the real HTTP lease protocol. Not a test module itself."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from tests.unit import corpus_split_harness as h

EPISODE_JOB, EPISODE_TASK = "swe-agentic-v1", "corpus-swe"
# One file per GPU call that carried spans: the proof that the trajectory was
# scored by the GPU process, with its spans, and not elsewhere.
SPANS_LOG = "gpu-spans.jsonl"
EXECUTOR_TOKENS = {"g0": "token-g0-" + "x" * 30, "g1": "token-g1-" + "y" * 30}
PROVIDERS = {"g0": "p0", "g1": "p1"}


def episode_manifest() -> dict:
    from tests.unit.test_corpus_job_episode import _manifest

    return _manifest(prompt_count=3, job_id=EPISODE_JOB, checkpoint_sha256=h.CHECKPOINT,
                     checkpoint_repo="org/Frozen", checkpoint_revision="abc123")


def env_pin() -> tuple[str, str]:
    env = episode_manifest()["episode"]["env"]
    return env["package"], env["version"]


def score_sequences(model, sequences, *, chunk_tokens, topk, batch_tokens):
    """The single-turn rows as the harness scores them; a trajectory row (its
    spans as a fourth field) gets honest chunks per span, and is logged."""
    from reliquary.validator.corpus_audit import MIN_CHUNK_TOKENS, span_chunk_count
    from tests.unit.corpus_split_fakes import HONEST_CHUNK

    single = [row for row in sequences if len(row) == 3]
    single_scores = iter(h.gpu_score_sequences(model, single, chunk_tokens=chunk_tokens,
                                               topk=topk, batch_tokens=batch_tokens)[0]
                         if single else [])
    scores = []
    for row in sequences:
        if len(row) == 3:
            scores.append(next(single_scores))
            continue
        tokens, prompt_len, proofs, spans = row
        count = sum(span_chunk_count(e - s, chunk_tokens, MIN_CHUNK_TOKENS) for s, e in spans)
        with open(h._root() / SPANS_LOG, "a") as log:
            log.write(json.dumps({"tokens": len(tokens), "prompt_len": prompt_len,
                                  "proofs": len(proofs), "spans": [list(s) for s in spans]})
                      + "\n")
        scores.append(("ok", tuple(HONEST_CHUNK for _ in range(count))))
    return scores, 0.0, 0.0


def install() -> None:
    """``corpus_split_harness.install`` plus the episode pieces, in every child."""
    h.install()
    from reliquary.infrastructure import corpus_executor_store
    from reliquary.validator import agentic_intake, corpus_audit, corpus_auditor
    from reliquary.validator import corpus_grade_remote

    corpus_executor_store.get_s3_client = lambda **kw: h.FileS3()
    corpus_audit.score_sequences = score_sequences
    corpus_auditor.score_sequences = score_sequences
    # The real lease check needs the SWE-smith task set; the fake source has none.
    corpus_grade_remote.check_replay_lease = lambda job, **kw: None

    def build_episode_intake(job, *, checkpoint_dir, tokenizer, vocab_size, chunk_tokens):
        from reliquary.environment.agentic_swe import SweSource
        from tests.unit.test_corpus_route_episode import ROWS, TOKENIZER
        from tests.unit.test_trajectory_parse import R

        return agentic_intake.EpisodeIntake(job=job, source=SweSource(ROWS), renderer=R,
                                            tokenizer=TOKENIZER, vocab_size=vocab_size,
                                            chunk_tokens=chunk_tokens)

    agentic_intake.build_episode_intake = build_episode_intake


def seed_episode(root: Path, *, hotkeys) -> None:
    """The episode job's manifest, its miners, and two grade executors on two
    providers in the registry."""
    from reliquary.corpus.audit_policy import MinerState
    from reliquary.infrastructure import corpus_executor_store as executor_store
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure import corpus_record_store as record_store
    from reliquary.validator.corpus_audit_remote import token_sha256

    h.put_raw(root, job_store._job_key(EPISODE_JOB), json.dumps(episode_manifest()).encode())
    miners = {hk: MinerState(audited_passed=1000).to_dict() for hk in hotkeys}
    h.put_raw(root, record_store._miners_key(EPISODE_JOB), json.dumps(miners).encode())
    package, version = env_pin()
    for executor_id, token in EXECUTOR_TOKENS.items():
        h.put_raw(root, executor_store._key(executor_id), json.dumps({
            "executor_id": executor_id, "token_sha256": token_sha256(token),
            "model_id": package, "model_revision": version, "expires_at": 1e12,
            "status": "active", "scope": "grade", "provider_id": PROVIDERS[executor_id],
        }).encode())


def submission(hotkey: str) -> dict:
    """An honest trajectory for prompt 0, as the route test builds it."""
    from tests.unit.test_corpus_route_episode import _request

    body = _request().model_dump()
    body.update(miner_hotkey=hotkey, checkpoint_sha256=h.CHECKPOINT,
                job_id=EPISODE_JOB)
    return body


class FakeExecutors:
    """Two grade executors in threads of the test process: they claim leases
    through ``/corpus/internal/grade/*`` and answer every grade as passed and
    every replay as matching, with the counts the lease asks for."""

    def __init__(self, base_url: str) -> None:
        self._base = base_url
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._loop, args=(eid,), daemon=True)
                         for eid in EXECUTOR_TOKENS]
        self.answered: list[tuple[str, str, str]] = []      # (executor, mode, outcome)
        self.errors: list[str] = []

    def _loop(self, executor_id: str) -> None:
        import httpx

        package, version = env_pin()
        auth = {"Authorization": f"Bearer {EXECUTOR_TOKENS[executor_id]}"}
        claim = {"executor_id": executor_id, "env_package": package, "env_version": version}
        with httpx.Client(base_url=self._base, timeout=30.0) as http:
            while not self._stop.is_set():
                try:
                    http.post("/corpus/internal/grade/heartbeat",
                              json={"executor_id": executor_id, "detail": {"leases": 0}},
                              headers=auth)
                    response = http.post("/corpus/internal/grade/claim", json=claim, headers=auth)
                    if response.status_code != 200:
                        if response.status_code not in (204,):
                            self.errors.append(f"{executor_id} claim {response.status_code}")
                        self._stop.wait(0.3)
                        continue
                    lease = response.json()
                    (item,) = lease["items"]
                    if item["mode"] == "grade":
                        result = {"status": "ok", "submission_id": item["submission_id"],
                                  "diff_applied": True, "tests_passed": True}
                    else:
                        compared = sum(1 for a in item["actions"]
                                       if a.get("observation") is not None)
                        result = {"status": "ok", "submission_id": item["submission_id"],
                                  "replay_diff_equal": True, "observations_compared": compared,
                                  "observations_mismatched": []}
                    answered = http.post(f"/corpus/internal/grade/{lease['lease_id']}/result",
                                         json={"results": [result]}, headers=auth)
                    outcome = (answered.json().get("outcome") if answered.status_code == 200
                               else f"HTTP {answered.status_code}")
                    self.answered.append((executor_id, item["mode"], outcome))
                except Exception as exc:  # noqa: BLE001 - the front may still be starting
                    self.errors.append(f"{executor_id}: {exc!r}")
                    self._stop.wait(0.5)

    def __enter__(self):
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        for thread in self._threads:
            thread.join(10)
        return False


def wait(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {what}")
