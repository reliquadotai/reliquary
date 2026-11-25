"""A chain order (``orders.chains``): each SciCode position's set built from the answers graded at the one
before it, published to the subnet bucket once, and the chain's rates in the combined report. On the
two-problem SciCode of ``test_scicode_steps`` (positions: 1.1; 1.2 and 62.2; 1.3). Skipped without `verifiers`."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("verifiers.v1")

from reliquary.corpus.delivery import LocalDirectorySink  # noqa: E402
from reliquary.eval import verifiers_source as vs  # noqa: E402
from reliquary.eval.storage import SubnetEvalStore, publish_set, subnet_key  # noqa: E402
from reliquary.orders import chains  # noqa: E402
from tests.unit import test_scicode_steps as sc  # noqa: E402
from tests.unit.test_admin_eval_jobs import MODEL, REVISION, admin  # noqa: E402,F401
from tests.unit.test_jobs_cli import registry  # noqa: E402,F401

FIRST = {"split": "test", "step": 1, "previous": {}}


class Memory:
    """The subnet bucket: create-only, as ``SubnetEvalStore``."""

    def __init__(self):
        self.objects = {}

    async def get_bytes(self, key):
        return self.objects.get(key)

    async def create_bytes(self, key, body):
        from reliquary.eval.storage import SetConflict

        if key in self.objects:
            if self.objects[key] != body:
                raise SetConflict(key)
            return False
        self.objects[key] = body
        return True


class Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return list(range(len(messages[-1]["content"]) // 10))


@pytest.fixture
def scicode(monkeypatch):
    monkeypatch.setattr(sc.ts, "read_problems", lambda split, revision=sc.ts.DATASET_REVISION: sc.PROBLEMS)
    monkeypatch.setattr(sc.chain, "read_problems", lambda split, revision=sc.ts.DATASET_REVISION: sc.PROBLEMS)
    monkeypatch.setitem(sc.ts.PROBLEMS, "test", 2)
    monkeypatch.setitem(sc.ts.SCORED_STEPS, "test", 4)
    clock = iter(range(1, 1000))

    def open_taskset(name, args):
        assert name == "scicode-steps"
        return vs.TasksetHandle(name, dict(args), sc.ts.SciCodeStepsTaskset(sc.ts.SciCodeStepsConfig(**args)),
                                package="scicode-steps", package_version="0.1.1", verifiers_version="test")

    def build(args, out):
        card = vs.build_set("verifiers:scicode-steps", out=out, start=0, count=None, sample=None, seed=None,
                            set_id=None, taskset_args=args, open_taskset=open_taskset,
                            clock=lambda: float(next(clock)))
        return card

    def next_set(eval_id, platform, subnet, tmp_path):
        return asyncio.run(chains.next_set(
            chain_id="scicode-steps", eval_id=eval_id, platform=platform, subnet=subnet,
            open_taskset=open_taskset, load_tokenizer=lambda model, revision: (Tokenizer(), 100),
            clock=lambda: float(next(clock)), work_dir=tmp_path / "work"))

    return type("SciCode", (), {"build": staticmethod(build), "next_set": staticmethod(next_set),
                                "open_taskset": staticmethod(open_taskset)})


def grade(platform, subnet, eval_id, set_id, answers, *, thinking=False, complete=True, **provenance):
    """What a grading leaves: its report (the set, the model) and graded rows; ``answers`` by step in set order:
    (completion, correct), or None for a problem with no row."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from reliquary.eval.grading import _graded_schema

    card = json.loads(asyncio.run(subnet.get_bytes(subnet_key(set_id, "set.json"))))
    order = [json.loads(line)["problem_id"]
             for line in asyncio.run(subnet.get_bytes(subnet_key(set_id, "prompts.jsonl"))).decode().splitlines()]
    assert len(order) == len(answers) == card["count"]
    rows = [{"env": card["env"], "set_id": set_id, "problem_id": pid, "sample_index": 0, "completion": text,
             "tokens": 5, "finish_reason": "stop", "correct": correct,
             "score": None if correct is None else float(correct),
             "grader_detail": "", "turns": None, "tool_calls": None, "transcript_tokens": None,
             "episode_stop_reason": None}
            for pid, answer in zip(order, answers) if answer is not None for text, correct in [answer]]
    root = platform._path(f"evaluations/{eval_id}/graded.parquet")
    root.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_graded_schema()), root)
    report = {"complete": complete, "envs": {card["env"]: {"pass@1": {"value": 0.5}, "pass@k": {"1": {"value": 0.5}},
                                                       "no_graded_rows": False}},
              "provenance": {"model": MODEL, "revision": REVISION, "thinking": thinking,
                             "checkpoint_sha256": "c" * 64, "sets": [{"set_id": set_id}], **provenance}}
    asyncio.run(platform.put_json(f"evaluations/{eval_id}/report.json", report))
    asyncio.run(platform.put_json(f"evaluations/{eval_id}/manifest.json", {"eval_id": eval_id}))


def first(scicode, subnet, tmp_path):
    card = scicode.build(FIRST, tmp_path / "p1")
    asyncio.run(publish_set(tmp_path / "p1", platform=None, subnet=subnet))
    return card["set_id"]


def test_each_position_is_built_from_the_answers_graded_before_it(scicode, tmp_path):
    platform, subnet = LocalDirectorySink(tmp_path / "bucket"), Memory()
    p1 = first(scicode, subnet, tmp_path)
    grade(platform, subnet, "order-eval-x-1", p1,
          [("<think>no</think>```python\nimport numpy as np\ndef f(a):\n    return a\n```", True)])
    second = scicode.next_set("order-eval-x-1", platform, subnet, tmp_path)
    assert (second["position"], second["positions"], second["count"]) == (2, 3, 2)
    card = json.loads(asyncio.run(subnet.get_bytes(subnet_key(second["set_id"], "set.json"))))
    # The answer as its grader read it (after the reasoning block), its code as upstream extracts it.
    assert card["taskset"]["args"] == {"split": "test", "step": 2,
                                       "previous": {"1.1": "\ndef f(a):\n    return a\n"}}
    prompts = asyncio.run(subnet.get_bytes(subnet_key(second["set_id"], "prompts.jsonl"))).decode()
    assert "def f(a):\n    return a" in json.loads(prompts.splitlines()[0])["messages"][-1]["content"]
    assert second["prompt_tokens"]["max"] > 0 and second["prompt_tokens"]["total"] >= second["prompt_tokens"]["max"]
    # Asked again (a lost answer): the set already there, though its card was written at another time.
    assert scicode.next_set("order-eval-x-1", platform, subnet, tmp_path) == second
    # A step with no graded row is carried as no code.
    grade(platform, subnet, "order-eval-x-2", second["set_id"], [None, ("```python\ndef g(a): pass\n```", False)])
    third = scicode.next_set("order-eval-x-2", platform, subnet, tmp_path)
    card = json.loads(asyncio.run(subnet.get_bytes(subnet_key(third["set_id"], "set.json"))))
    assert card["taskset"]["args"]["previous"] == {"1.1": "\ndef f(a):\n    return a\n", "1.2": "",
                                                   "62.2": "\ndef g(a): pass\n"}
    grade(platform, subnet, "order-eval-x-3", third["set_id"], [("x", True)])
    with pytest.raises(chains.ChainRefused, match="chain_complete"):
        scicode.next_set("order-eval-x-3", platform, subnet, tmp_path)

    report = asyncio.run(chains.chain_report(chain_id="scicode-steps", platform=platform, subnet=subnet,
                                             part_ids=["order-eval-x-1", "order-eval-x-2", "order-eval-x-3"]))
    # Problem 1: 1.1 and 1.3 passed, 1.2 never answered; problem 62: 62.2 failed.
    assert (report["steps_passed"], report["problems_solved"], report["complete"]) == (2, 0, True)
    assert report["subproblem_pass_rate"] == 0.5 and report["main_problem_resolve_rate"] == 0.0
    # Positions not reached count as failed; a chain starts at its first position and stays linked.
    partial = asyncio.run(chains.chain_report(chain_id="scicode-steps", platform=platform, subnet=subnet,
                                              part_ids=["order-eval-x-1"]))
    assert (partial["steps_passed"], partial["complete"], partial["positions_graded"]) == (1, False, 1)
    for parts in (["order-eval-x-2"], ["order-eval-x-1", "order-eval-x-3"]):
        with pytest.raises(chains.ChainRefused):
            asyncio.run(chains.chain_report(chain_id="scicode-steps", platform=platform, subnet=subnet,
                                            part_ids=parts))


def test_a_position_not_graded_or_of_another_set_is_refused(scicode, tmp_path):
    platform, subnet = LocalDirectorySink(tmp_path / "bucket"), Memory()
    with pytest.raises(chains.PartNotGraded):
        scicode.next_set("order-eval-y-1", platform, subnet, tmp_path)
    with pytest.raises(chains.ChainRefused, match="chain_unknown"):
        asyncio.run(chains.next_set(chain_id="nope", eval_id="order-eval-y-1", platform=platform, subnet=subnet))
    p1 = first(scicode, subnet, tmp_path)
    card = json.loads(asyncio.run(subnet.get_bytes(subnet_key(p1, "set.json"))))
    subnet.objects[subnet_key("other-set", "set.json")] = json.dumps({**card, "source": "verifiers:x"}).encode()
    subnet.objects[subnet_key("other-set", "grading.jsonl")] = subnet.objects[subnet_key(p1, "grading.jsonl")]
    grade(platform, subnet, "order-eval-y-1", p1, [("x", False)])
    report = json.loads((tmp_path / "bucket" / "evaluations" / "order-eval-y-1" / "report.json").read_text())
    report["provenance"]["sets"] = [{"set_id": "other-set"}]
    asyncio.run(platform.put_json("evaluations/order-eval-y-1/report.json", report))
    with pytest.raises(chains.ChainRefused, match="not 'verifiers:scicode-steps'"):
        scicode.next_set("order-eval-y-1", platform, subnet, tmp_path)


def test_the_admin_builds_the_next_position_and_combines_the_chain(admin, scicode, tmp_path,  # noqa: F811
                                                                   monkeypatch):
    from reliquary.validator import eval_control

    monkeypatch.setattr(vs, "open_taskset", scicode.open_taskset)
    monkeypatch.setattr(eval_control, "load_cpu_tokenizer", lambda model, revision: (Tokenizer(), 100))
    platform, subnet = LocalDirectorySink(admin.root / "p"), SubnetEvalStore()
    p1 = first(scicode, subnet, tmp_path)
    assert admin("POST", "/admin/v1/evaluations/order-eval-z-1/chain-next",
                 {"chain": "scicode-steps"}).status_code == 409  # not graded yet
    grade(platform, subnet, "order-eval-z-1", p1, [("```python\ndef f(a): return a\n```", True)])
    assert admin("POST", "/admin/v1/evaluations/order-eval-z-1/chain-next",
                 {"chain": "nope"}).status_code == 422
    response = admin("POST", "/admin/v1/evaluations/order-eval-z-1/chain-next", {"chain": "scicode-steps"})
    assert response.status_code == 200, response.text
    second = response.json()
    assert second["position"] == 2 and second["count"] == 2
    grade(platform, subnet, "order-eval-z-2", second["set_id"], [("x", True), ("y", True)])
    combined = admin("POST", "/admin/v1/evaluations/order-eval-z/combine",
                     {"parts": ["order-eval-z-1", "order-eval-z-2"], "chain": "scicode-steps"})
    assert combined.status_code == 200, combined.text
    assert combined.json()["complete"] is False  # position 3 never ran
    report = json.loads((admin.root / "p" / "evaluations" / "order-eval-z" / "report.json").read_text())
    assert report["chain"]["steps_passed"] == 3 and report["chain"]["problems_solved"] == 1
    assert report["macro"]["pass@1"] == 0.75 and report["macro"]["basis"] == "chain_steps"
    # The same parts as a plain combination: another evaluation under this id.
    assert admin("POST", "/admin/v1/evaluations/order-eval-z/combine",
                 {"parts": ["order-eval-z-1", "order-eval-z-2"]}).status_code == 409


def _edit_report(platform, eval_id, change):
    report = asyncio.run(platform.get_json(f"evaluations/{eval_id}/report.json"))
    change(report)
    asyncio.run(platform.put_json(f"evaluations/{eval_id}/report.json", report))


def test_only_a_complete_grading_of_one_model_seeds_and_reports_a_chain(scicode, tmp_path):
    platform, subnet = LocalDirectorySink(tmp_path / "bucket"), Memory()
    p1 = first(scicode, subnet, tmp_path)
    grade(platform, subnet, "order-eval-w-1", p1, [("```python\ndef f(a): return a\n```", True)])
    _edit_report(platform, "order-eval-w-1", lambda r: r.update(complete=False))
    with pytest.raises(chains.ChainRefused, match="position_incomplete"):
        scicode.next_set("order-eval-w-1", platform, subnet, tmp_path)
    _edit_report(platform, "order-eval-w-1", lambda r: r.update(complete=True))
    second = scicode.next_set("order-eval-w-1", platform, subnet, tmp_path)
    grade(platform, subnet, "order-eval-w-2", second["set_id"], [("x", True), ("y", False)])
    parts = ["order-eval-w-1", "order-eval-w-2"]
    asyncio.run(chains.chain_report(chain_id="scicode-steps", platform=platform, subnet=subnet, part_ids=parts))
    for field, value in (("model", "other/Model"), ("revision", "d" * 40), ("thinking", True)):
        _edit_report(platform, "order-eval-w-2", lambda r: r["provenance"].update({field: value}))
        with pytest.raises(chains.ChainRefused, match="another model"):
            asyncio.run(chains.chain_report(chain_id="scicode-steps", platform=platform, subnet=subnet,
                                            part_ids=parts))
        grade(platform, subnet, "order-eval-w-2", second["set_id"], [("x", True), ("y", False)])


@pytest.mark.parametrize("rows", [
    lambda rows: [{**rows[0], "source_index": "0"}],
    lambda rows: [{k: v for k, v in rows[0].items() if k != "source_index"}],
    lambda rows: [{**rows[0], "source_index": 7}],
    lambda rows: ["not a row"],
])
def test_a_malformed_grading_file_is_a_refusal_not_a_crash(admin, scicode, tmp_path, monkeypatch,  # noqa: F811
                                                           rows):
    import hashlib

    from reliquary.validator import eval_control

    monkeypatch.setattr(vs, "open_taskset", scicode.open_taskset)
    monkeypatch.setattr(eval_control, "load_cpu_tokenizer", lambda model, revision: (Tokenizer(), 100))
    platform, subnet = LocalDirectorySink(admin.root / "p"), SubnetEvalStore()
    card = scicode.build(FIRST, tmp_path / "p1")
    original = [json.loads(line) for line in (tmp_path / "p1" / "grading.jsonl").read_text().splitlines()]
    body = "".join(json.dumps(r) + "\n" for r in rows(original)).encode()
    (tmp_path / "p1" / "grading.jsonl").write_bytes(body)
    card["grading_sha256"] = hashlib.sha256(body).hexdigest()
    (tmp_path / "p1" / "set.json").write_text(json.dumps(card))
    asyncio.run(subnet.create_bytes(subnet_key(card["set_id"], "grading.jsonl"), body))
    asyncio.run(subnet.create_bytes(subnet_key(card["set_id"], "set.json"), json.dumps(card).encode()))
    asyncio.run(subnet.create_bytes(subnet_key(card["set_id"], "prompts.jsonl"),
                                    (tmp_path / "p1" / "prompts.jsonl").read_bytes()))
    grade(platform, subnet, "order-eval-v-1", card["set_id"], [("x", True)])
    response = admin("POST", "/admin/v1/evaluations/order-eval-v-1/chain-next", {"chain": "scicode-steps"})
    assert response.status_code == 409, response.text


def test_a_position_that_can_get_no_more_work_goes_on_its_missing_steps_failed(scicode, tmp_path):
    """Order 7f5362da: position 1 graded 63 of 64, one prompt exhausted (every attempt refused by the audit), the
    grading incomplete. The chain goes on: that step's code is carried as none and it fails, as upstream counts a
    missing answer; the chain report says how many steps each position missed."""
    platform, subnet = LocalDirectorySink(tmp_path / "bucket"), Memory()
    p1 = first(scicode, subnet, tmp_path)
    # Graded while its prompt could still take work (a grading neither complete nor at the job's end): refused.
    grade(platform, subnet, "order-eval-u-1", p1, [None], complete=False, job_complete=False)
    with pytest.raises(chains.ChainRefused, match="position_incomplete"):
        scicode.next_set("order-eval-u-1", platform, subnet, tmp_path)
    # Its only prompt exhausted (job_complete: every prompt holds its sample or used every attempt).
    grade(platform, subnet, "order-eval-u-1", p1, [None], complete=False, job_complete=True, prompts_exhausted=1)
    second = scicode.next_set("order-eval-u-1", platform, subnet, tmp_path)
    assert second["from"] == {"position": 1, "asked": 1, "answered": 0, "graded": 0, "missing": 1, "ungraded": 0}
    card = json.loads(asyncio.run(subnet.get_bytes(subnet_key(second["set_id"], "set.json"))))
    assert card["taskset"]["args"]["previous"] == {"1.1": ""}
    # Cut by the run deadline (graded with what the job held): goes on too; an ungraded answer is carried
    # forward (its code exists) and fails its step.
    grade(platform, subnet, "order-eval-u-2", second["set_id"], [("```python\ndef g(a): pass\n```", None),
                                                                 ("y", True)],
          complete=False, job_complete=False, allow_incomplete=True)
    third = scicode.next_set("order-eval-u-2", platform, subnet, tmp_path)
    assert third["from"]["ungraded"] == 1
    card = json.loads(asyncio.run(subnet.get_bytes(subnet_key(third["set_id"], "set.json"))))
    assert card["taskset"]["args"]["previous"]["1.2"] == "\ndef g(a): pass\n"
    grade(platform, subnet, "order-eval-u-3", third["set_id"], [("z", True)])
    report = asyncio.run(chains.chain_report(chain_id="scicode-steps", platform=platform, subnet=subnet,
                                             part_ids=["order-eval-u-1", "order-eval-u-2", "order-eval-u-3"]))
    # 1.1 missing, 1.2 ungraded: failed; 62.2 and 1.3 passed.
    assert (report["steps_passed"], report["problems_solved"]) == (2, 1)
    assert (report["steps_missing"], report["steps_ungraded"]) == (1, 1)
    assert [(p["position"], p["missing"], p["ungraded"], p["graded"]) for p in report["per_position"]] == [
        (1, 1, 0, 0), (2, 0, 1, 1), (3, 0, 0, 1)]


def test_a_position_mostly_unanswered_stops_the_chain(scicode, tmp_path, monkeypatch):
    """Under the graded share (default 90 %, at least one step may always miss) something systemic happened:
    the chain stops rather than carrying blanks."""
    assert chains.missing_allowed(64) == 6 and chains.missing_allowed(2) == 1 and chains.missing_allowed(1) == 1
    monkeypatch.setenv(chains.MIN_GRADED_ENV, "0.5")
    assert chains.missing_allowed(64) == 32
    monkeypatch.setenv(chains.MIN_GRADED_ENV, "bad")
    assert chains.min_graded() == chains.DEFAULT_MIN_GRADED
    monkeypatch.delenv(chains.MIN_GRADED_ENV)
    platform, subnet = LocalDirectorySink(tmp_path / "bucket"), Memory()
    p1 = first(scicode, subnet, tmp_path)
    grade(platform, subnet, "order-eval-t-1", p1, [("```python\ndef f(a): return a\n```", True)])
    second = scicode.next_set("order-eval-t-1", platform, subnet, tmp_path)
    grade(platform, subnet, "order-eval-t-2", second["set_id"], [None, ("y", None)], complete=False,
          job_complete=True, prompts_exhausted=1)
    with pytest.raises(chains.ChainRefused, match="graded 0 of 2 steps"):
        scicode.next_set("order-eval-t-2", platform, subnet, tmp_path)
