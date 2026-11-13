"""The quick single-turn eval tasksets of ``tasksets/``: their graders against
gold and wrong answers, and their tasks as eval-set problems (no download: each
taskset's ``read_rows`` is replaced). Skipped without `verifiers`."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

vf = pytest.importorskip("verifiers.v1")
pytest.importorskip("math_verify")

from reliquary.eval import verifiers_source as vs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "tasksets"


def module(package: str):
    path = str(ROOT / package)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(f"{package}.taskset")


def scored(task, answer: str) -> float:
    assert vs.refusal(task) is None
    assert not vs.needs_runtime(task)
    return vs.score_trace(task, vs.freeze(task), answer)


# --- HMMT February 2026 -------------------------------------------------------

HMMT_GOLD_RIGHT_WRONG = [
    ("-\\frac{1}{21}", ["\\boxed{-\\frac{1}{21}}", "\\boxed{-1/21}", "\\boxed{-\\dfrac{1}{21}}"],
     ["\\boxed{\\frac{1}{21}}", "\\boxed{-\\frac{1}{20}}", "-1/21 (unboxed)"]),
    ("74^\\circ", ["\\boxed{74^\\circ}", "\\boxed{74}", "\\boxed{74°}", "\\boxed{74^{\\circ}}"],
     ["\\boxed{75^\\circ}", "\\boxed{106°}"]),
    ("(3+\\sqrt{6})^{-1/3}", ["\\boxed{\\frac{1}{\\sqrt[3]{3+\\sqrt{6}}}}"], ["\\boxed{(3+\\sqrt{6})^{1/3}}"]),
    ("\\frac{\\sqrt{1740}}{3}", ["\\boxed{\\frac{2\\sqrt{435}}{3}}"], ["\\boxed{\\frac{\\sqrt{1740}}{2}}"]),
    ("420261", ["\\boxed{420261}", "\\boxed{420,261}"], ["\\boxed{420262}"]),
    ("3^{2025}", ["\\boxed{3^{2025}}"], ["\\boxed{3^{2024}}"]),
]


@pytest.mark.parametrize("gold,right,wrong", HMMT_GOLD_RIGHT_WRONG)
def test_hmmt_grades_latex_answers(gold, right, wrong):
    hmmt = module("hmmt_feb_2026")
    for answer in right:
        assert hmmt.score(f"Some work. {answer}", gold) == 1.0, answer
    for answer in wrong:
        assert hmmt.score(f"Some work. {answer}", gold) == 0.0, answer
    assert hmmt.score(None, gold) == 0.0


def test_hmmt_tasks_are_set_problems(monkeypatch):
    hmmt = module("hmmt_feb_2026")
    rows = [{"problem_idx": 2, "problem": "Find $x$.", "answer": "48"},
            {"problem_idx": 1, "problem": "Compute $y$.", "answer": "-\\frac{1}{21}"}]
    monkeypatch.setattr(hmmt, "read_rows", lambda revision: rows)
    tasks = list(hmmt.HMMTFeb2026Taskset(hmmt.HMMTConfig(id="hmmt-feb-2026")).load())
    assert [t.data.name for t in tasks] == ["hmmt-feb-2026-1", "hmmt-feb-2026-2"]
    assert vs.freeze(tasks[0]).prompt == hmmt.INSTRUCTION + "Compute $y$."
    assert scored(tasks[0], "so \\boxed{-\\frac{1}{21}}") == 1.0
    assert scored(tasks[0], "so \\boxed{\\frac{1}{21}}") == 0.0
    notes = vs.card_notes(hmmt.HMMTFeb2026Taskset(hmmt.HMMTConfig(id="x")))
    assert notes["license"] == "CC-BY-NC-SA-4.0" and "non-commercial" in notes["license_note"].lower()


# --- GraphWalks ---------------------------------------------------------------

def test_graphwalks_set_f1_is_upstreams():
    gw = module("graphwalks_single_turn")
    gold = ["a1", "b2", "c3"]
    assert gw.set_f1("thinking...\nFinal Answer: [a1, b2, c3]", gold) == 1.0
    assert gw.set_f1("Final Answer: [c3, a1, b2, a1]\n", gold) == 1.0   # a set; trailing newline
    assert gw.set_f1("Final Answer: [a1, b2]", gold) == pytest.approx(0.8)
    assert gw.set_f1("Final Answer: [a1, b2, c3, d4]", gold) == pytest.approx(6 / 7)
    assert gw.set_f1("Final Answer: [x9]", gold) == 0.0
    assert gw.set_f1("Final Answer: [a1, b2, c3]\nhope that helps", gold) == 0.0   # not the last line
    assert gw.set_f1("[a1, b2, c3]", gold) == 0.0                                 # no "Final Answer:"
    assert gw.set_f1("Final Answer: []", []) == 1.0
    assert gw.set_f1("Final Answer: [a1]", []) == 0.0
    assert gw.set_f1(None, gold) == 0.0


def test_graphwalks_tasks_filter_and_score(monkeypatch):
    gw = module("graphwalks_single_turn")
    rows = [{"prompt": "graph A", "answer_nodes": ["n1"], "prompt_chars": 7, "problem_type": "bfs"},
            {"prompt": "graph B" * 10, "answer_nodes": ["n2", "n3"], "prompt_chars": 70,
             "problem_type": "parents"}]
    monkeypatch.setattr(gw, "read_rows", lambda revision: rows)
    every = list(gw.GraphWalksSingleTurnTaskset(gw.GraphWalksConfig(id="x")).load())
    assert [t.data.idx for t in every] == [0, 1]
    short = list(gw.GraphWalksSingleTurnTaskset(gw.GraphWalksConfig(id="x", max_prompt_chars=10)).load())
    parents = list(gw.GraphWalksSingleTurnTaskset(gw.GraphWalksConfig(id="x", problem_type="parents")).load())
    assert [t.data.idx for t in short] == [0] and [t.data.idx for t in parents] == [1]
    assert vs.freeze(every[0]).prompt == "graph A"
    assert scored(every[1], "Final Answer: [n3, n2]") == 1.0
    assert scored(every[1], "Final Answer: [n3]") == pytest.approx(2 / 3)


# --- MRCR ---------------------------------------------------------------------

def test_mrcr_grade_is_upstreams():
    mrcr = module("openai_mrcr_single_turn")
    answer = "abc123XYZ0A poem about tapirs\nsecond line"
    prefix = "abc123XYZ0"
    assert mrcr.grade(answer, answer, prefix) == 1.0
    assert mrcr.grade("\n" + answer + "\n", answer, prefix) == 1.0
    assert mrcr.grade(answer.strip(), answer + "  \n", prefix) == 1.0   # gold answers end in whitespace
    assert mrcr.grade("A poem about tapirs\nsecond line", answer, prefix) == 0.0   # no hash
    assert mrcr.grade("Sure! " + answer, answer, prefix) == 0.0                    # hash not first
    partial = mrcr.grade(prefix + "A poem about rocks\nsecond line", answer, prefix)
    assert 0.5 < partial < 1.0
    assert mrcr.grade(None, answer, prefix) == 0.0


def test_mrcr_renders_the_conversation_as_one_turn():
    mrcr = module("openai_mrcr_single_turn")
    messages = [{"role": "user", "content": "examples"}, {"role": "user", "content": "write a poem"},
                {"role": "assistant", "content": "a poem"},
                {"role": "user", "content": "Prepend h to the 1st poem."}]
    text = mrcr.render(messages)
    assert text == (mrcr.PREAMBLE + "User: examples\n\nUser: write a poem\n\nAssistant: a poem"
                    + mrcr.REQUEST + "Prepend h to the 1st poem.")
    with pytest.raises(ValueError):
        mrcr.render(messages[:-1])
    assert [mrcr.context_bin(n) for n in (100, 4096, 4097, 131072, 131073, 2_000_000)] == \
        ["(0,4096]", "(0,4096]", "(4096,8192]", "(65536,131072]", "(131072,262144]", ">1048576"]


def test_mrcr_tasks_keep_rows_within_the_context_bound(monkeypatch):
    mrcr = module("openai_mrcr_single_turn")

    def row(n, words):
        return {"prompt": json.dumps([{"role": "user", "content": "w " * words},
                                      {"role": "user", "content": f"Prepend h{n} to it."}]),
                "answer": f"h{n}text", "random_string_to_prepend": f"h{n}",
                "desired_msg_index": 1, "total_messages": 2}

    monkeypatch.setattr(mrcr, "read_rows", lambda revision, needles: iter([row(0, 10), row(1, 5000)]))
    monkeypatch.setattr(mrcr, "o200k_tokens",
                        lambda messages, answer: sum(len(m["content"].split()) for m in messages))
    tasks = list(mrcr.OpenAIMRCRSingleTurnTaskset(mrcr.MRCRConfig(id="x", max_context_tokens=100)).load())
    assert [t.data.random_string_to_prepend for t in tasks] == ["h0"]
    assert tasks[0].data.context_bin == "(0,4096]"
    assert scored(tasks[0], "h0text") == 1.0
    assert scored(tasks[0], "text") == 0.0


# --- HLE-Diamond (text, deterministic subset) --------------------------------

@pytest.mark.parametrize("answer_type,gold,kind", [
    ("multipleChoice", "C", "letter"), ("multipleChoice", "C and D", None),
    ("exactMatch", "42", "number"), ("exactMatch", "-0.25", "number"),
    ("exactMatch", "Mercury", "word"), ("exactMatch", "anti-Markovnikov", "word"),
    ("exactMatch", "1/2", None), ("exactMatch", "$\\sqrt{2}$", None), ("exactMatch", "the red one", None),
    ("exactMatch", "1,000", None), ("exactMatch", "", None), ("other", "x", None),
])
def test_hle_qualifies_only_rule_gradable_answers(answer_type, gold, kind):
    assert module("hle_diamond_text_det").kind_of(answer_type, gold) == kind


def test_hle_scores_letters_numbers_and_words():
    hle = module("hle_diamond_text_det")
    letter = lambda body: hle.score(f"Explanation: because\n{body}\nConfidence: 80%", "C", "letter")  # noqa: E731
    for right in ("Answer: C", "Answer: (C)", "**Answer:** C. The third", "answer: c)"):
        assert letter(right) == 1.0, right
    for wrong in ("Answer: B", "Answer: The answer is C", "Exact Answer: C", "C"):
        assert letter(wrong) == 0.0, wrong
    assert hle.score("Answer: A\nAnswer: C", "C", "letter") == 1.0   # the last answer line counts

    number = lambda value: hle.score(f"Exact Answer: {value}", "0.25", "number")  # noqa: E731
    for right in ("0.25", "0.250", "$0.25$", "0.25.", "**0.25**"):
        assert number(right) == 1.0, right
    for wrong in ("1/4", "0.26", "0.25 m", "25%", ""):
        assert number(wrong) == 0.0, wrong
    assert hle.score("Exact Answer: 12,000", "12000", "number") == 1.0
    assert hle.score("Answer: 0.25", "0.25", "number") == 0.0   # the exact-answer line is required

    word = lambda value: hle.score(f"Exact Answer: {value}", "Mercury", "word")  # noqa: E731
    for right in ("Mercury", "mercury", "\"Mercury\"", "Mercury."):
        assert word(right) == 1.0, right
    for wrong in ("Venus", "Planet Mercury", "Mercury (Hg)"):
        assert word(wrong) == 0.0, wrong


def test_hle_tasks_are_text_only_and_rule_graded(monkeypatch):
    hle = module("hle_diamond_text_det")
    rows = [
        {"id": "a", "question": "Pick one: A) x B) y", "image": "", "answer": "B",
         "answer_type": "multipleChoice", "category": "Math", "partition": "Reasoning"},
        {"id": "b", "question": "With a figure", "image": "data:image/png;base64,xx", "answer": "A",
         "answer_type": "multipleChoice", "category": "Math", "partition": "Reasoning"},
        {"id": "c", "question": "How many?", "image": None, "answer": "7",
         "answer_type": "exactMatch", "category": "Bio", "partition": "Knowledge"},
        {"id": "d", "question": "Describe it", "image": None, "answer": "a long free answer",
         "answer_type": "exactMatch", "category": "Bio", "partition": "Knowledge"},
    ]
    monkeypatch.setattr(hle, "read_rows", lambda revision: rows)
    tasks = list(hle.HLEDiamondTextDetTaskset(hle.HLEConfig(id="x")).load())
    assert [t.data.name for t in tasks] == ["a", "c"]
    frozen = vs.freeze(tasks[0])
    assert frozen.system == hle.SYSTEM_MC and frozen.prompt == "Pick one: A) x B) y"
    assert vs.freeze(tasks[1]).system == hle.SYSTEM_EXACT
    assert scored(tasks[0], "Explanation: y\nAnswer: B\nConfidence: 90%") == 1.0
    assert scored(tasks[0], "Explanation: x\nAnswer: A\nConfidence: 90%") == 0.0
    assert scored(tasks[1], "Exact Answer: 7") == 1.0 and scored(tasks[1], "Exact Answer: 8") == 0.0
    knowledge = list(hle.HLEDiamondTextDetTaskset(hle.HLEConfig(id="x", partition="knowledge")).load())
    assert [t.data.name for t in knowledge] == ["c"]
