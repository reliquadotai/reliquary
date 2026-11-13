"""hle-diamond-text-det: HLE-Diamond (cais/hle-diamond), text-only, deterministic subset.

HLE grades with an LLM judge; a judge's verdict is not a fact two gradings both
reproduce, so only questions whose answer can be checked by a rule are kept:

* ``multipleChoice`` with a single-letter gold answer: the letter of the reply's
  last ``Answer:`` line must equal it;
* ``exactMatch`` whose gold is a plain number (``-12``, ``0.25``): the reply's
  last ``Exact Answer:`` value, as a decimal, must equal it;
* ``exactMatch`` whose gold is one word (letters, digits, ``-``, ``'``): the
  reply's value must equal it, case-insensitively.

Everything else (expressions, phrases, LaTeX, any image) is left out. Normalisation
is conservative: surrounding whitespace, ``$``/``**``/quotes and one trailing period
are dropped, and thousands separators in numbers. A correct answer written another
way (``1/4`` for ``0.25``, a synonym, units) scores 0: this subset can only
under-count, never accept a wrong answer.

The prompts are HLE's own response-format system prompts; the exact-answer one asks
for the bare value.
"""

import re
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from typing import Literal

import verifiers.v1 as vf

DATASET = "cais/hle-diamond"
DATASET_FILE = "data/test-00000-of-00001.parquet"
DATASET_REVISION = "04eeb7efa7e3e4f83a00cbd5ce436a38fd5dda23"

# centerforaisafety/hle's system prompts; the exact-answer one also asks for the bare value.
SYSTEM_MC = ("Your response should be in the following format:\n"
             "Explanation: {your explanation for your answer choice}\n"
             "Answer: {your chosen answer}\n"
             "Confidence: {your confidence score between 0% and 100% for your answer}")
SYSTEM_EXACT = ("Your response should be in the following format:\n"
                "Explanation: {your explanation for your final answer}\n"
                "Exact Answer: {your succinct, final answer}\n"
                "Confidence: {your confidence score between 0% and 100% for your answer}\n"
                "Give the exact answer as the bare value only: no units, words or LaTeX.")

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]{0,39}")
LETTER = re.compile(r"[A-Z]")
_ANSWER_LINE = re.compile(r"^\W*(exact answer|answer)\W*:\s*(.*)$", re.IGNORECASE)


def kind_of(answer_type: str, gold: str) -> str | None:
    """How a question is graded, or None when no rule can grade it."""
    gold = gold.strip()
    if answer_type == "multipleChoice":
        return "letter" if LETTER.fullmatch(gold) else None
    if answer_type == "exactMatch":
        if NUMBER.fullmatch(gold):
            return "number"
        if WORD.fullmatch(gold) and any(c.isalpha() for c in gold):
            return "word"
    return None


def final_value(reply: str | None, label: str) -> str | None:
    """The value of the reply's last ``<label>:`` line (``Answer`` or ``Exact Answer``)."""
    found = None
    for line in (reply or "").splitlines():
        match = _ANSWER_LINE.match(line.strip())
        if match and match.group(1).lower() == label.lower():
            found = match.group(2)
    return None if found is None else clean(found)


def clean(value: str) -> str:
    value = value.strip()
    for _ in range(3):
        before = value
        value = value.strip().strip("*").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "$\"'`":
            value = value[1:-1]
        value = value.removesuffix(".").strip()
        if value == before:
            break
    return value


def _decimal(text: str) -> Decimal | None:
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    if not NUMBER.fullmatch(text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def score(reply: str | None, gold: str, kind: str) -> float:
    gold = gold.strip()
    if kind == "letter":
        value = final_value(reply, "answer")
        if not value:
            return 0.0
        letter = re.match(r"^\(?([A-Za-z])\)?(?:$|[\s.:)\]])", value)
        return 1.0 if letter and letter.group(1).upper() == gold else 0.0
    value = final_value(reply, "exact answer")
    if not value:
        return 0.0
    if kind == "number":
        got = _decimal(value)
        return 1.0 if got is not None and got == Decimal(gold) else 0.0
    if kind == "word":
        return 1.0 if value.casefold() == gold.casefold() else 0.0
    raise ValueError(f"unknown kind {kind!r}")


class HLEData(vf.TaskData):
    network_allow: list[str] = []
    answer: str
    kind: Literal["letter", "number", "word"]
    answer_type: str
    category: str
    partition: str


class HLETaskConfig(vf.TaskConfig):
    pass


class HLETask(vf.Task[HLEData, vf.State, HLETaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return score(trace.last_reply, self.data.answer, self.data.kind)


class HLEConfig(vf.TasksetConfig):
    dataset_revision: str = DATASET_REVISION
    partition: Literal["reasoning", "knowledge"] | None = None
    task: HLETaskConfig = HLETaskConfig()


COLUMNS = ["id", "question", "image", "answer", "answer_type", "category", "partition"]


def read_rows(revision: str) -> list[dict]:
    """The gated parquet (HF_TOKEN of an account the dataset admits); images are not read."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset", revision=revision)
    return pq.read_table(path, columns=COLUMNS).to_pylist()


def qualifying(rows: list[dict], partition: str | None = None) -> Iterator[tuple[int, dict, str]]:
    for i, row in enumerate(rows):
        if row.get("image"):
            continue
        if partition is not None and str(row["partition"]).lower() != partition:
            continue
        kind = kind_of(str(row["answer_type"]), str(row["answer"]))
        if kind is not None:
            yield i, row, kind


class HLEDiamondTextDetTaskset(vf.Taskset[HLETask, HLEConfig]):
    RELIQUARY_CARD = {
        "name": "HLE-Diamond (text, deterministic subset)",
        "license": "MIT",
        "license_note": ("cais/hle-diamond is gated and asks not to share, re-upload or redistribute "
                         "it: prompts.jsonl goes to miners' pods, so serving it needs CAIS's agreement."),
        "grading": ("multiple-choice letter or exact number/one-word answer, conservative "
                    "normalisation; questions needing a judge are left out"),
    }

    def load(self) -> Iterator[HLETask]:
        config = self.config
        for i, row, kind in qualifying(read_rows(config.dataset_revision), config.partition):
            yield HLETask(
                HLEData(idx=i, name=str(row["id"]), prompt=str(row["question"]),
                        system_prompt=SYSTEM_MC if kind == "letter" else SYSTEM_EXACT,
                        answer=str(row["answer"]).strip(), kind=kind,
                        answer_type=str(row["answer_type"]), category=str(row["category"]),
                        partition=str(row["partition"]).lower()),
                config.task,
            )
