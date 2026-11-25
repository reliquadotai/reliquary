"""openai-mrcr-single-turn: OpenAI MRCR (openai/mrcr), the conversation as one user turn.

Upstream sends a row's conversation as chat messages; an eval set is one user
turn, so every message but the last is written as a transcript ("User: ..." /
"Assistant: ..."), followed by the last user message (the "Prepend <hash> to the
Nth ..." request). This is the text-style rendering MRCR v2 itself uses, applied
to OpenAI's data: scores are comparable across models on this set, not equal to
OpenAI's multi-message numbers.

Graded as upstream's README: 0 unless the reply starts with the row's
``random_string_to_prepend``; otherwise the ``difflib.SequenceMatcher`` ratio of
the reply and the answer, both without that prefix and stripped ("the stripped
sampled answer is compared to the stripped ground truth answer"); the reply is
also stripped before the prefix check (a reasoning block's trailing newline).

Rows are selected by upstream's bin rule: OpenAI ``o200k_base`` tokens of every
message plus the answer, kept when at most ``max_context_tokens`` (default 131072:
the bins 4k-8k ... 64k-128k, 500 rows for 8 needles).
"""

import json
from collections.abc import Iterator
from difflib import SequenceMatcher
from typing import Literal

import verifiers.v1 as vf

DATASET = "openai/mrcr"
DATASET_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
FILES = {n: (f"{n}needle/{n}needle_0.parquet", f"{n}needle/{n}needle_1.parquet") for n in (2, 4, 8)}
BINS = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
PREAMBLE = ("Below is the transcript of an earlier conversation between a user and you, the "
            "assistant. Each turn starts with \"User:\" or \"Assistant:\".\n\n")
REQUEST = "\n\n=== END OF TRANSCRIPT ===\n\n"


def grade(response: str | None, answer: str, prefix: str) -> float:
    """Upstream's grade, on stripped text."""
    response = (response or "").strip()
    if not response.startswith(prefix):
        return 0.0
    return float(SequenceMatcher(None, response.removeprefix(prefix).strip(),
                                 answer.removeprefix(prefix).strip()).ratio())


def render(messages: list[dict]) -> str:
    """Every message but the last as a transcript, then the last (a user request)."""
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("an MRCR conversation ends with the user's request")
    labels = {"user": "User", "assistant": "Assistant"}
    turns = [f"{labels[m['role']]}: {m['content']}" for m in messages[:-1]]
    return PREAMBLE + "\n\n".join(turns) + REQUEST + messages[-1]["content"]


def context_bin(tokens: int) -> str:
    high = next((b for b in BINS if tokens <= b), None)
    if high is None:
        return f">{BINS[-1]}"
    low = BINS[BINS.index(high) - 1] if high != BINS[0] else 0
    return f"({low},{high}]"


class MRCRData(vf.TaskData):
    network_allow: list[str] = []
    answer: str
    random_string_to_prepend: str
    context_tokens_o200k: int
    """Upstream's bin measure: o200k_base tokens of every message plus the answer."""
    context_bin: str
    desired_msg_index: int
    total_messages: int


class MRCRTaskConfig(vf.TaskConfig):
    pass


class MRCRTask(vf.Task[MRCRData, vf.State, MRCRTaskConfig]):
    @vf.reward(weight=1.0)
    async def ratio(self, trace: vf.Trace) -> float:
        return grade(trace.last_reply, self.data.answer, self.data.random_string_to_prepend)


class MRCRConfig(vf.TasksetConfig):
    dataset_revision: str = DATASET_REVISION
    needles: Literal[2, 4, 8] = 8
    max_context_tokens: int = 131072
    min_context_tokens: int = 0
    task: MRCRTaskConfig = MRCRTaskConfig()


COLUMNS = ["prompt", "answer", "random_string_to_prepend", "desired_msg_index", "total_messages"]


def read_rows(revision: str, needles: int) -> Iterator[dict]:
    """Every row of the needle count's files, in file order, one batch at a time."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for name in FILES[needles]:
        path = hf_hub_download(DATASET, name, repo_type="dataset", revision=revision)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=COLUMNS):
            yield from batch.to_pylist()


def o200k_tokens(messages: list[dict], answer: str) -> int:
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    return sum(len(encoding.encode_ordinary(m["content"])) for m in messages) \
        + len(encoding.encode_ordinary(answer))


class OpenAIMRCRSingleTurnTaskset(vf.Taskset[MRCRTask, MRCRConfig]):
    RELIQUARY_CARD = {
        "name": "OpenAI MRCR 8-needle (≤128k, transcript in one turn)",
        "license": "MIT",
        "grading": "hash-prefix gate + difflib.SequenceMatcher ratio (upstream)",
        "context_note": "prompts reach 128k OpenAI tokens: a model must serve the set's max prompt tokens",
    }

    def load(self) -> Iterator[MRCRTask]:
        config = self.config
        for i, row in enumerate(read_rows(config.dataset_revision, config.needles)):
            messages = json.loads(row["prompt"])
            tokens = o200k_tokens(messages, row["answer"])
            if not config.min_context_tokens < tokens <= config.max_context_tokens:
                continue
            yield MRCRTask(
                MRCRData(idx=i, name=f"mrcr-{config.needles}needle-{row['random_string_to_prepend']}",
                         prompt=render(messages), answer=row["answer"],
                         random_string_to_prepend=row["random_string_to_prepend"],
                         context_tokens_o200k=tokens, context_bin=context_bin(tokens),
                         desired_msg_index=int(row["desired_msg_index"]),
                         total_messages=int(row["total_messages"])),
                config.task,
            )
