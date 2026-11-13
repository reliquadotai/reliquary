"""graphwalks-single-turn: OpenAI GraphWalks (openai/graphwalks), as OpenAI runs it.

The row's prompt (3-shot examples, the edge list, the operation) is the one user
turn; no tools. Graded as upstream's README: the reply's last line must carry
``Final Answer: [...]``; the score is the F1 of that node set against the gold set
(1.0 when both are empty, 0.0 when the line does not parse). Only difference:
trailing whitespace of the reply is dropped before its last line is read.

Only ``graphwalks_128k_and_shorter.parquet`` (750 rows, prompts up to 128k
OpenAI tokens) is read; the 256k-1M file is never downloaded.
"""

import re
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf

DATASET = "openai/graphwalks"
DATASET_FILE = "graphwalks_128k_and_shorter.parquet"
DATASET_REVISION = "be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7"


def get_list(response: str) -> tuple[list[str], bool]:
    """Upstream's extraction: (nodes, failed_to_parse) from the reply's last line."""
    line = response.rstrip().split("\n")[-1]
    if "Final Answer:" not in line:
        return [], True
    list_part = re.search(r"\[.*\]", line)
    if not list_part:
        return [], True
    items = list_part.group(0).strip("[]").split(",")
    return [item.strip() for item in items if item.strip()], False


def set_f1(response: str | None, gold: list[str]) -> float:
    sampled, failed = get_list(response or "")
    if failed:
        return 0.0
    sampled_set, truth = set(sampled), set(gold)
    if not sampled_set and not truth:
        return 1.0
    overlap = len(sampled_set & truth)
    recall = overlap / len(truth) if truth else 0.0
    precision = overlap / len(sampled_set) if sampled_set else 0.0
    return 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0


class GraphWalksData(vf.TaskData):
    network_allow: list[str] = []
    answer_nodes: list[str]
    problem_type: str
    prompt_chars: int


class GraphWalksTaskConfig(vf.TaskConfig):
    pass


class GraphWalksTask(vf.Task[GraphWalksData, vf.State, GraphWalksTaskConfig]):
    @vf.reward(weight=1.0)
    async def f1(self, trace: vf.Trace) -> float:
        return set_f1(trace.last_reply, list(self.data.answer_nodes))


class GraphWalksConfig(vf.TasksetConfig):
    dataset_revision: str = DATASET_REVISION
    problem_type: Literal["bfs", "parents"] | None = None
    """Keep only this operation (None: both)."""
    max_prompt_chars: int | None = None
    """Keep only prompts of at most this many characters (None: the whole file)."""
    task: GraphWalksTaskConfig = GraphWalksTaskConfig()


def read_rows(revision: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset", revision=revision)
    return pq.read_table(path, columns=["prompt", "answer_nodes", "prompt_chars",
                                        "problem_type"]).to_pylist()


class GraphWalksSingleTurnTaskset(vf.Taskset[GraphWalksTask, GraphWalksConfig]):
    RELIQUARY_CARD = {
        "name": "GraphWalks (≤128k)",
        "license": "MIT",
        "grading": "set F1 of the reply's last 'Final Answer: [...]' line against the gold nodes (upstream)",
        "context_note": "prompts reach 128k OpenAI tokens: a model must serve the set's max prompt tokens",
    }

    def load(self) -> Iterator[GraphWalksTask]:
        config = self.config
        for i, row in enumerate(read_rows(config.dataset_revision)):
            if config.problem_type is not None and row["problem_type"] != config.problem_type:
                continue
            if config.max_prompt_chars is not None and row["prompt_chars"] > config.max_prompt_chars:
                continue
            yield GraphWalksTask(
                GraphWalksData(idx=i, prompt=row["prompt"],
                               answer_nodes=[str(n) for n in row["answer_nodes"]],
                               problem_type=row["problem_type"],
                               prompt_chars=int(row["prompt_chars"])),
                config.task,
            )
