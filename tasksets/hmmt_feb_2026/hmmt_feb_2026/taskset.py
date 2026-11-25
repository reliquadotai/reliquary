"""hmmt-feb-2026: HMMT February 2026 (MathArena/hmmt_feb_2026), 33 problems.

The aime26 pattern: the model reasons and boxes its answer, scored by math
equivalence (`math-verify`) of the last ``\\boxed{}`` against the gold LaTeX
answer (fractions, radicals, powers, ``74^\\circ``). A degree sign is read as
``^\\circ`` first: math-verify parses ``74^\\circ`` but not ``74°``.
"""

from collections.abc import Iterator

import verifiers.v1 as vf

DATASET = "MathArena/hmmt_feb_2026"
DATASET_FILE = "data/train-00000-of-00001.parquet"
DATASET_REVISION = "589a53fe152febc0f4f99f3e561007bf8da8e664"
INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"


def score(reply: str | None, gold: str, *, timeout_seconds: int = 5) -> float:
    if not reply:
        return 0.0
    return vf.verify_boxed_math_answer(reply.replace("°", "^{\\circ}"), gold,
                                       timeout_seconds=timeout_seconds)


class HMMTData(vf.TaskData):
    network_allow: list[str] = []
    answer: str
    """The gold answer, as MathArena's LaTeX."""


class HMMTTaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5


class HMMTTask(vf.Task[HMMTData, vf.State, HMMTTaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return score(trace.last_reply, self.data.answer,
                     timeout_seconds=self.config.math_verify_timeout)


class HMMTConfig(vf.TasksetConfig):
    dataset_revision: str = DATASET_REVISION
    task: HMMTTaskConfig = HMMTTaskConfig()


def read_rows(revision: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset", revision=revision)
    return pq.read_table(path, columns=["problem_idx", "problem", "answer"]).to_pylist()


class HMMTFeb2026Taskset(vf.Taskset[HMMTTask, HMMTConfig]):
    # Copied onto the eval-set card by `reliquary eval build-set`.
    RELIQUARY_CARD = {
        "name": "HMMT February 2026",
        "license": "CC-BY-NC-SA-4.0",
        "license_note": ("MathArena/hmmt_feb_2026 is CC BY-NC-SA 4.0 (non-commercial, share-alike): "
                         "commercial use of this set needs confirmation before it is offered."),
        "grading": "math-verify equivalence of the last \\boxed{} with the gold LaTeX answer",
    }

    def load(self) -> Iterator[HMMTTask]:
        rows = sorted(read_rows(self.config.dataset_revision), key=lambda r: int(r["problem_idx"]))
        for i, row in enumerate(rows):
            yield HMMTTask(
                HMMTData(idx=i, name=f"hmmt-feb-2026-{int(row['problem_idx'])}",
                         prompt=INSTRUCTION + row["problem"], answer=str(row["answer"]).strip()),
                self.config.task,
            )
