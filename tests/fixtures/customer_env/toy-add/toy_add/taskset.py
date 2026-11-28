"""Ten additions, scored by the exact sum after the reasoning."""
from collections.abc import Iterator

import verifiers.v1 as vf


class AddData(vf.TaskData):
    network_allow: list[str] = []
    answer: str


class AddTask(vf.Task[AddData, vf.State, vf.TaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return 1.0 if (trace.last_reply or "").strip() == self.data.answer else 0.0


class ToyAddTaskset(vf.Taskset[AddTask, vf.TasksetConfig]):
    def load(self) -> Iterator[AddTask]:
        for i in range(10):
            yield AddTask(AddData(idx=i, name=f"add-{i}", prompt=f"What is {i} + {i}? Answer with the number only.",
                                  answer=str(2 * i)), vf.TaskConfig())
