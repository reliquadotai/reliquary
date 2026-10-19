"""A job's rows as tasks: the one place a ``prompt_index`` becomes ``(split, index)``.

A job's ``prompt_index`` lives in row space everywhere: its episode seed, the
slots it owns, walk order and payment. Only the task a row runs is looked up
through a ``RowMap``. A training job's map is the identity over
``[prompt_start, prompt_start + prompt_count)`` on its split (``train``), so
nothing about it changes. An eval job's prompt source names a frozen set of
row references (``reliquary.eval.prompt_source.reference_of``): row ``i`` runs
the source row the set's ``i``-th reference names, at the set's one split, and
must rebuild into the task the reference says (its ``instance_id``).

Maps are built once per ``(prompt_source, split, start, count)`` and shared by
every caller in the process: the validator's intake, the grader, the export
and the miner all ask for the same job's map on every row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import functools

from reliquary.eval.prompt_source import (
    is_eval_source, load_eval_rows, parse_eval_source, reference_of,
)


@dataclass(frozen=True)
class TaskRef:
    """The task one row runs: a source row of ``split``, and the instance it must
    rebuild into (None for the identity map: the row is the task)."""

    split: str
    index: int
    instance_id: str | None


class RowMap:
    """Row ``prompt_index`` of a job -> the task it runs. Either the identity over
    ``[start, start + count)`` or a tuple of ``(index, instance_id)``, one per row
    from 0."""

    __slots__ = ("_split", "_start", "_count", "_refs")

    def __init__(self, split: str, start: int, count: int,
                 refs: tuple[tuple[int, str], ...] | None) -> None:
        self._split = split
        self._start = start
        self._count = count
        self._refs = refs

    @classmethod
    def identity(cls, split: str, start: int, count: int) -> "RowMap":
        """Row ``i`` runs source row ``i`` of ``split``, for ``start <= i < start + count``."""
        return cls(split, start, count, None)

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping]) -> "RowMap":
        """Row ``i`` runs what reference row ``rows[i]`` names. Every row must be a
        well-formed reference, and all of one split: a job runs one split."""
        refs = [reference_of(row) for row in rows]
        for i, ref in enumerate(refs):
            if ref is None:
                raise ValueError(f"row {i} is not a well-formed row reference "
                                 "(split, source_index, instance_id)")
        splits = {split for split, _, _ in refs}
        if len(splits) != 1:
            raise ValueError(f"a job's row references name one split, not {sorted(splits)}")
        return cls(splits.pop(), 0, len(refs),
                   tuple((index, instance_id) for _, index, instance_id in refs))

    @property
    def split(self) -> str:
        """Read-only: one map is shared by every caller in the process."""
        return self._split

    @property
    def is_identity(self) -> bool:
        """Row ``i`` is source row ``i`` (a training job): nothing to look up or check."""
        return self._refs is None

    def __len__(self) -> int:
        return self._count

    def ref(self, prompt_index: int) -> TaskRef:
        """The task row ``prompt_index`` runs; IndexError outside the map."""
        if type(prompt_index) is not int:
            raise IndexError(f"row {prompt_index!r} is not a row number")
        offset = prompt_index - self._start
        if not 0 <= offset < self._count:
            raise IndexError(f"row {prompt_index!r} is outside the job's rows "
                             f"[{self._start}, {self._start + self._count})")
        if self._refs is None:
            return TaskRef(self.split, prompt_index, None)
        index, instance_id = self._refs[offset]
        return TaskRef(self.split, index, instance_id)

    def index(self, prompt_index: int) -> int:
        return self.ref(prompt_index).index

    def check_instances(self, instance_id: Callable[[str, int], str]) -> None:
        """That every reference rebuilds into the instance it names:
        ``instance_id(split, index)`` is the environment's id for a source row.
        ValueError naming the first row that does not; nothing to check for the
        identity map."""
        if self._refs is None:
            return
        for row, (index, expected) in enumerate(self._refs):
            actual = instance_id(self.split, index)
            if actual != expected:
                raise ValueError(f"row {row} names instance {expected!r}, but {self.split} "
                                 f"row {index} is {actual!r}")

    def __repr__(self) -> str:
        kind = "identity" if self._refs is None else "references"
        return (f"RowMap({kind}, split={self.split!r}, "
                f"rows=[{self._start}, {self._start + self._count}))")


@functools.lru_cache(maxsize=64)
def _row_map(prompt_source: str, split: str, start: int, count: int) -> RowMap:
    if not is_eval_source(prompt_source):
        return RowMap.identity(split, start, count)
    source = parse_eval_source(prompt_source)
    if count != source.count:
        raise ValueError(f"the job's prompt count {count} is not its eval source's "
                         f"count {source.count}")
    if start != 0:
        raise ValueError(f"an eval job's rows start at 0, not {start}")
    mapped = RowMap.from_rows(load_eval_rows(source))
    if mapped.split != split:
        raise ValueError(f"eval set {source.set_id!r} references split {mapped.split!r}, "
                         f"but the job runs split {split!r}")
    return mapped


def row_map(prompt_source: str, *, split: str, start: int, count: int) -> RowMap:
    """The map of a job's rows: an eval source's references (its split must be
    ``split``, its count ``count``), anything else the identity over
    ``[start, start + count)`` of ``split``. Memoised: one map per process."""
    return _row_map(prompt_source, split, start, count)


def job_row_map(job) -> RowMap:
    """The map of ``job``'s rows, at its episode's split (``train`` without one)."""
    episode = getattr(job, "episode", None)
    split = episode.env.split if episode is not None else "train"
    return row_map(job.prompt_source, split=split, start=job.prompt_start,
                   count=job.prompt_count)
