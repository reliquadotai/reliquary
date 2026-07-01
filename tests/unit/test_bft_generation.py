import torch

from reliquary.miner.engine import _bft_assemble_rollouts


class _Phase2Model:
    """Returns a fixed phase-2 tensor (the forced answer generation)."""

    device = "cpu"

    def __init__(self, phase2):
        self._phase2 = phase2
        self.calls = 0
        self.batches = []

    def generate(self, batch, **kw):
        self.calls += 1
        self.batches.append(batch.detach().cpu().tolist())
        return self._phase2


def test_bft_injects_force_only_for_unfinished_rows():
    prompt = [1, 1]
    think_close = {248069}
    force_ids = [248069, 7, 8]  # </think> + tail stub
    eos = {99}
    # row0 finished thinking (has 248069) then answered + EOS;
    # row1 never closed </think> within budget → must be forced.
    phase1 = torch.tensor([
        [1, 1, 5, 248069, 42, 99],   # finished
        [1, 1, 5, 6, 7, 8],          # still thinking at budget
    ])
    # phase-2 runs on the one unfinished row (force appended, len 9), then answer:
    phase2 = torch.tensor([[1, 1, 5, 6, 7, 8, 248069, 7, 8, 55, 99]])
    model = _Phase2Model(phase2)

    rollouts = _bft_assemble_rollouts(
        model=model,
        phase1_tensor=phase1,
        prompt_tokens=prompt,
        think_close_ids=think_close,
        force_ids=force_ids,
        eos_ids=eos,
        answer_budget=8,
    )

    # finished row: untouched, truncated at its EOS, not forced
    assert rollouts[0]["forced"] is False
    assert rollouts[0]["tokens"] == [1, 1, 5, 248069, 42, 99]
    # forced row: force span injected, answer kept after it
    assert rollouts[1]["forced"] is True
    start, end = rollouts[1]["force_span"]
    assert rollouts[1]["tokens"][start:end] == force_ids
    assert rollouts[1]["tokens"][end:] == [55, 99]
    # phase-2 generation ran exactly once (only for the unfinished row)
    assert model.calls == 1


def test_bft_does_not_force_eos_without_think_close():
    # row terminated on EOS (99) but never closed </think> (777): malformed-but-
    # finished → must NOT be forced (else FORCE injected after EOS, and the
    # validator's force-position check would reject an honest rollout).
    phase1 = torch.tensor([[1, 1, 5, 6, 99, 0, 0]])  # EOS at idx 4, then pad
    model = _Phase2Model(torch.empty(0))

    rollouts = _bft_assemble_rollouts(
        model=model,
        phase1_tensor=phase1,
        prompt_tokens=[1, 1],
        think_close_ids={777},
        force_ids=[777, 7, 8],
        eos_ids={99},
        answer_budget=8,
    )

    assert rollouts[0]["forced"] is False
    assert rollouts[0]["tokens"] == [1, 1, 5, 6, 99]  # trimmed at EOS, no force
    assert model.calls == 0


def test_bft_continues_natural_close_without_eos():
    # Natural </think> before EOS means the thinking phase is done, but the
    # rollout still needs its answer budget. It must continue without injecting
    # the force template and without declaring a force_span.
    prompt = [1, 1]
    think_close = {777}
    force_ids = [777, 7, 8]
    eos = {99}
    phase1 = torch.tensor([[1, 1, 5, 777, 42, 43]])
    phase2 = torch.tensor([[1, 1, 5, 777, 42, 43, 55, 99]])
    model = _Phase2Model(phase2)

    rollouts = _bft_assemble_rollouts(
        model=model,
        phase1_tensor=phase1,
        prompt_tokens=prompt,
        think_close_ids=think_close,
        force_ids=force_ids,
        eos_ids=eos,
        answer_budget=8,
    )

    assert model.calls == 1
    assert model.batches == [[[1, 1, 5, 777, 42, 43]]]
    assert rollouts[0]["forced"] is False
    assert "force_span" not in rollouts[0]
    assert rollouts[0]["tokens"] == [1, 1, 5, 777, 42, 43, 55, 99]


def test_bft_no_force_when_all_rows_finished():
    prompt = [1, 1]
    phase1 = torch.tensor([
        [1, 1, 5, 248069, 42, 99],
        [1, 1, 6, 248069, 43, 99],
    ])
    model = _Phase2Model(torch.empty(0))

    rollouts = _bft_assemble_rollouts(
        model=model,
        phase1_tensor=phase1,
        prompt_tokens=prompt,
        think_close_ids={248069},
        force_ids=[248069, 7, 8],
        eos_ids={99},
        answer_budget=8,
    )

    assert all(r["forced"] is False for r in rollouts)
    assert "force_span" not in rollouts[0]
    # no unfinished rows → no phase-2 generation
    assert model.calls == 0
