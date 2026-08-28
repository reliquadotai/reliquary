"""_load_checkpoint: reload both hf_model and vllm_model from a local path."""

from unittest.mock import MagicMock, patch

import pytest


def _make_hf_mock(name):
    m = MagicMock(name=name)
    m.to.return_value = m
    m.eval.return_value = m
    return m


@pytest.fixture
def mock_engine():
    """Build a MiningEngine with mock models so we can observe reload calls."""
    from reliquary.miner.engine import MiningEngine

    # MiningEngine.__init__ does real work (HF imports, GRAIL init). Bypass
    # with object.__new__ and manually set attrs we need.
    eng = object.__new__(MiningEngine)
    eng.vllm_model = MagicMock(name="initial_vllm")
    eng.hf_model = MagicMock(name="initial_hf")
    eng.vllm_gpu = 0
    eng.proof_gpu = 1
    return eng


def test_load_checkpoint_swaps_both_models(mock_engine):
    """After successful reload, both hf_model and vllm_model are swapped."""
    mock_hf = _make_hf_mock("new_hf")
    mock_gen = _make_hf_mock("new_gen")

    with patch("reliquary.shared.modeling.load_text_generation_model",
               side_effect=[mock_hf, mock_gen]):
        result = mock_engine._load_checkpoint("/tmp/checkpoint-5")

    assert mock_engine.hf_model is mock_hf
    assert mock_engine.vllm_model is mock_gen
    assert result is mock_hf


def test_load_checkpoint_short_circuits_on_same_path(mock_engine):
    """Calling twice with the same path doesn't reload."""
    mock_hf = _make_hf_mock("new_hf")
    mock_gen = _make_hf_mock("new_gen")

    with patch("reliquary.shared.modeling.load_text_generation_model",
               side_effect=[mock_hf, mock_gen]) as mock_from_pretrained:
        mock_engine._load_checkpoint("/tmp/checkpoint-5")
        mock_engine._load_checkpoint("/tmp/checkpoint-5")

    # from_pretrained should have been called exactly twice (once per model, first call only)
    assert mock_from_pretrained.call_count == 2


def test_load_checkpoint_hf_load_failure_keeps_old_models(mock_engine):
    """If the shared model loader raises on hf load, old models stay."""
    original_hf = mock_engine.hf_model
    original_vllm = mock_engine.vllm_model

    with patch("reliquary.shared.modeling.load_text_generation_model",
               side_effect=RuntimeError("HF load failed")):
        with pytest.raises(RuntimeError, match="HF load failed"):
            mock_engine._load_checkpoint("/tmp/bad")

    assert mock_engine.hf_model is original_hf
    assert mock_engine.vllm_model is original_vllm


def test_load_checkpoint_vllm_load_failure_keeps_atomic_old_pair(mock_engine):
    """A failed second stage never publishes a mixed checkpoint pair."""
    mock_hf = _make_hf_mock("new_hf")
    original_hf = mock_engine.hf_model
    original_vllm = mock_engine.vllm_model

    with patch("reliquary.shared.modeling.load_text_generation_model",
               side_effect=[mock_hf, RuntimeError("gen GPU OOM")]):
        with pytest.raises(RuntimeError, match="gen GPU OOM"):
            mock_engine._load_checkpoint("/tmp/vllm_broken")

    assert mock_engine.hf_model is original_hf
    assert mock_engine.vllm_model is original_vllm
    assert getattr(mock_engine, "_loaded_checkpoint_path", None) is None
