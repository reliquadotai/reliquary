"""A published checkpoint loaded into the audit model (reliquary/shared/modeling.load_checkpoint_state)."""
import pytest
import torch

from reliquary.shared.modeling import load_checkpoint_state


class Config:
    def __init__(self, tied):
        self.tie_word_embeddings = tied

    def get_text_config(self):
        return self


class Tiny(torch.nn.Module):
    _tied_weights_keys = {"lm_head.weight": "embed.weight"}
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]

    def __init__(self, tied):
        super().__init__()
        self.config = Config(tied)
        self.embed = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        self.tied = 0
        if tied:
            self.lm_head.weight = self.embed.weight

    def tie_weights(self):
        self.tied += 1
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight


def test_a_tied_model_loads_without_its_head_and_its_mtp_weights():
    model = Tiny(tied=True)
    load_checkpoint_state(model, {"embed.weight": torch.full((4, 2), 3.0), "mtp.fc.weight": torch.zeros(1)})
    assert float(model.lm_head.weight[0, 0]) == 3.0 and model.tied == 1


def test_an_untied_model_missing_its_head_is_refused():
    with pytest.raises(RuntimeError, match="lm_head.weight"):
        load_checkpoint_state(Tiny(tied=False), {"embed.weight": torch.zeros(4, 2)})


def test_an_unknown_extra_weight_is_refused():
    with pytest.raises(RuntimeError, match="extra.weight"):
        load_checkpoint_state(Tiny(tied=True), {"embed.weight": torch.zeros(4, 2), "extra.weight": torch.zeros(1)})


def test_an_untied_model_loads_its_full_state():
    model = Tiny(tied=False)
    load_checkpoint_state(model, {"embed.weight": torch.zeros(4, 2), "lm_head.weight": torch.ones(4, 2)})
    assert float(model.lm_head.weight[0, 0]) == 1.0
