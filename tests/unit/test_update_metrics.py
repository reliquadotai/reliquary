import pytest
import torch

from reliquary.trainer.ckpt_publisher import update_metrics


def test_update_metrics_reports_relative_norms_per_group_and_changed_share():
    old = {"model.layers.0.w": torch.ones(4, dtype=torch.bfloat16), "lm_head.weight": torch.full((2,), 2.0),
           "model.embed_tokens.weight": torch.ones(2), "step": torch.tensor([1])}
    new = {"model.layers.0.w": torch.tensor([1.0, 1.0, 1.0, 2.0], dtype=torch.bfloat16),
           "lm_head.weight": torch.full((2,), 2.0), "model.embed_tokens.weight": torch.ones(2), "step": torch.tensor([2])}
    m = update_metrics(old, new)
    assert m["async/update/rel_norm_layer_000"] == pytest.approx(0.5)            # ||(0,0,0,1)|| / ||(1,1,1,1)||
    assert m["async/update/rel_norm_lm_head"] == 0.0 and m["async/update/rel_norm_embed"] == 0.0
    assert m["async/update/rel_norm_all"] == pytest.approx((1 / 14) ** 0.5)      # 1 / (4 + 8 + 2)
    assert m["async/update/rel_norm_tensor_max"] == pytest.approx(0.5)
    assert m["async/update/changed_ratio"] == pytest.approx(1 / 8)                # integer tensors are skipped


def test_the_publisher_measures_the_update_after_publishing(tmp_path):
    from reliquary.trainer.ckpt_publisher import DeltaPublisher

    class Store:
        def __init__(self):
            self.blobs = {}

        def get_bytes(self, key):
            return self.blobs.get(key)

        def put_bytes(self, key, blob):
            self.blobs[key] = blob

        def put_file(self, key, path):
            self.blobs[key] = open(path, "rb").read()

        def delete(self, key):
            self.blobs.pop(key, None)

    pub = DeltaPublisher(run="r", store=Store(), staging_dir=tmp_path)
    pub.publish(0, {"model.layers.0.w": torch.ones(4, dtype=torch.bfloat16)})
    assert pub.measure_update() == {}                                         # nothing before the first delta
    pub.publish(1, {"model.layers.0.w": torch.tensor([1.0, 1.0, 1.0, 2.0], dtype=torch.bfloat16)})
    assert pub.measure_update()["async/update/rel_norm_all"] == pytest.approx(0.5)
    assert pub.measure_update() == {}                                         # measured once, then released
