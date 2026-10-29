"""Public checkpoint admission without downloading model weights or using a GPU."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary.integrations import affine_owned_native_job as owned


def fixture(tmp_path):
    config = dict(architectures=["Qwen3_5ForConditionalGeneration"], text_config=dict(num_hidden_layers=32),
                  _name_or_path="original")
    files = {"config.json": json.dumps(config).encode(), "chat_template.jinja": b"frozen template",
             "tokenizer.json": b"frozen tokenizer", "generation_config.json": b'{"eos_token_id":[248046,248044]}',
             "model.safetensors.index.json": b'{"weight_map":{"a":"model-1.safetensors"}}'}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    rows = [SimpleNamespace(rfilename=n, size=len(b), lfs=None) for n,b in files.items()]
    rows.append(SimpleNamespace(rfilename="model-1.safetensors", size=100, lfs={"sha256":"a"*64}))
    info = SimpleNamespace(sha="b"*40, private=False, gated=False, siblings=rows)
    calls = []
    def download(repo, name, **kwargs):
        calls.append(name)
        assert kwargs == dict(revision="b"*40, token=False)
        assert not name.endswith(".safetensors")
        return str(tmp_path / name)
    baseline = dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()
                           if n != "model.safetensors.index.json"})
    api = SimpleNamespace(model_info=lambda *a,**kw:info)
    return config, baseline, info, api, download, calls


def discover(tmp_path, mutate=lambda *_:None):
    config, baseline, info, api, download, calls = fixture(tmp_path)
    mutate(config, baseline, info)
    value = owned.discover_checkpoint("owner/model", "b"*40, baseline, config, api=api, download=download)
    return value, calls


def test_pins_exact_public_revision_without_downloading_weights(tmp_path):
    value, calls = discover(tmp_path)
    assert value["weightBytes"] == 100
    assert value["checkpoint"]["files"]["model-1.safetensors"] == "a"*64
    assert value["checkpoint"]["id"] == owned.digest(value["checkpoint"]["files"])
    assert calls and not any(n.endswith(".safetensors") for n in calls)


@pytest.mark.parametrize("field,value", [("private",True),("gated","auto"),("sha","c"*40)])
def test_refuses_private_gated_or_moved_revision(tmp_path, field, value):
    with pytest.raises(ValueError, match="public ungated"):
        discover(tmp_path, lambda c,b,i:setattr(i,field,value))


def test_refuses_changed_tokenizer_and_missing_sha256(tmp_path):
    config, baseline, info, api, download, _ = fixture(tmp_path)
    baseline["files"]["tokenizer.json"] = "d"*64
    with pytest.raises(ValueError, match="tokenizer"):
        owned.discover_checkpoint("owner/model","b"*40,baseline,config,api=api,download=download)
    baseline["files"]["tokenizer.json"] = hashlib.sha256(b"frozen tokenizer").hexdigest()
    info.siblings[-1].lfs = {"sha256":"a"*40}
    with pytest.raises(ValueError, match="SHA256"):
        owned.discover_checkpoint("owner/model","b"*40,baseline,config,api=api,download=download)


def test_refuses_changed_geometry_and_oversized_checkpoint(tmp_path):
    config, baseline, info, api, download, _ = fixture(tmp_path)
    other = dict(config, text_config=dict(num_hidden_layers=64))
    with pytest.raises(ValueError, match="geometry"):
        owned.discover_checkpoint("owner/model","b"*40,baseline,other,api=api,download=download)
    info.siblings[-1].size = owned.MAX_CHECKPOINT_BYTES
    with pytest.raises(ValueError, match="byte ceiling"):
        owned.discover_checkpoint("owner/model","b"*40,baseline,config,api=api,download=download)


def test_allows_training_origin_metadata_without_changing_geometry(tmp_path):
    config, baseline, info, api, download, _ = fixture(tmp_path)
    config.update(_name_or_path="finetuned", transformers_version="new")
    value = owned.discover_checkpoint("owner/model","b"*40,baseline,config,api=api,download=download)
    assert value["revision"] == "b"*40


@pytest.mark.parametrize("name", ["../model.safetensors","dir/model.safetensors","x.py","x..json"])
def test_rejects_unsafe_checkpoint_members(name):
    assert not owned._safe_name(name)


def test_deadline_reserves_provider_and_customer_window():
    assert owned.bounded_deadline(1000, 8200, 9000, 900) == 7600
    assert owned.bounded_deadline(1000, 8200, 8000, 900) == 7400
    with pytest.raises(ValueError, match="window"):
        owned.bounded_deadline(1000, 3024, 9000, 900)
    with pytest.raises(ValueError, match="window"):
        owned.bounded_deadline(1000, 8200, 9000, 3600)
