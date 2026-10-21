"""Lossless sparse checkpoint deltas: bit-exact rebuild, cumulative deltas,
hash checks (docs/rl-async-design.md §4.8)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reliquary.shared.ckpt_delta import (
    ChecksumMismatch,
    DeltaError,
    _decode_indices,
    _encode_indices,
    apply_delta,
    build_delta,
    checkpoint_sha256,
    compose_deltas,
    decode_delta,
    encode_delta,
    load_anchor_shards,
    write_anchor_shards,
)


def make_masters(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "embed.weight": torch.randn(257, 33, generator=g),
        "layers.0.attn.q.weight": torch.randn(64, 48, generator=g),
        "layers.0.norm.weight": torch.ones(48),
        "layers.0.bias": torch.randn(7, generator=g),
        "head.weight": torch.randn(31, 64, generator=g) * 1e-3,
    }


def publish(masters, extras=None):
    state = {n: t.to(torch.bfloat16) for n, t in masters.items()}
    state["rope.inv_freq"] = torch.linspace(0, 1, 16)  # fp32, normally constant
    state["step.counter"] = torch.tensor(0, dtype=torch.int64)
    state.update(extras or {})
    return state


def train_step(masters, seed, lr=1e-3):
    g = torch.Generator().manual_seed(1000 + seed)
    return {n: t - lr * torch.randn(t.shape, generator=g) for n, t in masters.items()}


def bits_equal(a, b):
    assert set(a) == set(b)
    for name in a:
        assert a[name].dtype == b[name].dtype and a[name].shape == b[name].shape
        x = a[name].contiguous().reshape(-1).view(torch.uint8)
        y = b[name].contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(x, y), name
    return True


def clone(state):
    return {n: t.clone() for n, t in state.items()}


def chain(n_steps, seed=0):
    masters = make_masters(seed)
    states = [publish(masters)]
    for k in range(n_steps):
        masters = train_step(masters, k)
        states.append(publish(masters))
    return states


def test_round_trip_is_bit_exact_through_the_wire_format():
    s0, s1 = chain(1)
    delta = build_delta(s0, s1, from_version=0, to_version=1)
    assert 0 < delta.changed_elements < sum(t.numel() for t in s1.values())
    decoded = decode_delta(encode_delta(delta))
    rebuilt = clone(s0)
    apply_delta(rebuilt, decoded)
    assert bits_equal(rebuilt, s1)
    assert checkpoint_sha256(rebuilt) == delta.target_sha256 == checkpoint_sha256(s1)


def test_bits_not_values_signed_zero_and_nan_payloads_are_changes():
    a = {"w": torch.tensor([0.0, 1.0, float("nan"), 2.0], dtype=torch.bfloat16)}
    b = {"w": a["w"].clone()}
    bits = b["w"].view(torch.int16)
    bits[0] = torch.tensor(-0.0, dtype=torch.bfloat16).view(torch.int16)
    bits[2] = bits[2] | 1  # another NaN payload
    delta = build_delta(a, b, from_version=0, to_version=1)
    assert delta.patches[0].indices.tolist() == [0, 2]
    rebuilt = clone(a)
    apply_delta(rebuilt, decode_delta(encode_delta(delta)))
    assert bits_equal(rebuilt, b)


def test_unchanged_checkpoint_gives_an_empty_delta():
    s0 = chain(0)[0]
    delta = build_delta(s0, clone(s0), from_version=3, to_version=4)
    assert delta.patches == ()
    rebuilt = clone(s0)
    apply_delta(rebuilt, decode_delta(encode_delta(delta)))
    assert bits_equal(rebuilt, s0)


def test_non_bf16_tensors_are_stored_whole_only_when_changed():
    s0, s1 = chain(1)
    s1["step.counter"] = torch.tensor(7, dtype=torch.int64)
    delta = build_delta(s0, s1, from_version=0, to_version=1)
    whole = {p.name for p in delta.patches if p.whole}
    assert whole == {"step.counter"}  # rope.inv_freq did not change
    rebuilt = clone(s0)
    apply_delta(rebuilt, decode_delta(encode_delta(delta)))
    assert bits_equal(rebuilt, s1)


@pytest.mark.parametrize("change", ["added", "removed", "reshaped", "retyped"])
def test_structure_changes_are_refused(change):
    s0, s1 = chain(1)
    if change == "added":
        s1["new.weight"] = torch.zeros(3, dtype=torch.bfloat16)
    elif change == "removed":
        del s1["head.weight"]
    elif change == "reshaped":
        s1["layers.0.bias"] = torch.zeros(8, dtype=torch.bfloat16)
    else:
        s1["layers.0.bias"] = s1["layers.0.bias"].float()
    with pytest.raises(DeltaError):
        build_delta(s0, s1, from_version=0, to_version=1)


def test_apply_refuses_a_checkpoint_of_another_shape():
    s0, s1 = chain(1)
    delta = build_delta(s0, s1, from_version=0, to_version=1)
    s0["embed.weight"] = torch.zeros(3, dtype=torch.bfloat16)
    with pytest.raises(DeltaError):
        apply_delta(s0, delta)


def test_cumulative_equals_sequential_and_applies_from_any_version_inside():
    states = chain(4)
    deltas = [build_delta(states[k], states[k + 1], from_version=k, to_version=k + 1)
              for k in range(4)]
    cumulative = deltas[0]
    for d in deltas[1:]:
        cumulative = compose_deltas(cumulative, d)
    cumulative = decode_delta(encode_delta(cumulative))
    assert (cumulative.from_version, cumulative.to_version) == (0, 4)
    direct = build_delta(states[0], states[4], from_version=0, to_version=4)
    # The union covers at least every position that differs end to end.
    assert cumulative.changed_elements >= direct.changed_elements
    for start in range(4):
        rebuilt = clone(states[start])
        apply_delta(rebuilt, cumulative)
        assert bits_equal(rebuilt, states[4])
    sequential = clone(states[0])
    for d in deltas:
        apply_delta(sequential, d)
    assert bits_equal(sequential, states[4])


def test_cumulative_handles_reverted_positions_for_a_miner_in_between():
    a = {"w": torch.zeros(8, dtype=torch.bfloat16)}
    b = {"w": a["w"].clone()}
    b["w"][3] = 1.0
    c = {"w": a["w"].clone()}  # position 3 went back to 0 at version 2
    d1 = build_delta(a, b, from_version=0, to_version=1)
    d2 = build_delta(b, c, from_version=1, to_version=2)
    cumulative = compose_deltas(d1, d2)
    rebuilt = clone(b)  # a miner at version 1
    apply_delta(rebuilt, cumulative)
    assert bits_equal(rebuilt, c)


def test_compose_mixes_whole_and_sparse_patches():
    s0, s1, s2 = chain(2)
    s1["step.counter"] = torch.tensor(1, dtype=torch.int64)
    s2["step.counter"] = torch.tensor(2, dtype=torch.int64)
    cumulative = compose_deltas(build_delta(s0, s1, from_version=0, to_version=1),
                                build_delta(s1, s2, from_version=1, to_version=2))
    rebuilt = clone(s0)
    apply_delta(rebuilt, decode_delta(encode_delta(cumulative)))
    assert bits_equal(rebuilt, s2)


def test_compose_refuses_non_consecutive_deltas():
    s0, s1, s2 = chain(2)
    d1 = build_delta(s0, s1, from_version=0, to_version=1)
    with pytest.raises(DeltaError):
        compose_deltas(d1, build_delta(s0, s2, from_version=0, to_version=2))


def test_wrong_base_or_tampered_values_fail_the_hash():
    s0, s1, s2 = chain(2)
    delta = build_delta(s1, s2, from_version=1, to_version=2)
    with pytest.raises(ChecksumMismatch):
        apply_delta(clone(s0), delta)  # applied on the wrong base
    delta.patches[0].values[0] ^= 1
    with pytest.raises(ChecksumMismatch):
        apply_delta(clone(s1), delta)


def test_corrupt_wire_bytes_are_refused():
    s0, s1 = chain(1)
    blob = encode_delta(build_delta(s0, s1, from_version=0, to_version=1))
    with pytest.raises(DeltaError):
        decode_delta(b"XXXXXXXX" + blob[8:])
    with pytest.raises(Exception):
        decode_delta(blob[: len(blob) - 50])



def _sparse_pair(seed=0):
    """bf16 tensors with sparse, dense and empty regions across chunk borders."""
    g = torch.Generator().manual_seed(seed)
    a = {"big": torch.randn(37, 101, generator=g).to(torch.bfloat16),
         "small": torch.randn(9, generator=g).to(torch.bfloat16)}
    b = clone(a)
    bits = b["big"].view(-1).view(torch.int16)
    bits[:400] += 1  # dense: bitmask chunks
    bits[1000:3000:97] += 1  # sparse: gap chunks; 3000.. untouched: no chunk
    bits[-1] += 1  # last element of a partial last chunk
    return a, b


@pytest.mark.parametrize("chunk", [1, 7, 64, 100, 3737, 2**24])
def test_chunked_wire_format_round_trips_and_ignores_worker_count(chunk):
    a, b = _sparse_pair()
    delta = build_delta(a, b, from_version=0, to_version=1)
    blob = encode_delta(delta, chunk_elements=chunk, workers=1)
    assert encode_delta(delta, chunk_elements=chunk, workers=5) == blob
    for workers in (1, 3):
        decoded = decode_delta(blob, workers=workers)
        for p, q in zip(delta.patches, decoded.patches):
            np.testing.assert_array_equal(p.indices, q.indices)
            np.testing.assert_array_equal(p.values, q.values)
        rebuilt = clone(a)
        apply_delta(rebuilt, decoded, workers=workers)
        assert bits_equal(rebuilt, b)


def _rewrite_header(blob, edit):
    import json
    import struct

    (n,) = struct.unpack("<Q", blob[8:16])
    header = json.loads(blob[16:16 + n])
    edit(header)
    raw = json.dumps(header).encode()
    return blob[:8] + struct.pack("<Q", len(raw)) + raw + blob[16 + n:]


@pytest.mark.parametrize("edit", [
    lambda h: h.update(format=1),
    lambda h: h.update(chunk_elements=0),
    lambda h: h["tensors"][0].update(count=h["tensors"][0]["count"] + 1),
    lambda h: h["tensors"][0]["chunks"].reverse(),
    lambda h: h["tensors"][0]["chunks"][0].__setitem__(0, 10**6),
    lambda h: h["tensors"][0]["chunks"][0].__setitem__(2, "gaps64"),
])
def test_corrupt_chunk_tables_are_refused(edit):
    a, b = _sparse_pair()
    blob = encode_delta(build_delta(a, b, from_version=0, to_version=1), chunk_elements=500)
    with pytest.raises(DeltaError):
        decode_delta(_rewrite_header(blob, edit))

def test_hash_ignores_order_but_sees_names_dtypes_and_shapes():
    s0 = chain(0)[0]
    reordered = dict(reversed(list(s0.items())))
    assert checkpoint_sha256(reordered) == checkpoint_sha256(s0)
    renamed = clone(s0)
    renamed["head.weight2"] = renamed.pop("head.weight")
    reshaped = clone(s0)
    reshaped["embed.weight"] = reshaped["embed.weight"].reshape(33, 257).contiguous()
    retyped = clone(s0)
    retyped["layers.0.bias"] = retyped["layers.0.bias"].view(torch.float16)
    digests = {checkpoint_sha256(s) for s in (s0, renamed, reshaped, retyped)}
    assert len(digests) == 4


@pytest.mark.parametrize("numel,count,encoding", [
    (10_000, 10, "gaps32"), (10_000, 1_000, "gaps32"), (10_000, 5_000, "bitmask"), (2**32, 3, "gaps32")])
def test_index_encodings_round_trip(numel, count, encoding):
    rng = np.random.default_rng(0)
    pool = 10_000 if numel > 10_000 else numel
    idx = np.sort(rng.choice(pool, size=count, replace=False)).astype(np.int64)
    if numel > 10_000:
        idx[-1] = numel - 1
    got, raw = _encode_indices(idx, numel)
    assert got == encoding
    np.testing.assert_array_equal(_decode_indices(got, raw, numel, count), idx)


def test_anchor_shards_round_trip(tmp_path):
    s0 = chain(0)[0]
    paths = write_anchor_shards(s0, tmp_path, shard_bytes=4096)
    assert len(paths) > 1
    loaded = load_anchor_shards(paths)
    assert bits_equal(loaded, s0)
    assert checkpoint_sha256(loaded) == checkpoint_sha256(s0)
