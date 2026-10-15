#!/usr/bin/env python3
"""Measure checkpoint deltas on real checkpoints (docs/rl-async-design.md §4.8).

    python scripts/bench_ckpt_delta.py CKPT_A CKPT_B [CKPT_C ...] [--json out.json]

Each argument is a local directory of ``*.safetensors`` shards (a saved HF
checkpoint), in version order. For every consecutive pair it reports the fraction
of bf16 elements whose bits changed (per tensor with --per-tensor, and overall),
the delta size raw (int32 index + bf16 value per change, whole non-bf16 tensors)
and compressed, and build / encode / decode / apply / hash times, after checking
the rebuild is bit-exact. With three or more checkpoints it also composes the
cumulative delta first -> last (union growth over several steps).

Not a test: it loads two full checkpoints in RAM (about 36 GB for a 9B).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reliquary.shared.ckpt_delta import (  # noqa: E402
    apply_delta,
    changed_masks,
    checkpoint_sha256,
    compose_deltas,
    decode_delta,
    delta_from_masks,
    encode_delta,
)


def load_dir(folder: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no *.safetensors in {folder}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(shard)))
    return state


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def raw_size(delta) -> int:
    return sum(p.values.nbytes if p.whole else p.values.size * 6 for p in delta.patches)


def measure(old, new, sha_old, sha_new, k, args, out):
    label = f"{k - 1}->{k}"
    masks, t_masks = timed(changed_masks, old, new)
    delta, t_build = timed(delta_from_masks, new, masks, from_version=k - 1, to_version=k,
                           base_sha256=sha_old, target_sha256=sha_new)
    blob, t_encode = timed(encode_delta, delta, level=args.level)
    decoded, t_decode = timed(decode_delta, blob)
    rebuilt = {n: t.clone() for n, t in old.items()}
    _, t_apply = timed(apply_delta, rebuilt, decoded, verify=False)
    sha_rebuilt, t_hash = timed(checkpoint_sha256, rebuilt)
    del rebuilt
    bf16 = [n for n, t in new.items() if t.dtype == torch.bfloat16]
    total = sum(new[n].numel() for n in bf16)
    changed = sum(int(masks[n].sum()) for n in bf16)
    full_bytes = sum(t.numel() * t.element_size() for t in new.values())
    row = {
        "pair": label,
        "bf16_elements": total,
        "changed_bf16_elements": changed,
        "changed_fraction": changed / total if total else 0.0,
        "non_bf16_tensors_changed": sorted(p.name for p in delta.patches if p.whole),
        "checkpoint_bytes": full_bytes,
        "delta_raw_bytes": raw_size(delta),
        "delta_compressed_bytes": len(blob),
        "ratio_vs_full": full_bytes / max(1, len(blob)),
        "seconds": {"masks": t_masks, "build": t_build, "encode": t_encode,
                    "decode": t_decode, "apply": t_apply, "hash": t_hash},
        "bit_exact": sha_rebuilt == sha_new,
    }
    if args.per_tensor:
        row["per_tensor"] = {
            n: int(masks[n].sum()) / new[n].numel() for n in sorted(bf16) if new[n].numel()}
    out.append(row)
    print(f"{label}: changed {row['changed_fraction']:.4%} of {total:,} bf16 elements; "
          f"delta {row['delta_raw_bytes'] / 1e6:,.1f} MB raw, "
          f"{row['delta_compressed_bytes'] / 1e6:,.1f} MB zstd-{args.level} "
          f"({row['ratio_vs_full']:.0f}x smaller than {full_bytes / 1e9:.2f} GB); "
          f"masks {t_masks:.2f}s build {t_build:.2f}s encode {t_encode:.2f}s "
          f"decode {t_decode:.2f}s apply {t_apply:.2f}s hash {t_hash:.2f}s; "
          f"bit-exact {row['bit_exact']}")
    if args.per_tensor:
        for name, frac in sorted(row["per_tensor"].items(), key=lambda kv: -kv[1])[:args.top]:
            print(f"    {frac:8.4%}  {name}")
    if not row["bit_exact"]:
        raise SystemExit("rebuild is NOT bit-exact")
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--level", type=int, default=3, help="zstd level")
    parser.add_argument("--per-tensor", action="store_true")
    parser.add_argument("--top", type=int, default=20, help="tensors listed with --per-tensor")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("need at least two checkpoints")
    rows: list[dict] = []
    first = old = load_dir(args.checkpoints[0])
    sha_old = checkpoint_sha256(old)
    cumulative = None
    for k, folder in enumerate(args.checkpoints[1:], start=1):
        new = load_dir(folder)
        sha_new = checkpoint_sha256(new)
        delta = measure(old, new, sha_old, sha_new, k, args, rows)
        if len(args.checkpoints) > 2:
            cumulative = delta if cumulative is None else compose_deltas(cumulative, delta)
        if old is not first:
            del old
        old, sha_old = new, sha_new
    if cumulative is not None:
        blob, t_encode = timed(encode_delta, cumulative, level=args.level)
        apply_delta(first, cumulative, verify=False)  # first is not needed any more
        exact = checkpoint_sha256(first) == sha_old
        rows.append({"pair": f"0->{len(args.checkpoints) - 1} cumulative",
                     "changed_bf16_elements": cumulative.changed_elements,
                     "delta_compressed_bytes": len(blob), "bit_exact": exact,
                     "seconds": {"encode": t_encode}})
        print(f"cumulative 0->{len(args.checkpoints) - 1}: union "
              f"{cumulative.changed_elements:,} elements, {len(blob) / 1e6:,.1f} MB "
              f"(sum of steps {sum(r['delta_compressed_bytes'] for r in rows[:-1]) / 1e6:,.1f} MB);"
              f" bit-exact {exact}")
        if not exact:
            raise SystemExit("cumulative rebuild is NOT bit-exact")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
