"""TOPLOC per-span audit bands of the agentic corpus under one attention
kernel: re-run the validator's audit path over recorded multi-turn
trajectories (miner proofs made by vLLM) with ``--attn`` (sdpa or
flash_attention_2), and record every chunk's exp/mant measures.

Run it once per kernel on the SAME work files, then ``compare``:

    HF_HOME=/opt/hf python scripts/agentic_attention_bands.py measure --attn sdpa \
        --smoke /opt/smoke.json --m1 /opt/m1-27b.json /opt/m1-27b-off.json \
        --self-proofs-out sdpa-proofs.json --out sdpa.json
    HF_HOME=/opt/hf python scripts/agentic_attention_bands.py measure --attn flash_attention_2 \
        --smoke /opt/smoke.json --m1 /opt/m1-27b.json /opt/m1-27b-off.json \
        --cross-proofs sdpa-proofs.json --out fa2.json
    python scripts/agentic_attention_bands.py compare sdpa.json fa2.json --out bands.json

Smoke rows (``agentic_miner_smoke.py``: real SWE episodes, proofs from the
miner's span builder) go through the production path: ``span_hidden_states``,
``trajectory_chunk_scores``, ``trajectory_outcome``. M1 rows
(``agentic_proof_gate.py``: proofs from ``build_chunk_proofs``, no short-tail
merge) go through the gate's path, ``verify_chunk_proofs`` per span and
``sequence_verdict``; spans under ``MIN_CHUNK_TOKENS`` are reported apart,
since production does not judge them alone.

``--self-proofs-out`` writes proofs built from this run's own activations
(same builder as the source); ``--cross-proofs`` verifies such a file
against this run's activations: validator kernel A against validator kernel B.
"""

import argparse
import base64
import importlib
import json
import math
import platform
import sys

MODEL = "Qwen/Qwen3.8-27B"


def _versions():
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "flash_attn", "fla", "causal_conv1d", "triton"):
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "installed")
        except Exception as exc:  # noqa: BLE001
            out[name] = f"absent ({type(exc).__name__})"
    return out


def _sources(args):
    """(name, kind, tokens, absolute spans, per-span proof lists)."""
    out = []
    if args.smoke:
        rows = [r for r in json.load(open(args.smoke))["rows"] if "tokens" in r]
        for r in rows:
            prompt = r["prompt_ids"]
            spans = [(len(prompt) + s, len(prompt) + e) for s, e in r["spans"]]
            out.append((f"smoke:{r['index']}", "span", prompt + r["tokens"], spans, r["proofs"]))
    for path in args.m1 or []:
        for k, t in enumerate(json.load(open(path))["trajectories"]):
            out.append((f"{path.rsplit('/', 1)[-1]}:{k}", "chunk", t["tokens"],
                        [tuple(s) for s in t["spans"]], t["proofs"]))
    return out


def measure(args):
    import torch
    from huggingface_hub import snapshot_download

    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
    from reliquary.protocol.toploc import MIN_CHUNK_TOKENS, sequence_verdict
    from reliquary.protocol.toploc_proof import build_chunk_proofs, build_span_proofs, verify_chunk_proofs
    from reliquary.shared.modeling import load_text_only_model
    from reliquary.validator.corpus_audit import span_hidden_states, trajectory_chunk_scores, trajectory_outcome

    model = load_text_only_model(snapshot_download(MODEL), torch_dtype=torch.bfloat16,
                                 attn_implementation=args.attn).to("cuda").eval()
    used = getattr(model.config, "_attn_implementation", None)
    if used != args.attn:
        raise SystemExit(f"asked for {args.attn}, model runs {used}")
    cross = json.load(open(args.cross_proofs)) if args.cross_proofs else {}
    kw = dict(chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
    trajectories, self_proofs = [], {}
    for name, kind, tokens, spans, proofs in _sources(args):
        rows = span_hidden_states(model, tokens, spans)
        lengths = [e - s for s, e in spans]
        entry = {"name": name, "kind": kind, "tokens": len(tokens), "span_lengths": lengths}
        if kind == "span":
            builder = lambda h: build_span_proofs(h, **kw)  # noqa: E731
        else:
            builder = lambda h: build_chunk_proofs(h, **kw)  # noqa: E731

        def score(span_proofs):
            if kind == "span":
                flat = [p for turn in span_proofs for p in turn]
                status, results = trajectory_chunk_scores(rows, flat, **kw)
                outcome = trajectory_outcome(status, results, lengths, PROOF)
                # results come back in span order; split them per span
                per_span, at = [], 0
                from reliquary.protocol.toploc import span_chunk_count
                for n in lengths:
                    c = span_chunk_count(n, PROOF.chunk_tokens, MIN_CHUNK_TOKENS)
                    per_span.append(results[at:at + c])
                    at += c
                return {"status": status, "passed": outcome.passed, "reason": outcome.reason,
                        "spans": [_span(chunks, n, PROOF) for chunks, n in zip(per_span, lengths)]}
            per_span = []
            for hidden, p, n in zip(rows, span_proofs, lengths):
                chunks = verify_chunk_proofs(hidden, [base64.b64decode(x) for x in p], **kw)
                per_span.append(_span(chunks, n, PROOF, sequence_verdict))
            judged = [s for s in per_span if s["length"] >= MIN_CHUNK_TOKENS]
            return {"status": "ok", "passed": all(s["passed"] for s in judged) and bool(judged),
                    "reason": None, "spans": per_span}

        entry["miner"] = score(proofs)
        if args.self_proofs_out:
            self_proofs[name] = [[base64.b64encode(p).decode() for p in builder(h)] for h in rows]
        if name in cross:
            entry["cross"] = score(cross[name])
        trajectories.append(entry)
        del rows
        torch.cuda.empty_cache()
        print(f"{name}: miner passed={entry['miner']['passed']}"
              + (f" cross passed={entry['cross']['passed']}" if "cross" in entry else ""), flush=True)
    if args.self_proofs_out:
        json.dump(self_proofs, open(args.self_proofs_out, "w"))
    json.dump({"label": args.label or args.attn, "attn": args.attn, "model_attn": used, "versions": _versions(),
               "gpu": torch.cuda.get_device_name(), "host": platform.node(),
               "thresholds": {"exp": PROOF.exp_mismatch_threshold, "mant_mean": PROOF.mant_mean_threshold,
                              "mant_median": PROOF.mant_median_threshold},
               "trajectories": trajectories}, open(args.out, "w"))


def _span(chunks, length, proof, verdict=None):
    s = {"length": length, "chunks": [[c.exp_mismatches, c.mant_err_mean, c.mant_err_median] for c in chunks]}
    if verdict is not None:
        s["passed"], s["reason"] = verdict(chunks, proof.thresholds())
    return s


def _dist(values):
    v = sorted(x for x in values if isinstance(x, (int, float)) and math.isfinite(x))
    if not v:
        return None
    p = lambda q: v[min(len(v) - 1, int(math.ceil(q * len(v))) - 1)]  # noqa: E731
    return {"n": len(v), "mean": round(sum(v) / len(v), 3), "p50": p(0.5), "p99": p(0.99), "max": v[-1]}


def summarize(run, which="miner", min_len=0):
    chunks, spans, trajs, passed = [], 0, 0, 0
    for t in run["trajectories"]:
        if which not in t:
            continue
        trajs += 1
        passed += bool(t[which]["passed"])
        for s in t[which]["spans"]:
            if s["length"] < min_len:
                continue
            spans += 1
            chunks += s["chunks"]
    th = run["thresholds"]
    return {"trajectories": trajs, "trajectories_passed": passed, "spans": spans, "chunks": len(chunks),
            "exp": _dist([c[0] for c in chunks]), "mant_mean": _dist([c[1] for c in chunks]),
            "mant_median": _dist([c[2] for c in chunks]),
            "chunks_over_threshold": sum(c[0] > th["exp"] or c[1] > th["mant_mean"] or c[2] > th["mant_median"]
                                         for c in chunks)}


def compare(args):
    runs = [json.load(open(p)) for p in args.runs]
    out = {"runs": []}
    for path, run in zip(args.runs, runs):
        entry = {"file": path.rsplit("/", 1)[-1], "label": run.get("label") or run["attn"], "attn": run["attn"],
                 "versions": run["versions"], "gpu": run["gpu"], "thresholds": run["thresholds"]}
        for kind in ("span", "chunk"):
            sub = dict(run, trajectories=[t for t in run["trajectories"] if t["kind"] == kind])
            key = "smoke" if kind == "span" else "m1"
            entry[key] = {"miner_vllm_proofs": summarize(sub)}
            if kind == "chunk":
                entry[key]["miner_vllm_proofs_judged_spans"] = summarize(sub, min_len=8)
            if any("cross" in t for t in sub["trajectories"]):
                entry[key]["cross_validator_proofs"] = summarize(sub, "cross")
                if kind == "chunk":
                    entry[key]["cross_validator_proofs_judged_spans"] = summarize(sub, "cross", min_len=8)
        out["runs"].append(entry)
    json.dump(out, open(args.out, "w"), indent=1)
    for e in out["runs"]:
        for key in ("smoke", "m1"):
            for k, s in e[key].items():
                print(f"{e['label']:>28} {key:5} {k:38} traj {s['trajectories_passed']}/{s['trajectories']} "
                      f"chunks {s['chunks']} exp mean {s['exp']['mean']} p99 {s['exp']['p99']} max {s['exp']['max']} | "
                      f"mant mean {s['mant_mean']['mean']} p99 {s['mant_mean']['p99']} max {s['mant_mean']['max']} | "
                      f"over {s['chunks_over_threshold']}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="phase", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--attn", required=True, choices=["sdpa", "flash_attention_2", "eager"])
    m.add_argument("--smoke")
    m.add_argument("--m1", nargs="*")
    m.add_argument("--self-proofs-out")
    m.add_argument("--cross-proofs")
    m.add_argument("--label")
    m.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+")
    c.add_argument("--out", required=True)
    args = p.parse_args()
    measure(args) if args.phase == "measure" else compare(args)


if __name__ == "__main__":
    main()
