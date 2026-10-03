"""Gate M3: seconds and memory to audit one agentic trajectory on the 27B.

Times ``span_hidden_states`` plus per-span verification at several lengths, on
a synthetic token stream of each target length (cost depends on length, not on
content). A length that runs out of memory is a result, recorded as such.
"""

import argparse
import json
import time

import torch

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc_proof import build_chunk_proofs, verify_chunk_proofs
from reliquary.validator.corpus_audit import span_hidden_states


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="10000,20000,40000,50000,60000")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    from reliquary.shared.modeling import load_text_only_model

    model = load_text_only_model(snapshot_download(args.model), torch_dtype=torch.bfloat16,
                                 attn_implementation="sdpa").to("cuda").eval()
    vocab = model.get_input_embeddings().num_embeddings
    results = []
    for length in (int(x) for x in args.lengths.split(",")):
        tokens = [(i * 7919) % (vocab - 1) + 1 for i in range(length)]
        spans = [(s, min(s + 400, length)) for s in range(1000, length, 1000)]
        for rep in range(args.repeats):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            try:
                t0 = time.monotonic()
                rows = span_hidden_states(model, tokens, spans)
                torch.cuda.synchronize(); t1 = time.monotonic()
                for hidden in rows:
                    verify_chunk_proofs(hidden, build_chunk_proofs(
                        hidden, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk),
                        chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
                t2 = time.monotonic()
            except torch.cuda.OutOfMemoryError as exc:
                results.append({"tokens": length, "oom": True, "error": str(exc)[:200],
                                "peak_gb": torch.cuda.max_memory_allocated() / 2**30})
                print(results[-1], flush=True)
                break
            results.append({"tokens": length, "rep": rep, "prefill_s": t1 - t0,
                            "verify_s": t2 - t1,
                            "peak_gb": torch.cuda.max_memory_allocated() / 2**30})
            print(results[-1], flush=True)
    json.dump({"gate": "M3", "model": args.model, "gpu": torch.cuda.get_device_name(),
               "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
