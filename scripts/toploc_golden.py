"""Write tests/fixtures/toploc_golden.json from the reference library.

The toploc wheel is compiled against a 2025 torch ABI and does not import with
torch >= 2.12, so run this in a throwaway venv:

    python3 -m venv /tmp/toploc-ref && . /tmp/toploc-ref/bin/activate
    pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
    pip install toploc==0.1.6 numpy
    python scripts/toploc_golden.py

If `import toploc` still fails, try torch 2.6.0, then build from source
(github.com/PrimeIntellect-ai/toploc @ 7ab7bcd, `pip install .`).
"""

import json
from pathlib import Path

import torch
from toploc import build_proofs_bytes, verify_proofs_bytes

CHUNK, TOPK, ROWS, WIDTH = 32, 128, 70, 64
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "toploc_golden.json"


def bits(t):
    return t.contiguous().view(torch.int16).tolist()


cases = []
for seed in range(4):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(ROWS, WIDTH, generator=g).to(torch.bfloat16)
    rows = [hidden[i] for i in range(ROWS)]
    proofs = build_proofs_bytes(rows, decode_batching_size=CHUNK, topk=TOPK, skip_prefill=True)
    noise = torch.randn(ROWS, WIDTH, generator=g)
    verifiers = {
        "identical": hidden,
        "perturbed": (hidden.float() + 1e-2 * noise).to(torch.bfloat16),
        "unrelated": torch.randn(ROWS, WIDTH, generator=g).to(torch.bfloat16),
    }
    checks = {}
    for name, verifier in verifiers.items():
        # A tensor with skip_prefill=True takes the C++ path, whose median we port.
        results = verify_proofs_bytes(verifier, proofs, CHUNK, TOPK, skip_prefill=True)
        checks[name] = {
            "hidden_bits": bits(verifier),
            "results": [[r.exp_mismatches, r.mant_err_mean, r.mant_err_median] for r in results],
        }
    cases.append({"seed": seed, "hidden_bits": bits(hidden),
                  "proofs_hex": [p.hex() for p in proofs], "checks": checks})

OUT.write_text(json.dumps({"reference": "toploc==0.1.6", "chunk": CHUNK, "topk": TOPK,
                           "rows": ROWS, "width": WIDTH, "cases": cases}))
print(f"wrote {OUT}")
