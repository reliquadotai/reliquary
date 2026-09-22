"""Write tests/fixtures/toploc_golden.json from the reference library.

The toploc wheel is compiled against a 2025 torch ABI and does not import with
torch >= 2.12, so run this in a throwaway venv:

    python3 -m venv ~/toploc-ref-venv && . ~/toploc-ref-venv/bin/activate
    pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu
    pip install toploc==0.1.6 numpy
    python scripts/toploc_golden.py

If `import toploc` still fails, try torch 2.6.0, then build from source
(github.com/PrimeIntellect-ai/toploc @ 7ab7bcd, `pip install .`).
"""

import base64
import json
from pathlib import Path

import torch
from toploc import build_proofs_bytes, verify_proofs_bytes

CHUNK, TOPK = 32, 128
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "toploc_golden.json"
# 65497 is the field prime: a chunk wider than it makes the reference reduce
# indices, and two indices one prime apart force the modulus search downward.
COLLIDING_INDICES = (100, 100 + 65497)


def encode(tensor):
    return base64.b64encode(tensor.contiguous().view(torch.int16).numpy().tobytes()).decode()


def case(name, hidden, generator, verifiers):
    rows = [hidden[i] for i in range(hidden.shape[0])]
    proofs = build_proofs_bytes(rows, decode_batching_size=CHUNK, topk=TOPK, skip_prefill=True)
    noise = torch.randn(hidden.shape, generator=generator)
    candidates = {
        "identical": hidden,
        "perturbed": (hidden.float() + 1e-2 * noise).to(torch.bfloat16),
        "unrelated": torch.randn(hidden.shape, generator=generator).to(torch.bfloat16),
    }
    checks = {}
    for check in verifiers:
        verifier = candidates[check]
        # A tensor with skip_prefill=True takes the C++ path, whose median we port.
        results = verify_proofs_bytes(verifier, proofs, CHUNK, TOPK, skip_prefill=True)
        checks[check] = {
            "results": [[r.exp_mismatches, r.mant_err_mean, r.mant_err_median] for r in results],
        }
        if check != "identical":  # identical reuses the case's own tensor
            checks[check]["hidden_b64"] = encode(verifier)
    return {"name": name, "rows": hidden.shape[0], "width": hidden.shape[1],
            "hidden_b64": encode(hidden), "proofs_hex": [p.hex() for p in proofs],
            "checks": checks}


cases = []
for seed in range(4):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(70, 64, generator=g).to(torch.bfloat16)
    cases.append(case(f"narrow-seed{seed}", hidden, g, ("identical", "perturbed", "unrelated")))

g = torch.Generator().manual_seed(100)
wide = torch.randn(33, 2560, generator=g).to(torch.bfloat16)
cases.append(case("wide-2560", wide, g, ("identical", "perturbed")))

g = torch.Generator().manual_seed(101)
colliding = torch.randn(33, 2560, generator=g)
flat = colliding.view(-1)
flat[COLLIDING_INDICES[0]] = 50.0
flat[COLLIDING_INDICES[1]] = -50.0
cases.append(case("forced-collision", colliding.to(torch.bfloat16), g, ("identical", "perturbed")))

OUT.write_text(json.dumps({"reference": "toploc==0.1.6", "chunk": CHUNK, "topk": TOPK,
                           "cases": cases}))
print(f"wrote {OUT}")
