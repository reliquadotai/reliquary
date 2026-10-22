"""Check CUDA wheel and extension compatibility without loading model weights."""

import json
from importlib.metadata import version

import torch

# Torch loads the shared libraries required by the native extensions.
import causal_conv1d_cuda
import flash_attn_2_cuda
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from reliquary.shared.runtime_fingerprint import collect_runtime_fingerprint


expected = {
    "torch": "2.13.0",
    "flash-attn": "2.8.3.post2",
    "flash-linear-attention": "0.5.0",
    "causal-conv1d": "1.6.2.post1",
}
installed = {name: version(name) for name in expected}
for name, release in expected.items():
    assert installed[name].split("+")[0] == release, installed
assert torch.version.cuda == "13.0", torch.version.cuda
profile = collect_runtime_fingerprint()
assert all(callable(kernel) for kernel in (
    causal_conv1d_fn, causal_conv1d_update,
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
))
# Transformers enables these bindings only when a GPU is attached.
if torch.cuda.is_available():
    for key in (
        "qwen35_fast_path_all", "qwen35_fla_chunk", "qwen35_fla_recurrent",
        "qwen35_causal_conv_prefill", "qwen35_causal_conv_update",
    ):
        assert profile[key] is True, (key, profile[key])
print(json.dumps({"versions": installed, "cuda": torch.version.cuda,
                  "extensionImports": "passed", "gpuExecutionVerified": False}))
