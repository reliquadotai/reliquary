"""End-to-end GPU tests — runs on a real transformer model.

Verifies the complete GRAIL proof pipeline:
1. Load a small model (GPT-2 small, 124M params)
2. Run forward pass to get hidden states
3. Create sketch commitments (miner side)
4. Verify sketch commitments (validator side)
5. Test that forged/wrong commitments are rejected

Requires: GPU with CUDA, ~1GB VRAM
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for end-to-end GRAIL tests",
)

from reliquary.constants import CHALLENGE_K, LAYER_INDEX, PRIME_Q
from reliquary.protocol.crypto import indices_from_root
from reliquary.protocol.grail_verifier import GRAILVerifier
from reliquary.shared.forward import forward_single_layer
from reliquary.shared.hf_compat import resolve_hidden_size


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load GPT-2 small for testing — cached across all tests in this module."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to("cuda:0").eval()

    return model, tokenizer


@pytest.fixture
def randomness():
    return "aabbccddee11223344556677889900ff"


class TestForwardPassDeterminism:
    """The forward pass must be deterministic — miner and validator MUST
    get identical hidden states for the same (model, tokens, mask).
    """

    def test_two_passes_identical(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog", return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            h1, l1 = forward_single_layer(model, tokens, None, LAYER_INDEX)
            h2, l2 = forward_single_layer(model, tokens, None, LAYER_INDEX)

        assert torch.equal(h1, h2), "Forward pass is not deterministic!"
        assert torch.equal(l1, l2), "Logits are not deterministic!"

    def test_hidden_state_shape(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        tokens = tokenizer.encode("Hello world", return_tensors="pt").to("cuda:0")
        hidden_dim = resolve_hidden_size(model)

        with torch.no_grad():
            h, logits = forward_single_layer(model, tokens, None, LAYER_INDEX)

        seq_len = tokens.shape[1]
        assert h.shape == (1, seq_len, hidden_dim)
        assert logits.shape[0] == 1
        assert logits.shape[1] == seq_len


class TestProofRoundtripGPU:
    """Full proof roundtrip on real model: create commitments and verify them."""

    def test_self_verification_passes(self, model_and_tokenizer, randomness):
        """Miner and validator running the same model should always verify."""
        model, tokenizer = model_and_tokenizer
        hidden_dim = resolve_hidden_size(model)

        text = "In a distant galaxy far away, there existed a civilization"
        input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda:0")
        tokens = input_ids[0].tolist()
        seq_len = len(tokens)

        with torch.no_grad():
            hidden_states, _ = forward_single_layer(model, input_ids, None, LAYER_INDEX)
        hidden_states = hidden_states[0].cpu()  # [seq_len, hidden_dim]

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec(randomness)

        # Miner creates commitments
        commitments = verifier.create_commitments_batch(hidden_states, r_vec)
        assert len(commitments) == seq_len

        # Validator verifies at challenge positions
        k = min(CHALLENGE_K, seq_len)
        challenge_indices = indices_from_root(tokens, randomness, seq_len, k)

        for idx in challenge_indices:
            valid, diag = verifier.verify_commitment(
                hidden_states[idx], commitments[idx], r_vec, seq_len, idx
            )
            assert valid, (
                f"Self-verification failed at position {idx}: "
                f"sketch_diff={diag['sketch_diff']}, tolerance={diag['sketch_tolerance']}"
            )

    def test_batch_matches_single_on_real_model(self, model_and_tokenizer, randomness):
        """Batch commitment creation must match single-position creation."""
        model, tokenizer = model_and_tokenizer
        hidden_dim = resolve_hidden_size(model)

        text = "The transformer architecture has revolutionized natural language processing"
        input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            h, _ = forward_single_layer(model, input_ids, None, LAYER_INDEX)
        h = h[0].cpu()

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec(randomness)

        batch = verifier.create_commitments_batch(h, r_vec)
        for i in range(h.shape[0]):
            single = verifier.create_commitment(h[i], r_vec)
            assert batch[i]["sketch"] == single["sketch"], (
                f"Batch/single mismatch at pos {i}: "
                f"batch={batch[i]['sketch']} single={single['sketch']}"
            )


class TestForgeryDetectionGPU:
    """Verify that the proof system detects various forgery attempts
    on real model hidden states.
    """

    def test_wrong_model_fails(self, model_and_tokenizer, randomness):
        """Commitments from one model must fail verification on another.
        We simulate this by using a different random seed for hidden states.
        """
        model, tokenizer = model_and_tokenizer
        hidden_dim = resolve_hidden_size(model)

        text = "Security testing is critical for blockchain protocols"
        input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda:0")
        tokens = input_ids[0].tolist()
        seq_len = len(tokens)

        with torch.no_grad():
            real_hidden, _ = forward_single_layer(model, input_ids, None, LAYER_INDEX)
        real_hidden = real_hidden[0].cpu()

        # Simulate "wrong model" with random hidden states
        torch.manual_seed(999)
        fake_hidden = torch.randn_like(real_hidden)

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec(randomness)

        # Miner commits with fake hidden states
        fake_commitments = verifier.create_commitments_batch(fake_hidden, r_vec)

        # Validator verifies against real hidden states
        k = min(CHALLENGE_K, seq_len)
        challenge_indices = indices_from_root(tokens, randomness, seq_len, k)

        failures = 0
        for idx in challenge_indices:
            valid, _ = verifier.verify_commitment(
                real_hidden[idx], fake_commitments[idx], r_vec, seq_len, idx
            )
            if not valid:
                failures += 1

        # With completely different hidden states, almost all should fail
        assert failures >= k * 0.8, (
            f"Only {failures}/{k} positions failed with fake hidden states"
        )

    def test_corrupted_single_commitment_detected(self, model_and_tokenizer, randomness):
        """Corrupting even one commitment should cause verification to fail."""
        model, tokenizer = model_and_tokenizer
        hidden_dim = resolve_hidden_size(model)

        text = "Verifiable inference ensures honest computation"
        input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda:0")
        tokens = input_ids[0].tolist()
        seq_len = len(tokens)

        with torch.no_grad():
            h, _ = forward_single_layer(model, input_ids, None, LAYER_INDEX)
        h = h[0].cpu()

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec(randomness)
        commitments = verifier.create_commitments_batch(h, r_vec)

        # Corrupt one challenged position
        k = min(CHALLENGE_K, seq_len)
        challenge_indices = indices_from_root(tokens, randomness, seq_len, k)
        corrupt_idx = challenge_indices[0]
        original_sketch = commitments[corrupt_idx]["sketch"]
        commitments[corrupt_idx]["sketch"] = (original_sketch + PRIME_Q // 2) % PRIME_Q

        # Verify: the corrupted position should fail
        valid, diag = verifier.verify_commitment(
            h[corrupt_idx], commitments[corrupt_idx], r_vec, seq_len, corrupt_idx
        )
        assert not valid, f"Corrupted commitment passed! diff={diag['sketch_diff']}"

    def test_different_randomness_different_sketches(self, model_and_tokenizer):
        """Same hidden states with different randomness must produce different sketches."""
        model, tokenizer = model_and_tokenizer
        hidden_dim = resolve_hidden_size(model)

        text = "Randomness is the foundation of challenge selection"
        input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            h, _ = forward_single_layer(model, input_ids, None, LAYER_INDEX)
        h = h[0].cpu()

        verifier = GRAILVerifier(hidden_dim=hidden_dim)

        r_vec_a = verifier.generate_r_vec("aaaa" * 8)
        r_vec_b = verifier.generate_r_vec("bbbb" * 8)
        assert not torch.equal(r_vec_a, r_vec_b)

        commits_a = verifier.create_commitments_batch(h, r_vec_a)
        commits_b = verifier.create_commitments_batch(h, r_vec_b)

        differences = sum(
            1 for i in range(len(commits_a))
            if commits_a[i]["sketch"] != commits_b[i]["sketch"]
        )
        assert differences > 0, "Different randomness should produce different sketches"


class TestLongSequenceGPU:
    """Test with longer sequences to verify tolerance scaling."""

    def test_long_sequence_self_verification(self, model_and_tokenizer, randomness):
        """Longer sequences should still self-verify (tolerance grows with sqrt(position))."""
        model, tokenizer = model_and_tokenizer
        hidden_dim = resolve_hidden_size(model)

        # Generate a longer sequence by repeating text
        text = "This is a longer test sequence to verify tolerance scaling. " * 20
        input_ids = tokenizer.encode(text, return_tensors="pt")[:, :512].to("cuda:0")
        tokens = input_ids[0].tolist()
        seq_len = len(tokens)

        with torch.no_grad():
            h, _ = forward_single_layer(model, input_ids, None, LAYER_INDEX)
        h = h[0].cpu()

        verifier = GRAILVerifier(hidden_dim=hidden_dim)
        r_vec = verifier.generate_r_vec(randomness)
        commitments = verifier.create_commitments_batch(h, r_vec)

        k = min(CHALLENGE_K, seq_len)
        challenge_indices = indices_from_root(tokens, randomness, seq_len, k)

        for idx in challenge_indices:
            valid, diag = verifier.verify_commitment(
                h[idx], commitments[idx], r_vec, seq_len, idx
            )
            assert valid, (
                f"Self-verification failed at position {idx}/{seq_len}: "
                f"diff={diag['sketch_diff']}, tol={diag['sketch_tolerance']}"
            )
