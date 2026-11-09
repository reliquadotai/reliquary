from reliquary.protocol.submission import RejectReason


def test_seed_mismatch_reason_exists():
    assert RejectReason.SEED_MISMATCH.value == "seed_mismatch"


