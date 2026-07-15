import reliquary.constants as c


def test_forced_seed_constants_defaults():
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v2"
    assert c.FORCED_SEED_STOCHASTIC_MAXPROB == 0.99
    assert c.FORCED_SEED_CONSISTENCY_FLOOR == 0.80
    assert c.FORCED_SEED_MIN_STOCH_POSITIONS == 30
    assert c.FORCED_SEED_ENFORCE is True   # ships armed: merging the branch enforces
    # Clients that sample from the forced stream advertise this on the wire so
    # the operator can watch adoption in the shadow window.
    assert c.FORCED_SEED_PROTOCOL_VERSION == 2   # v2: hotkey dropped from the seed
    assert c.FORCED_SEED_CDF_BOUNDARY_EPSILON == 0.002
    assert c.FORCED_SEED_CDF_ENFORCE is False
