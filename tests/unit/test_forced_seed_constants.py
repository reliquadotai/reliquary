import reliquary.constants as c


def test_forced_seed_constants_defaults():
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v1"
    assert c.FORCED_SEED_STOCHASTIC_MAXPROB == 0.99
    assert c.FORCED_SEED_CONSISTENCY_FLOOR == 0.80
    assert c.FORCED_SEED_MIN_STOCH_POSITIONS == 30
    assert c.FORCED_SEED_ENFORCE_FROM_WINDOW == 2 ** 63 - 1   # sentinel: never, until armed
    # Clients that sample from the forced stream advertise this on the wire so
    # the operator can watch adoption in the shadow window.
    assert c.FORCED_SEED_PROTOCOL_VERSION == 1
