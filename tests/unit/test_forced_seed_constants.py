import reliquary.constants as c


def test_forced_seed_constants_defaults():
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v2"
    assert c.FORCED_SEED_STOCHASTIC_MAXPROB == 0.99
    assert c.FORCED_SEED_MIN_STOCH_POSITIONS == 30
    assert c.FORCED_SEED_CDF_BOUNDARY_EPSILON == 0.002
