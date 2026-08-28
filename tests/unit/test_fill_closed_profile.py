"""v6 coexists: it is selectable, and it leaves v4/v5 untouched."""
from reliquary.protocol.profiles import resolve_protocol_profile


def test_v6_is_selectable_and_carries_the_v5_generation_contract():
    v5 = resolve_protocol_profile("qwen3-4b-base-dapo-reasoning-v5")
    v6 = resolve_protocol_profile("qwen3-4b-base-dapo-fill-closed-v6")

    assert v6.protocol_version == 6
    # v6 changes the WINDOW, not what a miner generates. Everything a miner
    # samples from must be byte-identical or the change is not what it claims.
    assert v6.sampling == v5.sampling
    assert v6.model_id == v5.model_id
    assert v6.model_revision == v5.model_revision
    assert {name: env.max_new_tokens for name, env in v6.environments.items()} == \
           {name: env.max_new_tokens for name, env in v5.environments.items()}


def test_v6_has_no_throughput_tiebreak():
    """There is no ranking in v6, so there is nothing to break ties in."""
    v6 = resolve_protocol_profile("qwen3-4b-base-dapo-fill-closed-v6")
    assert v6.throughput_tiebreak is None
