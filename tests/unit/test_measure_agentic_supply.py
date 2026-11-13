from scripts.measure_agentic_supply import (active_rate, miner_rates, offers_against, sandbox_refusals,
                                            summarize)


def test_active_rate_skips_idle_gaps_and_each_sessions_first_record():
    # Session 1: 0, 600, 1200 (tokens 100, 200, 300); idle 10 000 s; session 2: 11 200, 11 800.
    points = [(600.0, 200), (0.0, 100), (1200.0, 300), (11200.0, 50), (11800.0, 400)]
    tokens, seconds = active_rate(points, idle_gap=1800)
    assert tokens == 200 + 300 + 400
    assert seconds == 1200 + 600


def test_miner_rates_and_ratio_per_model():
    single = [{"hotkey": "a", "received_at": t, "token_count": 1000} for t in (0, 60, 120, 180)]
    agentic = [{"hotkey": "b", "received_at": t, "token_count": 1000} for t in (0, 600, 1200)]
    assert miner_rates(single)["a"]["tokens_per_hour"] == 60_000
    assert miner_rates(agentic)["b"]["tokens_per_hour"] == 6_000
    jobs = [{"model": "m", "kind": "single_turn", "miners": miner_rates(single)},
            {"model": "m", "kind": "agentic", "miners": miner_rates(agentic)}]
    assert summarize(jobs)["m"]["single_over_agentic"] == 10


def test_offers_against_the_ceiling():
    state = {"budget": 64, "offers": {"hk-cheap": [1e-9, 4, 1000.0], "hk-dear": [8e-9, 2, 1000.0],
                                      "hk-stale": [1e-9, 1, 0.0]}}
    ceiling = {"max_price_alpha_per_mtoken": 0.66, "max_price_pool_share_per_token": 4e-9}
    view = offers_against(state, ceiling, now=1500.0, ttl=1000.0)
    assert view["live_offers"] == 2
    assert view["above_ceiling"] == 1
    assert view["capacity_offered"] == 6
    assert abs(view["offers"][0]["alpha_per_mtoken"] - 0.165) < 1e-12


def test_sandbox_refusals_by_reason_and_hotkey():
    lines = ["INFO sandbox session open refused for 5Fabc: fleet_full",
             "INFO sandbox session open refused for 5Fabc: fleet_full",
             "INFO sandbox session open refused for 5Gdef: open_refused_rate",
             "INFO unrelated"]
    out = sandbox_refusals(lines)
    assert out["by_reason"] == {"open:fleet_full": 2, "open:open_refused_rate": 1}
    assert out["by_hotkey"]["5Fabc"] == {"fleet_full": 2}
