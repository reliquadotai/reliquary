"""GET /rl/pay: what each RL environment pays per paid token, per hour and per
GPU-hour, from the last windows' archives and the chain."""

from __future__ import annotations

import asyncio

import pytest

from reliquary.validator.rl_pay import RlPayBoard, window_stat

MAPB = 0.41  # miners' alpha per block


def _group(env, *, completion, prompt=100, hotkey="hk"):
    rollouts = [{"tokens": [0] * (prompt + completion), "completion_length": completion,
                 "reward": 1.0} for _ in range(16)]
    return {"env_name": env, "selected_for_batch": True, "hotkey": hotkey, "rollouts": rollouts}


def _archive(window, *, open_round, close_round, math_groups=2, code_groups=1,
             math_len=400, code_len=800, pools=None, prices=None):
    batch = ([_group("math", completion=math_len) for _ in range(math_groups)]
             + [_group("code", completion=code_len) for _ in range(code_groups)])
    pools = pools or {"math": 0.02, "code": 0.03}
    return {
        "window_start": window, "window_status": "completed",
        "window_open_round": open_round, "window_close_round": close_round,
        "environments": ["math", "code"], "picks_target": 1, "batch_targets": {"math": 2, "code": 2},
        "window_pool": sum(pools.values()), "window_pool_by_environment": pools,
        "collect_ready_round_by_environment": {"math": open_round + 20, "code": open_round + 40},
        "batch": batch,
        "rejected": [{"env_name": "math", "reject_reason": "batch_filled"},
                     {"env_name": "code", "reject_reason": "out_of_zone"},
                     {"env_name": "code", "reject_reason": "out_of_zone"}],
        "rewards_by_hotkey": {"hk": 0.0},
        "upload_precommit_conservation_by_environment": {
            "math": {"capacity_rejections": {"precommit_hotkey_active_full": 3}}},
        "emission_price_shadow": {"by_environment": {
            e: {"price": (prices or {}).get(e, 0.05), "regime": "hold"} for e in ("math", "code")}},
    }


def _board(chain=None, *, fail=False, clock=None):
    async def read():
        if fail:
            raise RuntimeError("chain down")
        return chain or {"miner_alpha_per_block": MAPB, "alpha_price_tao": 0.005}

    board = RlPayBoard(chain_read=read, clock=clock or (lambda: 1000.0))
    asyncio.run(board.refresh_chain())
    board.add(window_stat(_archive(1, open_round=0, close_round=150)))
    board.add(window_stat(_archive(2, open_round=250, close_round=400)))  # cycle of w1 = 250 rounds
    return board


def test_window_stat_reads_pools_tokens_and_rejections():
    stat = window_stat(_archive(1, open_round=0, close_round=150))
    math = stat["environments"]["math"]
    assert math["pool"] == pytest.approx(0.02)
    assert math["slots"] == 2
    assert math["paid_groups"] == 2
    assert math["paid_share"] == pytest.approx(0.02)          # 2 of 2 slots paid
    assert math["completion_tokens"] == 2 * 16 * 400
    assert math["prompt_tokens"] == 2 * 16 * 100
    assert math["submitted_groups"] == 3                     # 2 paid + 1 rejected upload
    assert math["refused_precommits"] == 3
    assert math["rejections"] == {"batch_filled": 1, "precommit_hotkey_active_full": 3}
    assert math["accept_seconds"] == pytest.approx(60.0)
    code = stat["environments"]["code"]
    assert code["paid_share"] == pytest.approx(0.015)         # 1 of 2 slots of 0.03


def test_the_pay_board_formulas():
    board = _board()
    answer = asyncio.run(board.answer(None))
    math = answer["environments"]["math"]
    cycle = 250 * 3.0
    e_cycle = MAPB * cycle / 12
    assert answer["cycle_seconds_median"] == pytest.approx(cycle)
    assert math["market"]["alpha_per_mtoken_paid"] == pytest.approx(0.02 * e_cycle / 12800 * 1e6)
    assert math["market"]["paid_tokens_per_hour"] == pytest.approx(12800 / cycle * 3600)
    assert math["market"]["alpha_per_hour"] == pytest.approx(0.02 * MAPB * 300)
    assert math["market"]["accepting_fraction_of_cycle"] == pytest.approx(60.0 / cycle)
    assert math["market"]["paid_fraction_of_submitted"] == {
        "low": pytest.approx(4 / 12), "high": pytest.approx(4 / 6)}
    assert math["now"]["alpha_per_paid_group"] == pytest.approx(0.02 / 2 * e_cycle)
    assert math["now"]["alpha_per_mtoken_paid"] == pytest.approx(0.02 / 2 * e_cycle / 6400 * 1e6)
    assert math["price"] == {"value": 0.05, "regime": "hold"}


def test_an_unreadable_chain_nulls_only_the_alpha_fields():
    answer = asyncio.run(_board(fail=True).answer(None))
    math = answer["environments"]["math"]
    assert answer["chain"] is None
    assert math["market"]["alpha_per_mtoken_paid"] is None
    assert math["now"]["alpha_per_paid_group"] is None
    assert math["market"]["paid_tokens_per_hour"] > 0
    assert math["market"]["paid_fraction_of_submitted"]["high"] == pytest.approx(4 / 6)


def test_the_gpu_hour_estimate_is_bounded_by_demand():
    answer = asyncio.run(_board().answer(200.0))
    math = answer["environments"]["math"]
    for bound in ("low", "high"):
        produced = (200.0 * 3600 * math["market"]["accepting_fraction_of_cycle"]
                    * math["market"]["paid_fraction_of_submitted"][bound])
        bounded = min(produced, math["market"]["paid_tokens_per_hour"])
        assert answer["estimate"]["math"]["alpha_per_gpu_hour"][bound] == pytest.approx(
            math["now"]["alpha_per_mtoken_paid"] / 1e6 * bounded)


def test_one_recompute_per_minute():
    now = [1000.0]
    board = _board(clock=lambda: now[0])
    first = asyncio.run(board.answer(None))
    board.add(window_stat(_archive(3, open_round=500, close_round=650, math_len=10)))
    assert asyncio.run(board.answer(None))["generated_at"] == first["generated_at"]
    now[0] += 61
    assert asyncio.run(board.answer(None))["generated_at"] != first["generated_at"]


def test_an_old_archive_without_per_env_pools_splits_the_paid_total():
    archive = _archive(1, open_round=0, close_round=150)
    del archive["window_pool_by_environment"]
    archive["rewards_by_hotkey"] = {"hk": 0.03}
    stat = window_stat(archive)
    assert stat["environments"]["math"]["paid_share"] == pytest.approx(0.02)  # 2 of 3 paid groups
    assert stat["environments"]["code"]["paid_share"] == pytest.approx(0.01)


def test_an_empty_board_says_loading():
    async def read():
        return {"miner_alpha_per_block": MAPB}
    board = RlPayBoard(chain_read=read)
    assert asyncio.run(board.answer(None)) == {"status": "loading"}


def test_the_route_serves_the_board_publicly():
    from fastapi.testclient import TestClient

    from reliquary.validator.server import ValidatorServer

    server = ValidatorServer()
    client = TestClient(server.app)
    assert client.get("/rl/pay").status_code == 503          # no board yet
    server._rl_pay_board = _board()
    body = client.get("/rl/pay", params={"tokens_per_second": 150}).json()
    assert set(body["environments"]) == {"math", "code"}
    assert body["estimate"]["math"]["alpha_per_gpu_hour"]["low"] > 0
    assert client.get("/rl/pay", params={"tokens_per_second": -1}).status_code == 422


def test_seeding_merges_with_live_windows_and_relinks_cycles():
    async def read():
        return {"miner_alpha_per_block": MAPB}
    board = RlPayBoard(chain_read=read)
    board.add(window_stat(_archive(3, open_round=500, close_round=650)))
    board.seed([window_stat(_archive(1, open_round=0, close_round=150)),
                window_stat(_archive(2, open_round=250, close_round=400))])
    windows = [(s["window"], s["next_open_round"]) for s in board._stats]
    assert windows == [(1, 250), (2, 500), (3, None)]


def test_estimates_for_many_speeds_do_not_grow_the_cache():
    """I3: arbitrary tokens_per_second values must not each cache an answer."""
    board = _board()
    for tps in range(1, 200):
        asyncio.run(board.answer(float(tps)))
    assert len(board._cache) <= 1


def test_a_request_never_waits_on_the_chain():
    """I4: the chain is read behind requests; a slow read never stalls one."""
    async def slow():
        await asyncio.sleep(30)
        return {"miner_alpha_per_block": MAPB}

    async def scenario():
        board = RlPayBoard(chain_read=slow)
        board.add(window_stat(_archive(1, open_round=0, close_round=150)))
        board.add(window_stat(_archive(2, open_round=250, close_round=400)))
        answer = await asyncio.wait_for(board.answer(None), 1.0)
        return answer
    answer = asyncio.run(scenario())
    assert answer["chain"] is None and answer["environments"]["math"]["market"]["paid_tokens_per_hour"] > 0


def test_a_zero_length_cycle_is_skipped_not_a_crash():
    """M3: next_open == open must not divide by zero on every request."""
    board = _board()
    board.add(window_stat(_archive(3, open_round=400, close_round=400)))   # w2's next open == its... 
    board._stats[1]["next_open_round"] = board._stats[1]["open_round"]     # force a zero cycle
    answer = asyncio.run(board.answer(None))
    assert answer["environments"]["math"]["market"]["paid_tokens_per_hour"] > 0
