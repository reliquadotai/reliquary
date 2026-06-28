from scripts.report_auth_forensics import filter_records, summarize


def _records():
    return [
        {
            "window_start": 10,
            "env_name": "opencodeinstruct",
            "miner_hotkey": "hk_a",
            "prompt_idx": 1,
            "rollout_idx": 0,
            "rollout_reward": 1.0,
            "reward_positive": True,
            "label": "compare_op",
            "p_chosen": 2.0e-4,
            "token_text": " <=",
            "argmax_text": " ==",
            "counterfactual_checked": True,
            "counterfactual_reward": 0.0,
            "counterfactual_reward_flipped": True,
            "code_context": "assert f(x) <= 0",
        },
        {
            "window_start": 11,
            "env_name": "opencodeinstruct",
            "miner_hotkey": "hk_a",
            "prompt_idx": 2,
            "rollout_idx": 1,
            "rollout_reward": 1.0,
            "reward_positive": True,
            "label": "keyword:reverse",
            "p_chosen": 3.0e-4,
            "token_text": "F",
            "argmax_text": "T",
            "code_context": "sorted(xs, reverse=False)",
        },
        {
            "window_start": 12,
            "env_name": "opencodeinstruct",
            "miner_hotkey": "hk_b",
            "prompt_idx": 3,
            "rollout_idx": 2,
            "rollout_reward": 0.0,
            "reward_positive": False,
            "label": "constant:string",
            "p_chosen": 1.0e-5,
            "token_text": " and",
            "argmax_text": "\\n",
            "code_context": "message = 'a and b'",
        },
    ]


def test_filter_records_can_select_high_signal_code_findings():
    rows = filter_records(
        _records(),
        surface="code-semantic",
        code_signal="high",
        labels=[],
        hotkeys=[],
        reward_positive_only=False,
        min_prob_lt=None,
        since_window=None,
        last_n_windows=None,
    )

    assert [row["label"] for row in rows] == ["compare_op"]


def test_filter_records_can_select_high_or_review_positive_recent_findings():
    rows = filter_records(
        _records(),
        surface="code-semantic",
        code_signal="high-or-review",
        labels=[],
        hotkeys=[],
        reward_positive_only=True,
        min_prob_lt=None,
        since_window=11,
        last_n_windows=None,
    )

    assert [row["label"] for row in rows] == ["keyword:reverse"]


def test_code_semantic_report_includes_signal_and_counterfactual_guidance():
    report = summarize(
        _records(),
        top_n=5,
        title="Code-semantic private forensics report",
        surface="code-semantic",
        examples=2,
        threshold_sweep=True,
    )

    assert "Code signal buckets" in report
    assert "- high: records=1 positive=1" in report
    assert "- review: records=1 positive=1" in report
    assert "- low: records=1 positive=0" in report
    assert "Counterfactual regrade" in report
    assert "checked=1 reward_flips=1" in report
    assert "Interpretation" in report
    assert "Counterfactual reward flips found" in report
    assert "Examples" in report


def test_all_token_report_warns_against_punitive_gate():
    rows = [
        {
            "window_start": 10,
            "env_name": "openmathinstruct",
            "miner_hotkey": "hk_math",
            "reward_positive": True,
            "p_chosen": 1.0e-10,
            "token_text": "Thus",
            "argmax_text": "\\",
        },
    ]

    report = summarize(
        rows,
        top_n=3,
        title="All-token private forensics report",
        surface="all-token-shadow",
    )

    assert "broad anomaly smoke test" in report
    assert "Do not use it as a punitive gate" in report
