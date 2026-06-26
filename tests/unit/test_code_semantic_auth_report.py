import gzip
import json

from scripts.report_code_semantic_auth import (
    format_text_report,
    load_archives_from_paths,
    summarize_archives,
)


def _archive(window=10):
    return {
        "window_start": window,
        "environment": "openmathinstruct",
        "environments": ["openmathinstruct", "opencodeinstruct"],
        "batch": [
            {
                "hotkey": "hk_code_a",
                "env_name": "opencodeinstruct",
                "code_semantic_auth_findings": 2,
                "code_semantic_auth_min_prob": 0.0002,
                "code_semantic_auth_positive_findings": 1,
                "code_semantic_auth_positive_min_prob": 0.0003,
                "all_token_auth_shadow_findings": 3,
                "all_token_auth_shadow_min_prob": 4e-7,
                "all_token_auth_shadow_positive_findings": 2,
                "all_token_auth_shadow_positive_min_prob": 5e-6,
            },
            {
                "hotkey": "hk_math",
                "env_name": "openmathinstruct",
                "code_semantic_auth_findings": 99,
                "code_semantic_auth_min_prob": 1e-9,
                "all_token_auth_shadow_findings": 99,
                "all_token_auth_shadow_min_prob": 1e-9,
            },
        ],
        "runners_up": [
            {
                "hotkey": "hk_code_a",
                "env_name": "opencodeinstruct",
                "code_semantic_auth_findings": 0,
                "code_semantic_auth_min_prob": None,
                "all_token_auth_shadow_findings": 0,
                "all_token_auth_shadow_min_prob": 0.02,
            },
            {
                "hotkey": "hk_code_b",
                "env_name": "opencodeinstruct",
                "code_semantic_auth_findings": 1,
                "code_semantic_auth_min_prob": 8e-7,
                "code_semantic_auth_positive_findings": 0,
                "code_semantic_auth_positive_min_prob": None,
                "all_token_auth_shadow_findings": 2,
                "all_token_auth_shadow_min_prob": 7e-7,
                "all_token_auth_shadow_positive_findings": 0,
                "all_token_auth_shadow_positive_min_prob": None,
            },
        ],
    }


def test_summarize_archives_filters_env_and_counts_selected_and_runners_up():
    summary = summarize_archives([_archive()])
    data = summary.as_dict()

    assert data["windows"] == 1
    assert data["entries_seen"] == 4
    assert data["entries_matching_env"] == 3
    assert data["selected"]["submissions"] == 1
    assert data["selected"]["flagged_submissions"] == 1
    assert data["selected"]["findings"] == 2
    assert data["selected"]["min_prob"] == 0.0002
    assert data["selected"]["positive_findings"] == 1
    assert data["selected"]["positive_min_prob"] == 0.0003
    assert data["runners_up"]["submissions"] == 2
    assert data["runners_up"]["flagged_submissions"] == 1
    assert data["runners_up"]["findings"] == 1
    assert data["runners_up"]["min_prob"] == 8e-7
    assert data["by_hotkey"]["hk_code_a"]["submissions"] == 2
    assert data["by_hotkey"]["hk_code_a"]["findings"] == 2
    assert data["by_hotkey"]["hk_code_a"]["positive_findings"] == 1
    assert data["by_hotkey"]["hk_code_b"]["findings"] == 1
    assert data["by_hotkey"]["hk_code_b"]["positive_findings"] == 0
    assert data["by_window"]["10"]["findings"] == 3
    assert data["by_window"]["10"]["positive_findings"] == 1


def test_summarize_archives_can_exclude_runners_up():
    summary = summarize_archives([_archive()], include_runners_up=False)
    data = summary.as_dict()

    assert data["entries_seen"] == 2
    assert data["entries_matching_env"] == 1
    assert data["selected"]["findings"] == 2
    assert data["selected"]["positive_findings"] == 1
    assert data["runners_up"]["submissions"] == 0
    assert "hk_code_b" not in data["by_hotkey"]


def test_summarize_archives_can_read_all_token_shadow_surface():
    summary = summarize_archives(
        [_archive()],
        field_prefix="all_token_auth_shadow",
    )
    data = summary.as_dict()

    assert data["entries_matching_env"] == 3
    assert data["selected"]["flagged_submissions"] == 1
    assert data["selected"]["findings"] == 3
    assert data["selected"]["min_prob"] == 4e-7
    assert data["selected"]["positive_findings"] == 2
    assert data["selected"]["positive_min_prob"] == 5e-6
    assert data["runners_up"]["submissions"] == 2
    assert data["runners_up"]["flagged_submissions"] == 1
    assert data["runners_up"]["findings"] == 2
    assert data["runners_up"]["min_prob"] == 7e-7
    assert data["by_hotkey"]["hk_code_b"]["findings"] == 2


def test_format_text_report_includes_recommendation():
    summary = summarize_archives([_archive()])
    report = format_text_report(summary, top_n=1)

    assert "OpenCode semantic-token shadow report" in report
    assert "opencode_entries: 3" in report
    assert "flagged=2" in report
    assert "positive_findings=1" in report
    assert "hk_code_a" in report
    assert "shadow findings present" in report


def test_format_text_report_accepts_custom_surface_title():
    summary = summarize_archives(
        [_archive()],
        field_prefix="all_token_auth_shadow",
    )
    report = format_text_report(
        summary,
        top_n=1,
        title="All-token argmax-shadow report",
    )

    assert "All-token argmax-shadow report" in report
    assert "findings=5" in report


def test_load_archives_from_json_and_gzip_paths(tmp_path):
    archive = _archive(window=11)
    plain = tmp_path / "window-11.json"
    gzipped = tmp_path / "window-12.json.gz"
    plain.write_text(json.dumps(archive))
    gzipped.write_bytes(gzip.compress(json.dumps(_archive(window=12)).encode()))

    archives = load_archives_from_paths([str(plain), str(gzipped)])

    assert [a["window_start"] for a in archives] == [11, 12]
