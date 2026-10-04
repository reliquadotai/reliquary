import json

from reliquary.corpus.replay_compare import (
    Action, ReplayReport, actions_from_trace, compare, normalize,
)


def _trace(*turns):
    nodes = [{"message": {"role": "system", "content": "s"}}, {"message": {"role": "user", "content": "u"}}]
    for tool, args, out in turns:
        nodes.append({"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "name": tool, "arguments": args}]}})
        nodes.append({"message": {"role": "tool", "content": out}})
    nodes.append({"message": {"role": "assistant", "content": "done"}})
    return {"nodes": nodes}


def test_actions_follow_the_trace_order():
    t = _trace(("bash", json.dumps({"command": "ls"}), "a\nb\n"),
               ("edit", json.dumps({"path": "x", "old_str": "1", "new_str": "2"}), "Edited x"))
    assert actions_from_trace(t) == [
        Action("bash", '{"command": "ls"}', "a\nb\n"),
        Action("edit", '{"path": "x", "old_str": "1", "new_str": "2"}', "Edited x"),
    ]


def test_openai_style_function_field_is_read_too():
    t = _trace()
    t["nodes"].insert(2, {"message": {"role": "assistant", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}})
    t["nodes"].insert(3, {"message": {"role": "tool", "content": "x"}})
    assert actions_from_trace(t) == [Action("bash", "{}", "x")]


def test_normalize_hides_durations_and_carriage_returns():
    a = "== 34 passed in 0.75s ==\r\n"
    b = "== 34 passed in 1.02s ==\n"
    assert normalize(a) == normalize(b)


def test_compare_counts_mismatches_by_index():
    rec = [Action("bash", "{}", "same"), Action("bash", "{}", "old")]
    report = compare(rec, ["same", "new"], "diff", "diff")
    assert report == ReplayReport(compared=2, mismatched=[1], diff_equal=True)
    assert report.mismatch_share == 0.5


def test_a_short_replay_counts_missing_observations_as_mismatched():
    rec = [Action("bash", "{}", "a"), Action("bash", "{}", "b")]
    assert compare(rec, ["a"], "d", "d").mismatched == [1]


def test_non_utf8_replacement_characters_compare():
    rec = [Action("bash", "{}", "x�")]
    assert compare(rec, ["x�"], "", "").mismatched == []


def test_normalize_hides_object_addresses():
    assert normalize("<Raise l.1 at 0x73afef757920>") == normalize("<Raise l.1 at 0x758af2b72690>")


def test_normalize_hides_the_base_commit_hash():
    assert normalize("88dd52e base\n") == normalize("5143544 base\n")
    assert normalize("c7816e1 base") != normalize("c7816e1 other")


def test_normalize_hides_hash_in_built_version_strings():
    assert normalize("Project version: 0+untagged.1.g07cd5ea") == normalize("Project version: 0+untagged.1.g83c8afb")


def test_normalize_hides_ls_mtimes_but_not_sizes_or_names():
    a = "-rw-r--r-- 1 root root   664 Oct  2 19:36 /testbed/src/docx/image/__init__.py"
    b = "-rw-r--r-- 1 root root   664 Oct  3 09:31 /testbed/src/docx/image/__init__.py"
    assert normalize(a) == normalize(b)
    assert normalize(a) != normalize(b.replace("664", "665"))


def test_normalize_hides_date_output():
    assert normalize("Fri Oct  2 19:49:31 UTC 2026") == normalize("Sat Oct  3 09:32:48 UTC 2026")


def test_normalize_hides_git_show_commit_and_date():
    a = "commit 5b26444a37283a20ca4cef1cc14ef10498b85cb9\nAuthor: reliquary-swe <r@localhost>\nDate:   Fri Oct  2 19:49:46 2026 +0000\n"
    b = "commit ea9e4f8435e5ae79a89adf40ba36335089d60ff2\nAuthor: reliquary-swe <r@localhost>\nDate:   Sat Oct  3 09:32:51 2026 +0000\n"
    assert normalize(a) == normalize(b)
    assert normalize(a) != normalize(b.replace("Author: reliquary-swe", "Author: someone"))


# --- per-episode tolerance, from gate M2's recorded honest episodes ---------

from pathlib import Path  # noqa: E402

from reliquary.corpus.replay_compare import allowed_mismatches, within_tolerance  # noqa: E402

_M2 = Path(__file__).resolve().parents[2] / "docs/design/measurements/2026-10-03-m2-replay-agreement.json"


def _m2_episodes():
    return json.loads(_M2.read_text())["episodes"]


def test_allowed_mismatches_floor_then_share():
    assert allowed_mismatches(0) == 5
    assert allowed_mismatches(15) == 5
    assert allowed_mismatches(42) == 6
    assert allowed_mismatches(100) == 12


def test_every_honest_m2_episode_is_within_tolerance_with_two_to_spare():
    episodes = _m2_episodes()
    assert len(episodes) == 66
    for e in episodes:
        report = ReplayReport(compared=e["actions"], mismatched=list(e["mismatched"]),
                              diff_equal=e["diff_equal"])
        assert within_tolerance(report), e["instance_id"]
        assert allowed_mismatches(e["actions"]) - len(e["mismatched"]) >= 2


def test_real_m2_worst_pairs_are_timing_only_and_stay_within_tolerance():
    # The worst share in M2: 3 of 15 observations differ (20 %). Its visible
    # differences are pytest timing (the --durations listing, the clock of a
    # long run), which the B1 rules hide; the samples are truncated, so the
    # third pair's difference lies past the cut.
    worst = max(_m2_episodes(), key=lambda e: len(e["mismatched"]) / e["actions"])
    assert (len(worst["mismatched"]), worst["actions"]) == (3, 15)
    samples = {s["index"]: s for s in worst["samples"]}
    raw = [i for i, x in samples.items() if x["recorded"] != x["replayed"]]
    assert raw == [12, 13]
    recorded, replayed = [], []
    for i in range(worst["actions"]):
        s = samples.get(i)
        recorded.append(Action("bash", s["arguments"] if s else "{}", s["recorded"] if s else f"o{i}"))
        replayed.append(s["replayed"] if s else f"o{i}")
    report = compare(recorded, replayed, "diff", "diff")
    assert report.mismatched == []
    assert within_tolerance(ReplayReport(compared=worst["actions"], mismatched=list(worst["mismatched"]),
                                         diff_equal=True))


def test_too_many_mismatches_or_a_different_diff_fail():
    assert not within_tolerance(ReplayReport(compared=15, mismatched=list(range(6)), diff_equal=True))
    assert within_tolerance(ReplayReport(compared=15, mismatched=list(range(5)), diff_equal=True))
    assert not within_tolerance(ReplayReport(compared=15, mismatched=[], diff_equal=False))
    assert not within_tolerance(ReplayReport(compared=100, mismatched=list(range(13)), diff_equal=True))


def test_a_hex_literal_in_file_content_is_not_hidden():
    assert normalize("MAGIC = 0xdeadbeef00") != normalize("MAGIC = 0xdeadbeef01")


def test_a_duration_inside_an_identifier_or_a_bare_spaced_s_is_not_hidden():
    assert normalize("model_2.5s_v1") != normalize("model_3.5s_v1")
    assert normalize("took 12 s") != normalize("took 13 s")
    assert normalize("took 12 ms") == normalize("took 13 ms")
    assert normalize("took 0.75s") == normalize("took 0.80s")


def test_a_real_ls_mtime_and_commit_header_are_still_hidden():
    ls = "-rw-r--r-- 1 root root 1234 Oct  2 19:36 setup.py"
    assert normalize(ls) == normalize(ls.replace("Oct  2 19:36", "Nov 13 08:01"))
    assert normalize(ls) != normalize(ls.replace("19:36", "19:365"))
    c = "commit 5b26444a37283a20ca4cef1cc14ef10498b85cb9\nAuthor: x"
    assert normalize(c) == normalize(c.replace("5b26", "6c37"))


def test_an_mtime_outside_an_ls_line_is_not_hidden():
    assert normalize("released Oct  2 19:36") != normalize("released Oct  3 09:31")


def test_a_hash_after_commit_mid_line_is_not_hidden():
    a = "see commit 5b26444a37283a20ca4cef1cc14ef10498b85cb9 for details"
    assert normalize(a) != normalize(a.replace("5b26", "6c37"))


def test_an_action_without_a_recorded_observation_is_replayed_but_not_compared():
    from reliquary.corpus.replay_compare import compare

    report = compare([Action("bash", "{}", "x"), Action("bash", "{}", None)], ["x", "y"], "d", "d")
    assert report.compared == 1 and report.mismatched == [] and report.diff_equal


# --- B1 (2026-10-04): honest 27B mismatches measured on e2e + M2 episodes ---

def _same(recorded: str, replayed: str) -> bool:
    return compare([Action("bash", "{}", recorded)], [replayed], "d", "d").mismatched == []


def test_normalize_hides_the_behave_run_time_only():
    assert normalize("5 features passed\nTook 0m1.753s") == normalize("5 features passed\nTook 0m1.344s")
    assert normalize("Took 2m0.078s") == "Took <dur>"
    assert normalize("Took 3 apples") == "Took 3 apples"
    assert normalize("x Took 0m1.7s") != normalize("x Took 0m1.3s")       # line-anchored


def test_normalize_hides_pytest_clock_suffix_of_long_runs():
    a = "1670 passed, 21 skipped in 70.12s (0:01:10)"
    b = "1670 passed, 21 skipped in 71.40s (0:01:11)"
    assert normalize(a) == normalize(b) == "1670 passed, 21 skipped in <dur>"
    assert normalize("1670 passed (0:01:10)") == "1670 passed (0:01:10)"   # only after a duration


def test_normalize_hides_pytest_slowest_durations_entries_and_hidden_count():
    a = ("0.31s teardown pandas/tests/test_a.py::test_x\n0.20s call     pandas/tests/test_b.py::test_y[int]\n\n"
         "(14 durations < 0.005s hidden.  Use -vv to show these durations.)\n1413 passed in 9.10s")
    b = ("0.41s call     pandas/tests/test_c.py::test_z[a-b c]\n0.25s setup    pandas/tests/test_d.py::test_w\n\n"
         "(13 durations < 0.005s hidden.  Use -vv to show these durations.)\n1413 passed in 9.80s")
    assert normalize(a) == normalize(b)
    assert normalize("1413 passed in 9.10s") != normalize("1412 passed in 9.10s")
    # Only the slowest-durations entry shape: a phase word and a test id.
    assert normalize("0.31s call test_a.py::t FAILED") != normalize("0.31s call test_a.py::t PASSED")


def test_normalize_hides_ninja_step_targets_but_not_other_lines():
    a = "[143/152] Compiling C object pandas/_libs/join.so.p/join.pyx.c.o\n[144/152] Linking target pandas/_libs/join.so"
    b = "[143/152] Linking target pandas/_libs/interval.so\n[144/152] Compiling C++ object pandas/_libs/w.so.p/a.cpp.o"
    assert normalize(a) == normalize(b)
    assert normalize("[15/152] Generating pandas/_libs/algos_pxi with a custom command") == \
        normalize("[15/152] Generating pandas/_libs/index_pxi with a custom command")
    assert normalize("[1/2] Linking target a.so") != normalize("[2/2] Linking target a.so")   # step kept
    assert normalize("[1/2] Linking target a.so ALL TESTS PASS") != normalize("[1/2] Linking target b.so")
    assert normalize("[1/2] FAILED: a.o") != normalize("[1/2] FAILED: b.o")


def test_reordered_grep_and_find_lines_compare_equal():
    # Directory order is the host filesystem's (xfs keeps insertion order,
    # ext4 hashes with a per-filesystem seed), so `grep -r` / `find` list the
    # same hits in another order on another host.
    rec = ("/testbed/src/a.py:73:    def f(self):\n/testbed/src/b/c.py:263:    def g():\n---\n"
           "src/docx/x.py\nsrc/docx/oxml/y.py")
    rep = ("/testbed/src/b/c.py:263:    def g():\n/testbed/src/a.py:73:    def f(self):\n---\n"
           "src/docx/oxml/y.py\nsrc/docx/x.py")
    assert _same(rec, rep)


def test_reordering_is_only_within_a_run_of_path_lines():
    # Not across a separator,
    assert not _same("/t/a.py:1:x\n---\n/t/b.py:2:y", "/t/b.py:2:y\n---\n/t/a.py:1:x")
    # not for other lines (file contents),
    assert not _same("import os\nimport sys\nx = a/b", "import sys\nimport os\nx = a/b")
    assert not _same("    return a/b\n    x = 1", "    x = 1\n    return a/b")
    # and never a changed, added or dropped hit.
    assert not _same("/t/a.py:1:x\n/t/b.py:2:y", "/t/b.py:2:y\n/t/a.py:1:z")
    assert not _same("/t/a.py:1:x\n/t/b.py:2:y", "/t/b.py:2:y")
    assert not _same("/t/a.py\n/t/b.py", "/t/b.py\n/t/a.py\n/t/c.py")
