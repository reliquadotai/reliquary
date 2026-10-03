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
