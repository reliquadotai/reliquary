"""A real grade and a real replay on a real SWE-smith box (Docker host only)."""

import os

import pytest

pytest.importorskip("reliquary_swe")
pytestmark = pytest.mark.skipif(not os.environ.get("RELIQUARY_GRADE_DOCKER"),
                                reason="set RELIQUARY_GRADE_DOCKER=1 on a Docker host with the images")

INSTANCE = "python-openxml__python-docx.0cf6d71f.func_basic__y5p0lk6o"


async def test_the_gold_patch_grades_and_an_empty_episode_replays_to_an_empty_diff():
    from reliquary.validator.agentic_replay import swesmith_task
    from reliquary.validator.corpus_grade_executor import run_grade_item
    from reliquary.validator.corpus_grade_protocol import GradeItem

    gold = swesmith_task(INSTANCE).data.gold_patch
    graded = await run_grade_item(GradeItem(submission_id="a" * 64, task_index=0,
                                            instance_id=INSTANCE, mode="grade", final_diff=gold))
    assert graded == {"status": "ok", "diff_applied": True, "tests_passed": True}
    replayed = await run_grade_item(GradeItem(submission_id="a" * 64, task_index=0,
                                              instance_id=INSTANCE, mode="replay", final_diff=""))
    assert replayed["status"] == "ok" and replayed["replay_diff_equal"] is True


async def test_an_unchanged_checkout_fails_its_tests_and_a_real_action_replays():
    from reliquary.validator.corpus_grade_executor import run_grade_item
    from reliquary.validator.corpus_grade_protocol import GradeItem

    graded = await run_grade_item(GradeItem(submission_id="a" * 64, task_index=0,
                                            instance_id=INSTANCE, mode="grade", final_diff=""))
    assert graded["status"] == "ok" and graded["tests_passed"] is False
    actions = [{"tool": "bash", "arguments": '{"command": "echo hello"}', "observation": "hello"},
               {"tool": "bash", "arguments": '{"command": "echo other"}', "observation": "forged"}]
    replayed = await run_grade_item(GradeItem.model_validate({
        "submission_id": "a" * 64, "task_index": 0, "instance_id": INSTANCE, "mode": "replay",
        "final_diff": "", "actions": actions}))
    assert replayed == {"status": "ok", "replay_diff_equal": True, "observations_compared": 2,
                        "observations_mismatched": [1]}


def _item(actions):
    from reliquary.validator.corpus_grade_protocol import GradeItem

    return GradeItem.model_validate({"submission_id": "a" * 64, "task_index": 0,
                                     "instance_id": INSTANCE, "mode": "replay", "final_diff": "",
                                     "actions": actions})


async def test_calls_outside_the_harness_replay_as_the_harness_answers_them():
    from reliquary.validator.corpus_grade_executor import run_grade_item

    replayed = await run_grade_item(_item([
        {"tool": " bash", "arguments": '{"command": "touch /testbed/x"}',
         "observation": "error: unknown tool ' bash'"},
        {"tool": "bash", "arguments": '{"command ": "touch /testbed/x"}', "observation": ""},
        {"tool": "bash", "arguments": '{"command": "ls /testbed/x"}',
         "observation": "ls: cannot access '/testbed/x': No such file or directory"}]))
    assert replayed == {"status": "ok", "replay_diff_equal": True, "observations_compared": 3,
                        "observations_mismatched": []}


async def test_the_box_runs_under_the_executor_limits():
    from reliquary.validator.agentic_replay import BoxLimits
    from reliquary.validator.corpus_grade_executor import run_grade_item

    limits = BoxLimits(cpu=1.5, memory_gb=1.0, pids=256)
    replayed = await run_grade_item(_item([
        {"tool": "bash", "arguments": '{"command": "cat /sys/fs/cgroup/pids.max"}', "observation": "256"},
        {"tool": "bash", "arguments": '{"command": "cat /sys/fs/cgroup/memory.max"}',
         "observation": str(2 ** 30)},
        {"tool": "bash", "arguments": '{"command": "cat /sys/fs/cgroup/memory.swap.max"}',
         "observation": "0"},
        {"tool": "bash", "arguments": '{"command": "cat /sys/fs/cgroup/cpu.max"}',
         "observation": "150000 100000"}]), limits=limits)
    assert replayed["observations_mismatched"] == [], replayed


async def test_a_memory_hog_is_contained_and_reported_as_a_mismatch():
    from reliquary.validator.agentic_replay import BoxLimits
    from reliquary.validator.corpus_grade_executor import run_grade_item

    replayed = await run_grade_item(_item([
        {"tool": "bash", "arguments": '{"command": "python3 -c \\"b = bytearray(3 * 2**30); print(1)\\""}',
         "observation": "1"},
        {"tool": "bash", "arguments": '{"command": "echo alive"}', "observation": "alive"}]),
        limits=BoxLimits(cpu=1.0, memory_gb=1.0, pids=256))
    assert replayed["status"] == "ok" and replayed["observations_mismatched"] == [0], replayed


async def test_a_fork_bomb_is_contained_and_the_box_is_removed():
    import subprocess

    from reliquary.validator.agentic_replay import (
        BOX_NAME_PREFIX, BoxLimits, ReplayTimeout, replay_swe, swesmith_task,
    )
    from reliquary.corpus.replay_compare import Action

    bomb = Action("bash", '{"command": "python3 -c \\"import os\\nwhile True:\\n  os.fork()\\""}', "")
    try:
        observations, _ = await replay_swe(
            swesmith_task(INSTANCE), [bomb, Action("bash", '{"command": "echo after"}', "after")],
            command_timeout=10, trajectory_budget=180, limits=BoxLimits(cpu=1.0, memory_gb=1.0, pids=128))
    except ReplayTimeout:
        observations = None
    except Exception:  # the box may be too starved to finalize: an executor error, not a crash
        observations = None
    # The bomb starves its own box only: when the replay returns, the next
    # action could not run there.
    assert observations is None or observations[1].strip() != "after"
    # The host still forks, the executor still replays, no box of ours is left.
    assert subprocess.run(["true"]).returncode == 0
    from reliquary.validator.corpus_grade_executor import run_grade_item

    again = await run_grade_item(_item([{"tool": "bash", "arguments": '{"command": "echo ok"}',
                                         "observation": "ok"}]))
    assert again["status"] == "ok" and again["observations_mismatched"] == []
    left = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^{BOX_NAME_PREFIX}"],
                          capture_output=True, text=True).stdout.split()
    assert left == []


def test_the_sweep_removes_orphaned_boxes_only():
    import subprocess

    from reliquary.validator.agentic_replay import BOX_NAME_PREFIX, swesmith_task, sweep_orphan_boxes

    image = swesmith_task(INSTANCE).data.image
    names = [f"{BOX_NAME_PREFIX}orphan-test", "not-ours-grade-sweep-test"]
    for name in names:
        subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "--entrypoint", "sleep",
                        image, "300"], check=True, capture_output=True)
    try:
        assert sweep_orphan_boxes() >= 1
        alive = subprocess.run(["docker", "ps", "-aq", "--filter", f"name={names[0]}"],
                               capture_output=True, text=True).stdout.split()
        other = subprocess.run(["docker", "ps", "-q", "--filter", f"name={names[1]}"],
                               capture_output=True, text=True).stdout.split()
        assert alive == [] and len(other) == 1
    finally:
        subprocess.run(["docker", "rm", "-f", *names], capture_output=True)


async def test_a_max_turns_final_call_replays_into_the_recorded_diff():
    """F1, the shape probed with a real verifiers max_turns=2 episode on this
    host: the harness ran the final turn's call (no observation rendered) and
    the recorded diff holds its file, so the replay runs it too."""
    from reliquary.validator.corpus_grade_executor import run_grade_item

    recorded = ("diff --git a/reliquary_probe_one.txt b/reliquary_probe_one.txt\n"
                "new file mode 100644\nindex 0000000..3815b40\n--- /dev/null\n"
                "+++ b/reliquary_probe_one.txt\n@@ -0,0 +1 @@\n+probe-one\n"
                "diff --git a/reliquary_probe_two.txt b/reliquary_probe_two.txt\n"
                "new file mode 100644\nindex 0000000..6d9bb00\n--- /dev/null\n"
                "+++ b/reliquary_probe_two.txt\n@@ -0,0 +1 @@\n+probe-two\n")
    first = {"tool": "bash", "arguments": '{"command": "echo probe-one > reliquary_probe_one.txt"}',
             "observation": ""}
    final = {"tool": "bash", "arguments": '{"command": "echo probe-two > reliquary_probe_two.txt"}',
             "observation": None}
    item = _item([first, final]).model_copy(update={"final_diff": recorded})
    replayed = await run_grade_item(item)
    assert replayed == {"status": "ok", "replay_diff_equal": True, "observations_compared": 1,
                        "observations_mismatched": []}, replayed
    without = await run_grade_item(_item([first]).model_copy(update={"final_diff": recorded}))
    assert without["replay_diff_equal"] is False           # what dropping it did before F1


# F2 (ruling P23): the routes a fabricated trajectory could use to keep a box
# from judging it, on a real box. Each is now a fact the trajectory caused
# (box_lost / box_timeout, two providers void it unpaid) or a plain mismatch.

def _commands(*commands, final_diff=""):
    import json

    from reliquary.validator.corpus_grade_protocol import GradeItem

    return GradeItem.model_validate({
        "submission_id": "a" * 64, "task_index": 0, "instance_id": INSTANCE, "mode": "replay",
        "final_diff": final_diff,
        "actions": [{"tool": "bash", "arguments": json.dumps({"command": c}), "observation": ""}
                    for c in commands]})


async def test_kill_9_of_pid_1_does_not_kill_the_box():
    # The kernel protects a namespace's init from its own namespace's signals.
    from reliquary.validator.corpus_grade_executor import run_grade_item

    replayed = await run_grade_item(_commands("kill -9 1", "kill -9 -1", "echo after"))
    assert replayed["status"] == "ok" and replayed["observations_mismatched"] == [2]


async def test_rm_rf_git_leaves_an_empty_diff_that_mismatches_a_claimed_one():
    from reliquary.validator.corpus_grade_executor import run_grade_item

    replayed = await run_grade_item(_commands("rm -rf .git", final_diff="diff --git a/x b/x\n"))
    assert replayed["status"] == "ok" and replayed["replay_diff_equal"] is False


async def test_an_action_that_breaks_the_box_is_box_lost():
    from reliquary.validator.corpus_grade_executor import run_grade_item

    replayed = await run_grade_item(_commands("rm -f /bin/sh /usr/bin/sh /bin/dash /usr/bin/dash",
                                              "echo after"))
    assert replayed["status"] == "box_lost", replayed


async def test_a_long_sleep_spends_the_deadline_as_the_trajectorys_box_timeout():
    import functools

    from reliquary.validator.agentic_replay import replay_swe
    from reliquary.validator.corpus_grade_executor import run_grade_item

    replayed = await run_grade_item(_commands("echo first", "sleep 99999"),
                                    replay=functools.partial(replay_swe, trajectory_budget=60))
    assert replayed["status"] == "box_timeout", replayed


class _ShortScoring:
    def __init__(self, task, seconds):
        self._task = task
        self.data = task.data.model_copy(update={
            "timeout": task.data.timeout.model_copy(update={"scoring": seconds})})

    def __getattr__(self, name):
        return getattr(self._task, name)


async def test_hanging_tests_after_the_patch_are_the_trajectorys_box_timeout():
    from reliquary.validator.agentic_replay import swesmith_task
    from reliquary.validator.corpus_grade_executor import run_grade_item
    from reliquary.validator.corpus_grade_protocol import GradeItem

    # Source the tests import (a root conftest.py is cleaned by the test restoration).
    hang = ("diff --git a/src/docx/__init__.py b/src/docx/__init__.py\n"
            "index 2052210..fb6c683 100644\n--- a/src/docx/__init__.py\n+++ b/src/docx/__init__.py\n"
            "@@ -60,3 +60,5 @@ del (\n     StylesPart,\n     part_class_selector,\n )\n"
            "+import time\n+time.sleep(99999)\n")
    task = _ShortScoring(swesmith_task(INSTANCE), 90)
    graded = await run_grade_item(
        GradeItem(submission_id="a" * 64, task_index=0, instance_id=INSTANCE, mode="grade",
                  final_diff=hang), task_for=lambda _: task)
    assert graded["status"] == "box_timeout", graded


async def test_a_setup_slower_than_its_deadline_is_the_executors_timeout():
    """Ruling P25: provisioning and setup (pip install uv, uv sync) run under
    their own deadline; missing it is re-leased, never the trajectory's."""
    import functools

    from reliquary.validator.agentic_replay import replay_swe
    from reliquary.validator.corpus_grade_executor import run_grade_item

    replayed = await run_grade_item(_commands("echo first"),
                                    replay=functools.partial(replay_swe, setup_deadline=3))
    assert replayed["status"] == "timeout" and "setup deadline" in replayed["detail"], replayed


async def test_a_box_whose_image_declares_a_volume_is_refused_as_the_executors_error():
    """Ruling P25: a VOLUME escapes the box's disk limit; such an image is
    refused at box start (before any action), an executor error."""
    import subprocess

    from reliquary.validator.agentic_replay import swesmith_task
    from reliquary.validator.corpus_grade_executor import run_grade_item

    task = swesmith_task(INSTANCE)
    tag = "reliquary-test/volume-image:1"
    subprocess.run(["docker", "build", "-q", "-t", tag, "-"], check=True, capture_output=True,
                   input=f"FROM {task.data.image}\nVOLUME /data\n".encode())

    class WithVolume:
        def __init__(self):
            self.data = task.data.model_copy(update={"image": tag})

        def __getattr__(self, name):
            return getattr(task, name)

    try:
        replayed = await run_grade_item(_commands("echo hi"), task_for=lambda _: WithVolume())
        assert replayed["status"] == "error" and "volume" in replayed["detail"], replayed
        clean = await run_grade_item(_commands("echo hi"))
        assert clean["status"] == "ok"
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
