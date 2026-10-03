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
