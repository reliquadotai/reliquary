"""Ruling P24 on a real Docker host whose daemon bounds every box's writable
layer: overlay2 on xfs mounted with pquota, ``"storage-opts":
["overlay2.size=2G"]``. Point DOCKER_HOST at such a daemon (sandbox-dev-01:
a loopback xfs image under /opt/xfsq with its own dockerd) and set
RELIQUARY_GRADE_DOCKER_XFS=1."""

import os
import shutil

import pytest

pytest.importorskip("reliquary_swe")
pytestmark = pytest.mark.skipif(not os.environ.get("RELIQUARY_GRADE_DOCKER_XFS"),
                                reason="set RELIQUARY_GRADE_DOCKER_XFS=1 with DOCKER_HOST on an "
                                       "overlay2/xfs-pquota daemon with overlay2.size=2G")

INSTANCE = "python-openxml__python-docx.0cf6d71f.func_basic__y5p0lk6o"


def _action(command, observation=""):
    import json

    from reliquary.corpus.replay_compare import Action

    return Action("bash", json.dumps({"command": command}), observation)


def test_the_start_check_accepts_the_bounded_daemon_and_refuses_a_smaller_limit():
    from reliquary.validator.corpus_grade_executor import docker_disk_refusal, docker_storage_refusal

    assert docker_storage_refusal() is None
    assert docker_disk_refusal(2.0) is None
    assert "overlay2.size" in docker_disk_refusal(1.0)


async def test_a_disk_filler_fills_its_own_box_only():
    from reliquary.validator.agentic_replay import BoxLimits, BoxLost, replay_swe, swesmith_task

    root = "/opt/xfsq/mnt"
    free_before = shutil.disk_usage(root).free if os.path.isdir(root) else None
    limits = BoxLimits(cpu=1.0, memory_gb=2.0, pids=256, disk_gb=2.0)
    task = swesmith_task(INSTANCE)
    fill = "dd if=/dev/zero of=/tmp/fill bs=1M count=6000 2>&1 | tail -3"
    observations, _ = await replay_swe(
        task, [_action(fill + "; rm -f /tmp/fill"), _action("echo alive", "alive")], limits=limits)
    # dd stopped at the quota: well short of the 6000 MiB it asked for.
    assert "records out" in observations[0] and "6000+0 records out" not in observations[0], \
        observations[0]
    assert observations[1].strip() == "alive"
    if free_before is not None:
        assert free_before - shutil.disk_usage(root).free < 2.5 * 2 ** 30
    # Left full, the box cannot take the next request: the trajectory lost it.
    with pytest.raises(BoxLost, match="No space left"):
        await replay_swe(task, [_action(fill), _action("echo alive", "alive")], limits=limits)


async def test_a_flood_of_output_is_cut_one_char_past_the_lease_bound():
    from reliquary.validator.agentic_replay import BoxLimits, replay_swe, swesmith_task
    from reliquary.validator.corpus_grade_protocol import MAX_OBSERVATION_CHARS

    flood = "head -c 50000000 /dev/zero | tr '\\0' x"
    observations, _ = await replay_swe(
        swesmith_task(INSTANCE),
        [_action(flood),
         _action(flood + " > /proc/$PPID/fd/1"),   # behind the tool program's back
         _action("echo alive", "alive")],
        limits=BoxLimits(cpu=1.0, memory_gb=2.0, pids=256, disk_gb=2.0))
    assert len(observations[0]) == MAX_OBSERVATION_CHARS + 1
    assert len(observations[1]) <= MAX_OBSERVATION_CHARS + 1
    assert observations[2].strip() == "alive"
