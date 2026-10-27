"""eval schedule: milestone checkpoints evaluated one at a time on one GPU, idempotent, dry run by default."""

from __future__ import annotations

import json

import pytest

from reliquary.eval.job_grading import JobNotGradable
from reliquary.eval.rl_schedule import NoSamples, Ports, Scheduler, Settings, SuiteEntry, job_id, load_suite

RUN = "5F3sa2TJ-r7-trainer"
SUITE = [SuiteEntry("verifiers-aime25-r0-n30", "aime", 8, 0.02, 32768),
         SuiteEntry("verifiers-gpqa-diamond", "gpqa", 4, 0.02, 32768, prompt_count=100)]


class World:
    """The backups, the job store and the processes the scheduler sees."""

    def __init__(self):
        self.backups, self.incomplete = [], set()
        self.declared, self.drained, self.short = set(), set(), set()
        self.commands, self.graded, self.started, self.stopped, self.feeds = [], [], [], [], []
        self.dead = set()
        self.released, self.release_fails = [], 0
        self.empty, self.grade_fails = set(), set()
        self.now = 1000.0

    def ports(self):
        def run(argv):
            self.commands.append(argv)
            self.declared.add(argv[argv.index("--job-id") + 1])

        def grade(job, *, out, rl, allow_incomplete):
            if job in self.short and not allow_incomplete:
                raise JobNotGradable(f"job {job} misses samples on some problems")
            if job in self.grade_fails:
                raise RuntimeError("bucket unreachable")
            if job in self.empty:
                raise NoSamples(job)
            self.graded.append((job, rl, allow_incomplete))
            return {"complete": not allow_incomplete}

        def start(command, log):
            self.started.append(command)
            return 100 + len(self.started)

        def release(job, drained):
            if self.release_fails:
                self.release_fails -= 1
                raise RuntimeError("registry unreachable")
            self.released.append((job, drained))

        return Ports(backups=lambda: list(self.backups),
                     revision=lambda repo: None if repo in self.incomplete else "f" * 40,
                     fingerprint=lambda repo, rev: "a" * 64, job_exists=lambda job: job in self.declared,
                     drained=lambda job: job in self.drained, run=run, grade=grade, start=start,
                     alive=lambda pid: pid not in self.dead, stop=self.stopped.append,
                     feed=lambda jobs, base: self.feeds.append((jobs, base)) or {"points": len(jobs)},
                     release=release)


def scheduler(tmp_path, world, **overrides):
    settings = Settings(run_id=RUN, run_tag="r7", suite=SUITE, every=50, hf_namespace="ReliquaryForge",
                        hf_prefix="teutonic-rl-r7", model_architecture="Qwen3ForCausalLM", eos_token_id=248046,
                        results_dir=tmp_path / "results", state_path=tmp_path / "state.json",
                        baseline_model="ReliquaryForge/Teutonic-I-Reliquary-SFT", baseline_revision="b" * 40,
                        start_values={"wallet": "w", "hotkey": "h", "port": "9000"}, max_wait_s=3600)
    for key, value in overrides.items():
        setattr(settings, key, value)
    return Scheduler(settings, world.ports(), clock=lambda: world.now)


def test_job_ids_are_deterministic_and_valid():
    assert job_id("r7", 50, "aime") == "eval-r7-s00050-aime"
    assert job_id("r7", None, "gpqa") == "eval-r7-base-gpqa"
    with pytest.raises(ValueError):
        job_id("R7", 50, "aime")
    with pytest.raises(ValueError):
        job_id("r" * 50, 50, "aime")


def test_the_suite_file(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps([{"set_id": "s", "short": "aime", "slots_per_prompt": 8, "cap": 0.02,
                                 "max_new_tokens": 1024}]))
    assert load_suite(path) == [SuiteEntry("s", "aime", 8, 0.02, 1024)]
    for bad in ([], [{"set_id": "s", "short": "AIME", "slots_per_prompt": 8, "cap": 0.02, "max_new_tokens": 1}],
                [{"set_id": "s", "short": "a", "slots_per_prompt": 8, "cap": 0.02, "max_new_tokens": 1,
                  "extra": 1}]):
        path.write_text(json.dumps(bad))
        with pytest.raises((ValueError, TypeError)):
            load_suite(path)


def test_a_dry_run_reads_and_prints_but_changes_nothing(tmp_path):
    world = World()
    lines = scheduler(tmp_path, world).tick()
    assert not (tmp_path / "state.json").exists() and not world.commands and not world.started
    assert any("--job-id eval-r7-base-aime" in line and "--checkpoint-sha256 '<jobs fingerprint>'" in line
               for line in lines)
    assert any(line.startswith("start validator for ReliquaryForge/Teutonic-I-Reliquary-SFT") for line in lines)


def test_baseline_then_the_newest_milestone_one_at_a_time(tmp_path):
    world = World()
    s = scheduler(tmp_path, world)
    s.tick(execute=True)                                   # the baseline: declared and served
    assert [a[a.index("--job-id") + 1] for a in world.commands] == ["eval-r7-base-aime", "eval-r7-base-gpqa"]
    gpqa = world.commands[1]
    assert gpqa[gpqa.index("--prompt-count") + 1] == "100" and gpqa[gpqa.index("--slots-per-prompt") + 1] == "4"
    assert len(world.started) == 1 and "RELIQUARY_TASK_ID=eval-r7-base-aime" in world.started[0]
    assert "RELIQUARY_CORPUS_HOT_JOBS=1" in world.started[0] and "--hotkey h" in world.started[0]

    world.backups = [10, 50, 100, 150, 160]                # training ran ahead meanwhile
    assert any("wait eval-r7-base-aime" in line for line in s.tick(execute=True))
    assert len(world.started) == 1 and not world.graded   # one model at a time: nothing else starts
    world.drained |= {"eval-r7-base-aime", "eval-r7-base-gpqa"}
    lines = s.tick(execute=True)
    assert [g[0] for g in world.graded] == ["eval-r7-base-aime", "eval-r7-base-gpqa"]
    assert world.graded[0][1] == {"run_id": RUN, "step": 0}
    assert world.stopped == [101] and world.feeds == []    # no checkpoint point yet: no feed to publish
    # then straight to the newest milestone, the older ones skipped
    assert "skip step 50: evaluations behind, step 150 is newer" in lines
    assert "skip step 100: evaluations behind, step 150 is newer" in lines
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["active"] == "150" and state["checkpoints"]["50"]["status"] == "skipped"
    assert state["checkpoints"]["150"]["repo"] == "ReliquaryForge/teutonic-rl-r7-step-00000150"
    assert "eval-r7-s00150-aime" in world.declared and len(world.started) == 2

    world.drained |= {"eval-r7-s00150-aime", "eval-r7-s00150-gpqa"}
    world.backups.append(200)
    world.incomplete.add("ReliquaryForge/teutonic-rl-r7-step-00000200")    # still uploading
    lines = s.tick(execute=True)
    assert world.graded[-1] == ("eval-r7-s00150-gpqa", {"run_id": RUN, "step": 150}, False)
    assert world.feeds[-1] == (["eval-r7-s00150-aime", "eval-r7-s00150-gpqa"],
                               ["eval-r7-base-aime", "eval-r7-base-gpqa"])
    assert "backup step 200 not complete yet" in lines and "nothing new to evaluate" in lines
    assert json.loads((tmp_path / "state.json").read_text())["active"] is None

    world.incomplete.clear()
    s.tick(execute=True)
    assert json.loads((tmp_path / "state.json").read_text())["active"] == "200"
    declared = len(world.commands)
    s.tick(execute=True)                                   # idempotent: nothing new, nothing redone
    assert len(world.commands) == declared and len(world.started) == 3


def test_a_job_short_of_samples_waits_then_counts_them_as_failures(tmp_path):
    world = World()
    s = scheduler(tmp_path, world, baseline_model=None)
    world.backups = [50]
    s.tick(execute=True)
    world.drained |= {"eval-r7-s00050-aime", "eval-r7-s00050-gpqa"}
    world.short.add("eval-r7-s00050-gpqa")
    assert any("misses samples" in line for line in s.tick(execute=True))
    assert [g[0] for g in world.graded] == ["eval-r7-s00050-aime"] and not world.stopped
    world.now += 3601
    s.tick(execute=True)
    assert world.graded[-1] == ("eval-r7-s00050-gpqa", {"run_id": RUN, "step": 50}, True)
    assert world.stopped == [101]


def test_a_job_never_drained_is_abandoned_and_a_dead_validator_restarted(tmp_path):
    world = World()
    s = scheduler(tmp_path, world, baseline_model=None)
    world.backups = [50]
    s.tick(execute=True)
    world.dead.add(101)
    assert any("is gone: restart" in line for line in s.tick(execute=True))
    assert len(world.started) == 2
    world.drained.add("eval-r7-s00050-aime")
    world.now += 3601
    lines = s.tick(execute=True)
    assert any(line.startswith("abandon eval-r7-s00050-gpqa") for line in lines)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["checkpoints"]["50"]["abandoned"] == ["eval-r7-s00050-gpqa"]
    assert state["active"] is None and world.stopped == [102]


def test_a_declared_job_is_not_declared_again_and_another_runs_state_is_refused(tmp_path):
    world = World()
    world.declared |= {"eval-r7-base-aime"}
    s = scheduler(tmp_path, world)
    assert "job eval-r7-base-aime already declared" in s.tick(execute=True)
    assert [a[a.index("--job-id") + 1] for a in world.commands] == ["eval-r7-base-gpqa"]
    other = scheduler(tmp_path, world, run_id="other-r1-trainer")
    with pytest.raises(ValueError, match="state of run"):
        other.tick()


def test_eval_schedule_is_a_dry_run_unless_told(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.eval import rl_schedule

    world = World()
    world.backups = [50]
    monkeypatch.setattr(rl_schedule, "live_ports", lambda settings: world.ports())
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps([{"set_id": "verifiers-aime25-r0-n30", "short": "aime", "slots_per_prompt": 8,
                                  "cap": 0.02, "max_new_tokens": 32768}]))
    argv = ["eval", "schedule", "--run-id", RUN, "--run-tag", "r7", "--suite", str(suite), "--every", "50",
            "--hf-prefix", "teutonic-rl-r7", "--model-architecture", "Qwen3ForCausalLM",
            "--eos-token-id", "248046", "--results-dir", str(tmp_path / "results")]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert "[dry run] run reliquary jobs create" in result.output and "eval-r7-s00050-aime" in result.output
    assert not world.commands and not (tmp_path / "results" / "schedule-state.json").exists()
    refused = CliRunner().invoke(app, argv + ["--execute"])
    assert refused.exit_code == 1 and "--wallet-name" in refused.output
    done = CliRunner().invoke(app, argv + ["--execute", "--wallet-name", "w", "--hotkey", "h", "--http-port", "1"])
    assert done.exit_code == 0, done.output
    assert world.started and (tmp_path / "results" / "schedule-state.json").exists()


def test_an_agentic_set_declares_its_episode_harness_and_renderer(tmp_path):
    (tmp_path / "swe").mkdir()
    (tmp_path / "swe" / "episode.json").write_text(json.dumps({"harness": "bash", "max_turns": 48}))
    path = tmp_path / "suite.json"
    agentic = {"set_id": "reliquary-swe-eval-r0-n500-x7-k100-s7", "short": "swe-verified", "slots_per_prompt": 2,
               "cap": 0.02, "max_new_tokens": 8192, "episode_file": "swe/episode.json", "harness": "mini-swe@2",
               "renderer_id": "renderers:qwen38@0.1.11"}
    path.write_text(json.dumps([{"set_id": "s", "short": "aime", "slots_per_prompt": 8, "cap": 0.02,
                                 "max_new_tokens": 32768}, agentic]))
    one_turn, swe = load_suite(path)
    assert swe.episode_file == str(tmp_path / "swe" / "episode.json")       # relative to the suite file
    world = World()
    sch = scheduler(tmp_path, world, suite=[one_turn, swe])
    sch.tick(execute=True)
    plain, episode = world.commands
    assert "--episode-file" not in plain and "--harness" not in plain
    assert plain[plain.index("--renderer-id") + 1] == "chat-template-thinking-v1"
    assert episode[episode.index("--episode-file") + 1] == swe.episode_file
    assert episode[episode.index("--harness") + 1] == "mini-swe@2"
    assert episode[episode.index("--renderer-id") + 1] == "renderers:qwen38@0.1.11"
    assert episode[episode.index("--max-new-tokens") + 1] == "8192"
    assert episode[episode.index("--job-id") + 1] == "eval-r7-base-swe-verified"


@pytest.mark.parametrize("change", [{"harness": "mini-swe@2", "episode_file": None},
                                    {"episode_file": "missing.json"}, {"episode_file": "list.json"}])
def test_an_agentic_set_without_a_readable_episode_is_refused(tmp_path, change):
    (tmp_path / "list.json").write_text("[]")
    entry = {"set_id": "s", "short": "swe", "slots_per_prompt": 2, "cap": 0.02, "max_new_tokens": 8192, **change}
    path = tmp_path / "suite.json"
    path.write_text(json.dumps([entry]))
    with pytest.raises((ValueError, OSError)):
        load_suite(path)


def test_chosen_steps_and_sets_run_exactly_those(tmp_path):
    world = World()
    world.backups = [50, 100, 150]
    s = scheduler(tmp_path, world, baseline_model=None, every=None, steps=(100, 50), only=("gpqa",))
    s.tick(execute=True)
    assert [a[a.index("--job-id") + 1] for a in world.commands] == ["eval-r7-s00100-gpqa"]
    assert "RELIQUARY_TASK_ID=eval-r7-s00100-gpqa" in world.started[0]
    world.drained.add("eval-r7-s00100-gpqa")
    s.tick(execute=True)                                   # graded, then the next asked step, nothing skipped
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["active"] == "50" and "150" not in state["checkpoints"]
    world.drained.add("eval-r7-s00050-gpqa")
    assert "the asked steps are evaluated" in s.tick(execute=True)


def test_a_step_no_longer_on_hf_is_reported(tmp_path):
    world = World()
    world.backups = [50]
    world.incomplete.add("ReliquaryForge/teutonic-rl-r7-step-00000040")
    s = scheduler(tmp_path, world, baseline_model=None, every=None, steps=(40,))
    assert "step 40: no complete backup ReliquaryForge/teutonic-rl-r7-step-00000040" in s.tick(execute=True)
    assert not world.commands


def test_a_set_added_later_runs_on_the_baseline_and_checkpoint_with_their_weights(tmp_path):
    world = World()
    world.backups = [50]
    state = {"run_id": RUN, "active": None, "pid": None, "checkpoints": {"base": {
        "status": "graded", "repo": "ReliquaryForge/Teutonic-I-Reliquary-SFT", "revision": "b" * 40,
        "sha256": "c" * 64, "jobs": ["eval-seeded-sft-aime"], "graded": ["eval-seeded-sft-aime"],
        "abandoned": [], "started_at": 1.0}}}
    (tmp_path / "state.json").write_text(json.dumps(state))
    s = scheduler(tmp_path, world, every=None, steps=(50,))
    s.tick(execute=True)                                   # the seeded aime baseline counts: only gpqa runs
    assert [a[a.index("--job-id") + 1] for a in world.commands] == ["eval-r7-base-gpqa"]
    create = world.commands[0]
    assert create[create.index("--checkpoint-sha256") + 1] == "c" * 64
    assert "RELIQUARY_TASK_ID=eval-r7-base-gpqa" in world.started[0]
    world.drained.add("eval-r7-base-gpqa")
    s.tick(execute=True)
    base = json.loads((tmp_path / "state.json").read_text())["checkpoints"]["base"]
    assert base["graded"] == ["eval-seeded-sft-aime", "eval-r7-base-gpqa"]
    assert world.graded[0][0] == "eval-r7-base-gpqa"


def test_steps_or_every_and_known_sets_only(tmp_path):
    with pytest.raises(ValueError, match="--every"):
        scheduler(tmp_path, World(), every=None)
    with pytest.raises(ValueError, match="not in the suite"):
        scheduler(tmp_path, World(), only=("mmlu",))


def test_jobs_done_with_give_their_pool_share_back(tmp_path):
    world = World()
    s = scheduler(tmp_path, world, baseline_model=None)
    world.backups = [50]
    s.tick(execute=True)
    world.drained.add("eval-r7-s00050-aime")
    world.release_fails = 1
    assert any(line.startswith("release eval-r7-s00050-aime failed") for line in s.tick(execute=True))
    assert not world.released                              # retried on the next tick
    s.tick(execute=True)
    assert world.released == [("eval-r7-s00050-aime", True)]
    world.now += 3601                                      # gpqa never drains: abandoned, cap 0 only
    lines = s.tick(execute=True)
    assert world.released[-1] == ("eval-r7-s00050-gpqa", False) and "released eval-r7-s00050-gpqa" in lines
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["checkpoints"]["50"]["released"] == ["eval-r7-s00050-aime", "eval-r7-s00050-gpqa"]
    s.tick(execute=True)
    assert len(world.released) == 2                        # never released twice


def test_a_job_nobody_mined_is_abandoned_not_scored_zero(tmp_path):
    world = World()
    s = scheduler(tmp_path, world, baseline_model=None)
    world.backups = [50]
    s.tick(execute=True)
    world.drained |= {"eval-r7-s00050-aime", "eval-r7-s00050-gpqa"}  # an empty job is drained too
    world.empty |= {"eval-r7-s00050-aime", "eval-r7-s00050-gpqa"}
    world.now += 3601
    lines = s.tick(execute=True)
    assert not world.graded and not world.feeds
    assert any(line.startswith("abandon eval-r7-s00050-aime: no sample") for line in lines)
    assert world.released == [("eval-r7-s00050-aime", False), ("eval-r7-s00050-gpqa", False)]


def test_a_graded_job_survives_a_crash_later_in_the_tick(tmp_path):
    world = World()
    s = scheduler(tmp_path, world, baseline_model=None)
    world.backups = [50]
    s.tick(execute=True)
    world.drained |= {"eval-r7-s00050-aime", "eval-r7-s00050-gpqa"}
    world.grade_fails.add("eval-r7-s00050-gpqa")
    with pytest.raises(RuntimeError):
        s.tick(execute=True)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["checkpoints"]["50"]["graded"] == ["eval-r7-s00050-aime"]
    world.grade_fails.clear()
    s.tick(execute=True)
    assert [g[0] for g in world.graded] == ["eval-r7-s00050-aime", "eval-r7-s00050-gpqa"]  # aime not again


def test_a_serving_checkpoint_without_a_validator_starts_one(tmp_path):
    world = World()
    s = scheduler(tmp_path, world, baseline_model=None)
    world.backups = [50]
    s.tick(execute=True)
    state = json.loads((tmp_path / "state.json").read_text())
    state["pid"] = None
    (tmp_path / "state.json").write_text(json.dumps(state))
    lines = s.tick(execute=True)
    assert any("is gone: restart" in line for line in lines) and len(world.started) == 2


def test_the_validator_runs_the_merged_contract_of_every_job_it_serves(tmp_path):
    world = World()
    s = scheduler(tmp_path, world, baseline_model=None)
    world.backups = [50]
    s.tick(execute=True)
    assert "tasks contract --task-id eval-r7-s00050-aime --task-id eval-r7-s00050-gpqa >" in world.started[0]
    assert "RELIQUARY_TASK_ID=eval-r7-s00050-aime,eval-r7-s00050-gpqa " in world.started[0]
