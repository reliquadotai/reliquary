# RL environment settings carried by the task, not the compiled profile

Status: mapping and plan; nothing changed.

## Why

An evaluation or batch order carries its own generation settings (sampling, `max_tokens`, frozen prompts). An RL
training does not: it may only name environments of a compiled profile (`protocol/profiles.py`), whose per-env
entries (`EnvironmentProfile`: `max_new_tokens`, `thinking`, `prompt_cooldown_windows`, `batch_target`, episode
limits) every process reads from its own `ACTIVE_PROTOCOL_PROFILE`. That came from the first subnet, where peers
with no coordinator had to agree byte for byte. Async RL already has one controller per task, leases and draw
audits: the settings can travel with the task, as an order's do. A customer env then trains like any other.

## How a process gets its profile today

`resolve_protocol_profile()` (`profiles.py:1590`) runs once at import: `RELIQUARY_TASK_CONTRACT` (a contract file:
the task's carried profile) wins, else `RELIQUARY_PROTOCOL_PROFILE`, else the default. `constants.py` then derives
module-level values from it. So the controller and the trainer can already run a per-task contract (the trainer
adopts the controller's through `/rl/trainer/config`, `rl_async_service.trainer_task_profile`).

The gap is the **miner**: one miner process serves leases of several tasks, and a lease carries only the env name
(`miner/rl_async.HeldLease`: `environment, prompt_idx, checkpoint, nonce, ...`). The miner renders and caps by its
own compiled profile (`rl_async.py:1280-1318`: `thinking_for_environment`, `max_new_tokens_for_environment`). An
env its profile does not know gets the defaults (thinking on, `MAX_NEW_TOKENS_PROTOCOL_CAP`); if the controller
disagrees, the prompt digest or the length check fails and an honest miner is refused.

## The reads, sorted

85 lines reading the compiled profile outside `profiles.py` (`git grep ACTIVE_PROTOCOL_PROFILE\|PROFILES\[`), plus
36 calls of the per-env getters (`max_new_tokens_for_environment`, `thinking_for_environment`,
`prompt_cooldown_windows_for_environment`, `episode_limits_for_environment`).

| group | where | for a per-task env |
|---|---|---|
| **A. per-env generation settings, async RL** | miner `rl_async.py` (6 calls), `protocol/tokens.py` (prompt render, 1), validator `rl_async_service.py` (1), `rl_submit.py` (2), `verifier.py` (4), `admission.py` (5), `validator/training.py` (2) | **must** come from the task: the lease names it, both sides read the same settings |
| **B. training order checks** | `admin/training.py:101,160`, `rl_orders/preflight.py:117`, `rl_orders/supervisor.py:208` | take envs from the order (profile = template) |
| **C. process-wide run constants** | `constants.py` model id/revision, sampling (`T_PROTO`, `TOP_P`, `TOP_K`, draw, `lm_head_fp32`), rollouts `M_ROLLOUTS`, prompt encoding, protocol version; `grader/__init__.py` pool size | stay: they are the run's, and the controller already carries them in its contract |
| **D. legacy window path** | `miner/engine.py`, `episode_group_miner.py`, `validator/service.py`, `batcher.py`, `server.py`, `quarantine.py` | untouched: the window RL is not what training orders run |
| **E. corpus/generation (orders)** | `corpus_validator.py`, `corpus_split.py`, `corpus_service.py`, `generation_control.py`, `corpus/job.py` | untouched: orders already carry their contract |
| **F. tools and history** | `cli/main.py` (toploc proof, mix), `protocol/composition.py` (named past profiles), `reliquarylogic.py` | untouched |

Only groups A and B change.

## Plan

1. **`EnvSettings`** (one dataclass: `max_new_tokens`, `thinking`, `prompt_cooldown_windows`, `batch_target`,
   episode limits) and **`task_env_settings(entry) -> {env: EnvSettings}`**: the profile's entries for its own
   envs, overridden or extended by the order's `environments[]`. The task entry stores the result and its digest.
2. **Controller**: serves it (`GET /rl/task/env-settings`, with the digest) and puts the digest on each lease.
3. **Miner**: fetches the task's settings once per digest (cached), and renders and caps each lease by them
   instead of its compiled profile. A lease whose digest it cannot fetch is released, not guessed.
4. **Validator side of A**: `rl_submit`, `verifier`, `admission`, `validator/training` read the task's settings
   through the same function. The getters in `constants.py` gain a `settings=` argument; the compiled default stays
   for group D.
5. **B**: `training_refusal` accepts an env outside the profile when the order gives its settings and the env is
   either catalog-RL or a customer version qualified for `rl` and owned by the order's account; `training_entry`
   splits over the order's envs, not the profile's.
6. **Tests**: miner and validator agree on render and cap for an env only the task declares; a lease whose
   settings digest changed is refused before generation; a customer env trains end to end on fakes.
7. **GPU smoke** (one H100, `teutonic` profile + one customer env, a few steps) before any customer.

Estimate: 3-4 days of code and tests, plus the smoke. Risk: a group A read left on the compiled profile; the
miner/validator agreement test of step 6 is the guard.
