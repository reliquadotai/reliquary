# Sandbox capacity follows the pipeline (agentic orders)

Date: 2026-11-12. Status: design, implementation on `feat/sandbox-autoscale` (Catalyst) and
`feat/sandbox-autoscale` (platform).

## Problem

An agentic order runs through three stages:

```
miners (GPU generation) -> sandbox gateways (CPU: tools, grading) -> audit executors (GPU)
```

- Audit executors already scale with the work: the control measures one executor's throughput and asks for
  `executors_for_backlog` executors (`reliquary/validator/eval_control.py`); the platform rents them.
- Miners are already bounded by the sandbox fleet: the session quota (`reliquary/sandbox/corpus_quota.py`)
  refuses an open when the fleet is full.
- Gateways do not scale with the work. The platform rents by a fixed rule (`worker/sandbox-gateways.ts`):
  per order `min(rows, GATEWAY_ORDER_SESSIONS = 78)` concurrent sessions, `GATEWAY_CAPACITY = 39` per gateway,
  at most the 5 certificate hostnames. If audit and miners could go faster, the CPU caps them.

Goal: CPU is never the stage that limits an agentic order, within the order's budget and the fleet's limits.

## Rule

Little's law: concurrent sessions needed = target episode rate x episode wall time.

Per running agentic order (job):

- `episode_seconds`: measured median wall time of the job's finished signed episodes (session open to final
  record), over the last `N` (default 50); before any finished, the set's default (platform var, today 900 s).
- `audit_rate`: episodes per second the audit can decide = the job model's measured audited tokens per second
  per executor x executors wanted for it / mean tokens per episode of the job (measured; unknown before the
  first audits).
- `deadline_rate`: episodes left / seconds left to the order's deadline (platform: `created_at + max_hours`
  minus a safety margin for audit drain and delivery).
- `target_rate = max(deadline_rate, audit_rate)`, so the order meets its deadline and never leaves audit
  capacity idle; the platform caps it by what the order's quote paid for (gateway-hours) and the daily spend.
- `sessions_wanted = ceil(target_rate x episode_seconds x MARGIN)` (MARGIN default 1.3), at least 1 while
  episodes remain, at most `episodes_left`.
- Gateways per image group = `ceil(sum(sessions_wanted) / GATEWAY_CAPACITY)` over the group's running orders,
  bounded by the hostnames and the spend caps. Scale down only after a hold time (like `ExecutorSizing`).

The fixed `GATEWAY_ORDER_SESSIONS` stays only as the fallback while the control reports nothing for a job.

## Who computes what

The control measures; the platform decides (it holds the deadline, budget and fleet).

Control status (`GET /admin/v1/eval-control/status`, written every 30 s) gains, per job with an episode:

```json
"jobs": {"<job_id>": {"...": "...",
  "sandbox": {
    "episodes_total": 400, "episodes_finished": 120, "episodes_left": 280,
    "sessions_held": 61,
    "episode_seconds": 840.0,            // median of the last N finished, null before any
    "tokens_per_episode": 41000.0,       // mean audited (prompt+completion) tokens, null before any
    "audit_episodes_per_second": 0.07,   // null while audit throughput is unmeasured
    "quota_refusals_fleet_full": 37      // opens refused because the fleet was full, last 10 min
  }}}
```

Platform (`worker/sandbox-gateways.ts`): demand per group from these fields as above; an order whose
`quota_refusals_fleet_full` stays high while renting is blocked (no inventory, spend cap, hostnames) is marked
`cpu_limited` (visible in the order status), not silently slow.

## Limits to lift (separate work)

- More than 5 gateways: a certificate covering more names (`sbx-eval1..N`) or per-gateway names.
- A second VM provider when Targon has no inventory (observed 2026-11-12: no cpu-large/xlarge).
- Progressive readiness: a gateway serves the tasks whose images it already holds while the rest pull
  (`reliquary-sandbox`: the approved image list becomes dynamic; the capacity report already lists images and
  validators already place sessions by image).
- Billing on actual gateway-hours, capped at the quote.

## Non-goals

No change to the audit sizing or to the session quota's fairness. No change for non-agentic orders.
