# Sandbox capacity follows what miners want (agentic orders)

Date: 2026-11-13. Status: design and implementation, branch `feat/sandbox-sizing-miner-demand` (Catalyst
and platform). Builds on `2026-11-12-sandbox-capacity-autoscale.md` and composes with the verification tiers
(`2026-11-13-verification-tiers-precision.md`, PR #520: sampled audit by default, `audit_q` 0.05).

## Problem (live evals, 2026-11-13)

A miner never asks for more sandbox sessions than the `allowance` the control names: the signed runner grows to
`min(--max-live-per-job, allowance)`. The allowance comes from the fleet's size `C` (39 sessions per gateway)
through the corpus quota (`reliquary/sandbox/corpus_quota.py`). The platform's autoscaler sizes the fleet from
`max(deadline_rate, audit_rate) x episode_seconds x 1.3`, and its CPU-limited flag waits for fleet-full refusals.
Since miners limit themselves to the allowance, their real demand never shows: one gateway (39 sessions) for a
100 x 4 SWE order, and a 9B miner whose GPU could run 64-128 episodes gets about 10. The GPU idles and agentic
orders are unprofitable to mine.

Reacting to refusals is also too slow for CPU: a gateway boots in 15-30 min (swe-full about 2 h), while it costs
about 0.30 $/h, far less than the miners' GPUs it keeps busy.

## Rule

Per agentic job, the concurrent sessions to provide:

```text
sessions_target = min(demand, episodes_left, audit_capacity)        binding = the term that is least
```

- `demand`: what miners want on the job (below).
- `episodes_left`: problems x samples less what is accepted (`episode_counts`).
- `audit_capacity`: the sessions whose episodes the audit can keep up with at the most executors the order may
  have:

  ```text
  audit_capacity = tokens_per_second(one executor, completion + prompt) x executors_allowed
                   / (tokens_per_episode x audit_share) x episode_seconds
  ```

  `executors_allowed` is the control's `min(RELIQUARY_ORDER_MAX_AUDIT_EXECUTORS, order's declared executors)`;
  `audit_share` is the auditor's measured share of records it audits (`CorpusAuditor.audit_share`: at least the
  job's `audit_q`, probation and slow-hotkey audits included). At the default 5 % share it rarely binds; at 100 %
  it does. Until it is measured (no throughput, no audited episode yet) a conservative prior stands in:
  `AUDIT_PRIOR_SESSIONS_PER_EXECUTOR` (64) x `executors_allowed`, binding `audit_prior` (never `audit_limited`).

When `binding = audit`, executors already grow to `executors_allowed` with the audit backlog
(`executors_for_backlog`); beyond that, the order's budget is the limit, and the order shows `audit_limited`.
With audit waves (#520) the capacity is the average rate over the order: waves drain the same backlog later.

## Demand: miners declare it

The least invasive carrier is the session open the miner already signs: the corpus engagement gains an optional
`want` (1-512, `MAX_SESSION_WANT`), the live sessions this hotkey wants on this job at once. It is part of the
engagement the hotkey signs (`build_sandbox_open_binding` hashes the engagement), so nobody declares for another
hotkey, and every open refreshes it (a miner below its allowance opens continuously; one at its limit opens each
time an episode ends).

- Compatibility: a validator that reads it says so (`declares_want: true`) on every grant and quota refusal. The
  miner sends `want` only after seeing that, because an older validator refuses unknown fields (`extra=forbid`).
  A lost open is resent with the same want (same request id, same engagement digest). An older miner sends none:
  today's behaviour.
- Miner: `want = min(--max-live-per-job, ceil(--concurrency / identities))` (its batch; `declared_want`).
- Validator: the issuer hands the want to the quota (`CorpusQuota.declare`) once the job is known to be served to
  the hotkey (not for `miner_banned`, `job_not_served`, `job_not_signed`, `prompt_mismatch`, `job_complete`),
  whatever the slot or quota answer. The quota keeps the last want per (job, hotkey) for 6 h
  (`FORECAST_WINDOW_S`).
- A declaration is never an entitlement: it changes neither the allocation nor the allowance (fair shares,
  newcomer pool and low-yield cap stay exactly as they are; probation is the audit's). It only makes the demand
  visible, so `C` grows; the allowance, which is already the hotkey's potential share of the whole fleet, then grows
  with it, newcomers included when there is spare capacity.

Per job (`order_sandbox_status.job_demand`), a want counts only as far as the hotkey showed it runs sessions:

```text
shown     = max(held now, sessions held in the last 10 min, episodes finished in the last 30 min)   (on the job)
counted   = max(held, min(want, max(4, 2 x shown)))      shown > 0
          = 1                                             shown = 0, declared in the last 10 min (its first open)
          = 0                                             shown = 0 after that (a want that never ran a session)
declared  = sum of counted over hotkeys that declared on the job within 30 min (or still hold a session there)
undeclared= sessions held on the job by hotkeys that never declared (older miners)
forecast  = sum over hotkeys idle now (no fresh declaration and no session on another served job)
            of min(want, max(4, 2 x peak)) for their latest declaration on a retired job of the same model, last
            6 h, where peak is the most sessions the hotkey held on that job when declaring (peak 0: nothing)
demand    = declared + undeclared + forecast      (None while declared and forecast are both 0)
```

A miner's count doubles as its sessions do (it holds what the fleet allows, the fleet grows to twice that, ...),
which the platform's doubling scale-up follows; one word never rents more than twice what was shown.
Low-yield hotkeys count nothing. A hotkey is counted once: its own declaration on the job replaces its forecast.
`None` keeps the platform on its earlier rule, so a fleet of only older miners is sized as today.

The forecast makes sizing predictive: a new order is sized from the miners that mined the previous order of its
model, before its first open. `/miner/setup` declarations are not used: the order control does not serve that
route (only the RL service does) and they name neither the model nor the env; recent activity on the model is the
better signal, and the env does not matter for signed sandboxes (tools run on the gateways, not the miner).

### Abuse bounds

A hotkey (or many) may declare 512. It counts one session for 10 minutes, then nothing, unless it actually runs
sessions; once it does, at most twice what it ran. So N registered hotkeys that never run anything add at most N
sessions for 10 minutes. Before the audit is measured, the prior caps the job at 64 sessions per allowed executor.
Beyond that: the job's episodes left, the order's paid sessions (its quote's gateway-hours), the daily spend cap,
`GATEWAY_MAX` and the hostnames; the surplus is released after the scale-down hold. Low-yield hotkeys are excluded;
a hotkey must open (signed, served job) to declare. The allowance is unchanged, so a declaration buys no sessions.

A validator rolled back to a version that does not read the field refuses an open carrying it
(`malformed_request`): the miner then stops declaring and sends that open again without it (a later
`declares_want` turns it back on).

## Status fields

`GET /admin/v1/eval-control/status`, `jobs[id].sandbox` gains:

```json
"demand_sessions": 192, "demand_declared_sessions": 128, "demand_forecast_sessions": 64, "demand_hotkeys": 3,
"audit_share": 0.05, "executors_allowed": 2, "audit_capacity_sessions": 2352, "audit_capacity_measured": true,
"sessions_target": 192, "binding": "demand"          // binding: demand | episodes | audit | audit_prior
```

## Platform: predictive and fast

- Sizing: a job with a `sessions_target` takes `min(sessions_target, episodes_left, paid sessions)` (rule
  `demand`, no margin: it is a count of sessions); one without keeps the deadline/audit-rate rule.
- At order start the control already reports the forecast, so the group rents for it without waiting for a
  refusal.
- Scale up fast: a group short of gateways may double its serving gateways per cron run (1, 2, 4, 8 ...), each rent
  paying its projection against the daily cap. Scale down slow: unchanged (`GATEWAY_SCALE_DOWN_SECONDS` hold, one
  per run, drained before destroyed).
- Warm pool (`GATEWAY_WARM_POOL`, e.g. `swe=1,tmax=1,deepswe=1`, off by default): a group with no running order
  keeps that many pre-booted gateways (images pulled), adopted by the next order at once, while the day's spend
  leaves an hour of them.
- Progressive readiness (platform #201) already lets a new gateway serve the set's first images while the rest pull,
  so the doubled gateways take sessions early.
- `audit_limited` on the order (like `cpu_limited`) while a job's binding term is the measured audit (`audit`, not
  `audit_prior`).

## Cost

Gateway: about 0.30 $/h on Targon (the platform projects with `GATEWAY_HOURLY_PRICE_USD`, 0.60 by default).

| case | extra CPU | what it buys |
| --- | --- | --- |
| renting 4 gateways instead of 1 for the first hour of a SWE order | 3 x 0.30 = 0.90 $ (1.80 $ at 0.60) | 156 sessions ready one boot after the start instead of 39 |
| a forecast nobody uses: 5 gateways, boot (0.5 h) + scale-down hold (10 min) + drain | about 5 x 0.30 x 0.75 = 1.1 $ | the bound on a wrong forecast |
| warm pool of 1 gateway, 24 h | 7.2 $/day per group (swe + tmax + deepswe: 21.6 $/day, 11 % of a 200 $/day cap) | an order starts on a ready gateway, no 15-30 min boot (2 h for swe-full) |

Against it: an H100 costs a miner about 2-2.5 $/h. A 9B miner held to 10 sessions when its batch is 64 idles
most of its GPU, about 1.7 $/h of waste per miner; three such miners waste about 5 $/h, several times the CPU the
extra gateways cost. Under-provisioning CPU is the expensive mistake, so the fleet errs high and releases slowly.

## Rollout and releases

1. Catalyst control (order control, corpus validator): accepts `engagement.want`, says `declares_want`, publishes the
   status fields. Safe alone: no miner declares yet, the platform ignores the new fields.
2. Platform: reads them (falls back to its earlier rule while `sessions_target` is null); migration 0069
   (`evaluation_orders.audit_limited_since`, numbered after #210's 0067-0068); `GATEWAY_WARM_POOL` left unset.
3. Miner release (needed for the demand to show): `reliquary/miner/signed_episode.py` (declares `want`),
   `reliquary/miner/agentic_miner.py` (`declared_want`), `reliquary/protocol/sandbox_session.py` (the field),
   `corpus mine-agentic --concurrency` help. Miners should set `--concurrency` to the episodes their GPU runs at
   once (and `--max-live-per-job` at least that). Older miners keep working.

## Not changed

The quota's fairness, the newcomer and low-yield pools, the allowance rule, the audit sizing per model, non-agentic
orders. Raising `executors_allowed` beyond what the order bought is not automatic: an `audit_limited` order shows it.
