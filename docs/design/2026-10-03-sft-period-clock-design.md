# Corpus tasks paid on their own clock

2026-10-03. Scope: tasks of mechanism `corpus-generation` (SFT corpus jobs and
evaluation jobs) declared after this ships. RL (`rl-discovered-price`) and the
corpus tasks already running keep the code path they have today, byte for byte.

## Problem

A corpus task borrows RL's clock, and its pay neither conserves nor stops.

- The settler dates each payment by the RL window index it last saw
  (`choose_window(other_max=…)`), and advances on its own only when RL stalls, at
  RL's cadence (`RL_WINDOW_SECONDS`).
- The weight setter replays each task's archives over one shared horizon: the
  highest window of **any** task, 216 windows back (`weight_only.submit_once`).
- Its EMA steps only on the task's **own** archives (`_replay_ema`): a window in
  which the task produced nothing does not decay it.

Consequences, measured on prod on 2026-10-03, where three jobs are full
(`code`, `if`, `logic`, caps 0.04 + 0.02 + 0.04, still `active`):

1. **A finished task keeps paying at almost its full rate** until its last archives
   leave the 216-window horizon: about 180 windows of pay, where the EMA owes about
   1/α ≈ 36 (α = 2/73) to return its start-up lag. About 140 windows, roughly 1.5 days
   at 16 min a window, are paid for no work, mostly to whoever mined last.
2. **The length of that tail depends on RL.** 2.4 days at 16-minute windows, 4.5 at
   30, and with RL stopped the horizon never moves, so the tail never ends.
3. **Smoothing has no fixed meaning in time.** 72 windows is 19 h or 36 h depending
   on RL's speed.
4. **A token is dated by its verdict, not its work.** An audit backlog lands two
   periods' work in one settlement. That settlement pays one cap for both, and the
   empty period's cap burns.
5. **Budget is never freed.** `total_cap` counts retired entries, and the sum is
   already 1.0, so no new task (an evaluation at 0.02 included) can be declared.

## Goals

- **Conservation.** Each hotkey is paid, in total, exactly the share of cap its
  verified tokens earned in each period of work. No bonus after the work, no loss
  at the start.
- **A finished task stops by itself**, within a fixed time in hours, whatever RL
  does.
- **Robust to a skipped or late weight-set**: the error is a fraction of one
  period's difference, not a whole period paid twice or never.
- **RL is unchanged**, and so is every corpus task declared before this ships: same
  archives, same EMA, same horizon.

## Non-goals

- Per-token pricing (a job budget at a fixed price per token). Each period's cap is
  still split by the tokens of that period.
- A ledger of what the chain actually paid (debts). The EMA's conservation assumes
  one weight-set per period, and the error when that fails is bounded below.

## Design

### 1. The clock: drand periods of 72 minutes

`period(t) = floor((t − genesis) / 4320)`, from the drand chain info the auditor
already uses (`drand.get_current_chain()`: `genesis_time`, `period`). 4320 s is 72
minutes, one Bittensor epoch, the rate at which weights are set. The clock always
advances, whatever any task does.

### 2. A token belongs to the period it was submitted in

Each submission record already carries `received_at` (the validator's receipt
time). The auditor writes it into the verdict (`received_at`, added beside
`audited_at`). The settler dates a verdict by `period(received_at)`. A verdict
written before this change has no `received_at` and is dated by `audited_at`.

### 3. A period is closed when nothing received in it is still undecided

Period `p` is **closed** once:
- the auditor has no pending submission (accepted, no verdict yet) with
  `received_at` in `p`, and
- `now ≥ end(p) + admission slack`, so a submission received at the very end of
  `p` has reached the store.

The auditor already holds every pending submission's `received_at` (`_meta`). It
exposes `oldest_pending_received_at()`; every period ending before
`min(that, now − slack)` is closed. With sampled audits (`audit_q < 1`), an
unaudited pass is decided at the end of its hold (`audit_hold_seconds`, 4320 s by
default), so a period typically closes one or two periods after it ends. A
validator restart re-seeds the pending set from the store before closing anything.

### 4. Settlement: one archive per closed period of work

For each closed period `p` with verdicts:
`rewards_p = cap × tokens_hk / tokens_total`, over the passed, non-voided verdicts
received in `p` (`rewards_for`, unchanged). A period with no passed token writes
nothing, and its cap burns.

The archive is written once, write-once, under a new prefix that RL listings never
see:

```
reliquary/corpus-periods/<task_id>/<work period p>.json.gz
{"schema": "reliquary/corpus-period/v1", "task_id", "job_id",
 "work_period": p, "entry_period": e, "rewards_by_hotkey", "tokens", "verdicts"}
```

`entry_period` `e` is the period after the one in which `p` was settled, the period
from which its pay starts: written during period `c`, the archive may land after
`c`'s weight-set, so it enters at `c + 1` and no share of it is ever skipped. The settlement state (per job, CAS-written) keeps the last closed
period and the pending archive, so a crash delays a payment and never repeats it
(the two-phase write is unchanged). A `period-ema-v1` job's settler never reads `other_max`: the stall rules and
`RL_WINDOW_SECONDS` stay only for jobs settled the old way.

### 5. Replay: an EMA that decays every period, from the period the pay entered

For each declared task whose entry carries `params.settlement = "period-ema-v1"`,
at weight-set time:

```
P = period(now)
ema_hk = Σ over archives a of the task with e_a ≤ P:
         α · (1 − α)^(P − e_a) · rewards_a[hk]
```

- α = 2 / (N + 1) with **N = 6**, so α = 2/7 ≈ 0.286: half-life ≈ 2.1 periods
  (2.5 h), mean delay ≈ 2.5 periods (3 h), 99 % of a period's pay delivered within
  14 periods (17 h).
- Archives with `P − e_a > K` are not read. K = 24 periods (29 h) leaves
  (1 − α)^24 ≈ 0.03 % unpaid, the only loss.
- Indexing by `entry_period`, not work period, is what makes it conserve: a period
  closed late enters late and is paid in full from there. Indexed by work period, a
  period closed L periods late would lose a fraction 1 − (1 − α)^L of its pay.
- Several periods closing in the same entry period add up. The per-task clamp to
  `cap` (`_clamp_to_cap`) stays: it only binds on a catch-up of more than about
  1/α periods at once, and is logged.
- The floor (`min_incentive_share`) is applied to the result per task, as today.

Every other task replays exactly as today. A `period-ema-v1` task writes no window
archive, so the window listings, the shared horizon and the RL EMA see nothing new.
The weight vector is the sum of both replays; the global backstop clamp is
unchanged.

### 6. Why the error stays small

- **Weight-set skipped or late.** The previous weights stay one more epoch. The
  error is `ema_P − ema_{P−1} = α (r_P − ema_{P−1})`, about 29 % of the change
  between two periods, not a whole period.
- **Drift between drand periods and chain epochs.** An epoch that falls twice in
  one period, or skips one, is the same case as above.
- **A hotkey that deregisters during its tail** loses the rest of it, as with any
  scheme that pays after the work.

### 7. Budget

A cap of 0 pays nothing new and counts for nothing in `total_cap`, so freeing a
finished task's share needs no new registry status (one an older binary could not
read). See §9 for how a task is closed.

### 8. Transition: a job keeps the rule it was declared under

`jobs create` writes `settlement: "period-ema-v1"` into the task entry's params for
every new corpus or evaluation job (`--settlement windows` keeps the old way).
Order jobs declared by the admin service keep the old settlement for now. An entry without it is settled and replayed by
today's code, unchanged. Nothing is converted:

- Converting a running job's state is not exact. The old rule owes its start-up lag
  in 16-minute windows (about 36 of them, about 8 periods' worth of cap). Entered as
  one archive, that would hit the per-task cap clamp. Mixing both rules on one job
  makes the clamp scale new work down to pay the old tail.
- **The running math job** (`corpus-math-omi-v1`) finishes under the old rule.
  When it is full and drained, `tasks close --cut-tail` ends its frozen tail and
  frees its share.
- **The three full jobs** (`code`, `if`, `logic`): `tasks close --cut-tail` ends
  their tail and frees 0.10.

**Deploy order.** The corpus validator and every weight setter run this binary
before the first `period-ema-v1` job is declared. Declared earlier, an older
validator settles it by window and an older weight setter ignores the field: it is
paid the old way, not lost.

The settler picks its mode from the entry's params, so one validator process can
serve old and new jobs side by side.

Evaluation jobs are corpus tasks: once this ships they are declared
`period-ema-v1` like any new job.

**2026-10-08: the window settlement is gone for corpus tasks.** Every corpus
entry is declared `period-ema-v1` (`jobs create` and the admin service alike;
`--settlement windows` is refused). The corpus validator, its split judges and
the order control settle only period tasks; a window-settled one is refused
when wired hot and left out with an ERROR at startup (never a crash, so a stale
`RELIQUARY_TASK_ID` does not stop the others). The weight setter no longer
replays a corpus task's window archives, which stay in R2; their windows still
count in the shared horizon, so RL's replay is byte-identical. Every corpus
window task left in prod was at cap 0 or retired when this shipped.

### 9. Closing a task

**The cap governs only the periods still to be worked.** Until 2026-10-08 the
weight setter held every archive, and the task, to the task's *current* cap, so
setting the cap to 0 when a job finished cut the tail it had earned (measured:
`corpus-science-v1`, cap 0 eleven periods after its last entry, lost about
0.01 pool·period, 0.5 % of what it earned). Now:

- each archive records the cap it was settled under (`cap`);
- the weight setter pays it up to that cap, never past the task's pay ceiling
  `max(cap, tail_cap)`, where `tail_cap` is the highest cap the task had,
  written by `set_cap` whenever a period task's cap is lowered. An archive
  written before `cap` was recorded is held to the ceiling;
- the per-task bound is `CATCHUP_ENTRIES` × the ceiling (was × the cap);
- a period settled at cap 0 writes no archive: nothing new is ever paid.

So a lowered cap lets the earned tail run out in full (to the loss
`(1 − α)^(K+1)`), and no period pays more than the cap it was settled under. A
cap lowered by a binary older than this writes no `tail_cap`, and its tail is
cut as before.

**A finished job closes itself.** The corpus validator's job set looks, at most
every 10 minutes, for a served job that is full (every prompt's slots taken, or
every eval prompt complete) and drained (no write being admitted, nothing
ungraded or held by its grader, every submission audited, every verdict
settled, so no period open and the last archive written), with its task active,
period-settled, paying and not paused. It then writes cap 0 through
`set_task_cap` (compare-and-swap, guarded on the entry still being that job's,
active and paying), once, logged at INFO with the job's counts. A cap already 0
is skipped, so a restart writes nothing again. `RELIQUARY_CORPUS_AUTOCLOSE=0`
turns it off.

The task is **not retired** automatically: a retired task named in a corpus
validator's `RELIQUARY_TASK_ID` stops that validator from starting (and
rollback containers on older images would loop on it), while a retirement
frees nothing a 0 cap has not already freed.

**`reliquary tasks close --task-id X`** sets the cap to 0 (if it is not) and
retires the task, refusing unless its job is drained. It no longer waits for
the tail to decay, since the cap no longer cuts it; it prints the tail still
being paid. `--cut-tail` is accepted and ignored.

**Budget while a tail pays.** A 0 cap frees the share in `total_cap` at once,
while the tail is still paid from it for up to K periods (about 29 h, 99 %
within 17 h). A share reassigned meanwhile can make the sum paid exceed the
pool; the weight setter's global clamp then scales every miner down for that
time. Reassign once the tail has run out.

## Parameters

| Name | Value | Where |
|---|---|---|
| period | 4320 s (72 min), drand time | protocol constant |
| N, α | 6, 2/7 | protocol constant |
| replay depth K | 24 periods | protocol constant |
| admission slack | the auditor's accept slack (420 s) | existing |
| autoclose pass | at most every 600 s | `corpus_autoclose` |

## Tests

- Conservation: random reward streams (gaps, bursts, late closes, a job that ends);
  over a long enough replay, every hotkey's paid total equals its earned total to
  within (1 − α)^K.
- A finished task's weight falls below 1 % of its cap within 14 periods, with RL
  stalled and with RL running.
- Dating: a backlog that lands two periods' verdicts in one settlement pays each
  period its own cap; an old verdict without `received_at` is dated by `audited_at`.
- Closing: a period with one pending submission never closes; a restart re-seeds
  the pending set before closing anything.
- Unchanged paths: byte-identical weights on recorded prod archives (RL and the
  current corpus tasks) when no `period-ema-v1` task exists; period archives never
  appear in window listings or move the horizon.
- Mode: an entry without `settlement` is settled by today's settler; `jobs create`
  writes `period-ema-v1`.
- `tasks close`: refused while undrained; a real corpus entry takes cap 0 (its
  `tail_cap` kept) then retires; a 0 cap frees its share in `total_cap`.
- Tail (§9): a cap set to 0 while archives still queue pays exactly what they
  earned, nothing more; a period settled at cap 0 writes nothing.
- Autoclose (§9): full but undrained, paused, retired or already at cap 0: no
  write; finished: one write, none after a restart.

## Rulings from the branch review

- **One entry period per archive, strictly increasing** (`last_entry` in the
  settlement state). An entry period never carries more than one period's cap, so a
  catch-up after an audit backlog is paid in full, later, instead of being clamped
  at the task's cap (a 12-period backlog entering at once lost ~24 %). It also makes
  every archive key unique: a late verdict of a settled period never overwrites the
  archive that paid it. Archive writes refuse another document under an existing key.
- **An archive not written yet when its entry period is under way enters again**,
  at the next period, before it is written: the weight-sets that ran meanwhile
  never saw it. An archive already written before a crash is not written again.
- **Closing bounds** (the auditor's `oldest_pending_received_at`): an id admitted
  live and not read yet counts from its enqueue time less the accept slack (the
  route stamps the arrival before a write that can take that long); a split judge
  never closes past its arrival feed's coverage less the slack, and closes nothing
  while that coverage is unreadable; an id found by a listing whose arrival stays
  unreadable holds periods open for two periods, then is left out with an error,
  so one corrupt record cannot freeze a task's pay. An undecided submission older
  than six periods is reported.
- **A late verdict** (one whose period was already settled, which the bounds above
  should prevent) is still paid against the period's whole token count, logged as
  an error: the period pays slightly over its cap rather than the worker nothing.
- **The period origin is a protocol constant** (drand quicknet's genesis,
  1692803367): no weight-set depends on reaching a drand relay.
- **Window and period pay of one task add up** in the weight setter; neither tail
  is dropped if a job's settlement changes under it. A period settler refuses a job
  whose settlement state is a window one.
- **Deploying:** `jobs create` refuses a `period-ema-v1` job without
  `--fleet-knows-period-settlement`, the corpus-generation guard's pattern.
- **A cap change applies to the periods still to be worked** (2026-10-08, §9):
  archives already written pay up to the cap they were settled under. (Before,
  it applied to what was replayed from then on, earned tails included.)
