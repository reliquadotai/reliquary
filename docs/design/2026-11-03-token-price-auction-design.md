# Token price auction: one price mechanism for RL, corpus and orders

Status: v1 coded for corpus and order draw jobs (§11), off unless a task declares `pricing: "auction"`. RL is
unchanged.

## 1. Why

Every job on the subnet buys the same thing: verified tokens of a pinned model, under the forced draw. Today we
guess what those tokens should cost.

- **RL** (`reliquary/validator/rl_price.py`): one multiplier per environment, steered from the training queue
  (supply / demand ratio, rise x1.05, decay, steering under shortage). It oscillated, pinned every env at the
  ceiling for hours (2026-10-22), and needed guards on guards.
- **RL leases** (`rl_leases.py`, `rl_lease_quota.py`): a hotkey may hold `min(cap, target)` leases, with `cap`
  following AIMD from 4 and `target` from Little's law over paid deliveries. Inside a ~1 s interval, contested
  prompts go to a random requester with quota left. The quota needed five manual restores in the last run, and the
  2026-10-31 replay found that measuring capacity from paid groups is the remaining blocker.
- **Corpus**: a price ceiling per token set by hand.
- **Orders (batch and eval)**: priced from our own GPU-hour estimates.

We do not know a miner's cost, so we either overpay everyone or get no supply. The forced draw makes miners
interchangeable: for a given prompt and draw there is one valid completion, whoever computes it. The cheapest
miner therefore does not produce worse data, and price can be the only thing that decides who works.

**Goal:** the cheapest GPUs do the work, and we pay the least that covers the demand, with no capacity
measurement, no quota and no hand-tuned price.

## 2. The mechanism

### 2.1 Asks

Each hotkey posts one ask: the price per token at which it works, per environment (RL, corpus) or per pinned model
(orders). No quantity, no capacity. Changing an ask takes effect at the next round; at most one change per round
counts. Asks are private to the validator; the round prices are public.

### 2.2 Leases served cheapest first

Leases keep their current role (exclusive draw slot, nonce, deadline, mandatory submission). Only the order in
which requests are served changes:

- Requests arriving within the collection interval (today ~1 s, `LeaseCollector`) are served in ascending ask
  order while work is open, ties in random order (the current shuffle).
- A hotkey whose ask is above the ceiling is never served.
- There is no per-hotkey quota. A request is served as long as there is open work.

### 2.3 Round price

A round is one unit of the bottleneck: one trainer step in RL, a fixed period (e.g. 10 min) for corpus and orders.

> **P = the highest ask actually served in the round, capped by the ceiling.**

Every token leased in the round is paid P. The miner knows at grant that it earns at least its own ask; P is
published when the round closes. Unspent budget burns.

Why this rule:

- **It needs no knowledge of supply.** P is read from what happened. When miners are scarce, everyone is served
  and P is the ask of the most expensive one, which is the least that keeps all of them working. When miners are
  plentiful, the expensive ones are not served and P falls to the most expensive one still needed.
- **A low ask buys volume.** The cheapest are served first, so under competition they do all the work, and they
  keep `P − cost` per token.
- **A high ask loses volume.** The marginal miner can raise P by raising its ask, but it is then served last, and
  it loses everything as soon as cheaper supply is enough. The more miners, the riskier inflating gets.
- **Uniform beats pay-as-bid for our goal.** Under pay-as-bid, miners learn P and bid just under it, so we end up
  paying about P anyway (revenue equivalence), and part of the competition shifts to who guesses P best rather than
  who runs cheapest. Under the uniform price, the only way to win more is a lower cost. To be confirmed by the
  simulation (§7).

### 2.4 Ceiling and budget

> **ceiling per token = budget per round ÷ tokens the bottleneck absorbs per round**

- RL and corpus: the budget is the task's emission share for the round. The period settlement (`rl_ledger`:
  normalised to the task cap, rest burned) stays as the last guard and should never bind.
- Orders: each order brings its budget, `client price × (1 − margin)` per token, so its ceiling is per order.

A faster bottleneck spreads the same budget over more tokens: the ceiling per token falls, and buying more tokens
moves P up the supply curve. The total never exceeds the budget. The useful speed is the one where P reaches the
ceiling and nothing burns; beyond it we wait for miners.

### 2.5 Expiry costs the lease's price

A lease not submitted by its deadline earns nothing and costs the hotkey the lease's value (tokens × P), deducted
from its next accruals. Holding leases one cannot serve becomes a loss, which replaces the quota. Our own failures
stay neutral, as today: sandbox releases, validator downtime and our fault restores (`ops_quota_restores` causes).

A new hotkey has no accruals to deduct from: its debt carries to future pay, and registration cost bounds how
often a hotkey can be abandoned to escape it. Open question: whether a small number of leases before the first
delivery is still needed (§8).

### 2.6 Worked example (RL, illustrative numbers)

One round = one trainer step of 20 M tokens. Budget 10 $ per round, so the ceiling is 0.50 $/M.

| Miner | Ask | Produces per round |
|---|---|---|
| A | 0.15 | 8 M |
| B | 0.20 | 6 M |
| C | 0.30 | 5 M |
| D | 0.45 | 4 M |
| E | 0.70 | 10 M |

- **Round 1.** E is above the ceiling. A, B and C take 19 M and D takes 1 M, so P = 0.45. The round costs 9 $, and
  1 $ burns.
- **Round 2.** F arrives at 0.25 with 3 M. A, B and F take 17 M and C takes 3 M; D is not served. P = 0.30, the
  round costs 6 $, and D must go under 0.30 or leave.
- **Round 3.** A is down, and B, F, C and D make 18 M. Everyone is served and P = 0.45. 2 M stay open: this is the
  shortage signal.

## 3. How much work is opened

The bottleneck sets the work, measured rather than declared:

- **RL:** the trainer (unchanged: the trainer takes `batch_groups` per step, leases are only on unrevealed `S_k`).
- **Corpus and orders:** the audit card. Each round opens leases at the rate the card actually digested over the
  last rounds: the audit queue grows, so fewer leases open; it is empty, so more open. This is flow control, not a
  capacity estimate.
- **Shares:** jobs that share a card (e.g. a client order and a corpus job on the same model) get a share of its
  measured throughput, as environments get a share of the trainer.
- **A second card** for a model is rented only when the queue stays full for several rounds after the leases were
  cut, and is released when it drains. It is a volume decision only.

## 4. Verification

The forced draw checks every audited token. We start simple and loosen with measurements.

1. **Start at 100 % audit** for corpus and orders (`audit_q = 1`, already a job parameter; evals already run at
   1). It removes probation, held pay and retroactive voids from the critical path, verifies logprobs on every
   sample when we add them, and measures two unknowns: the honest failure rate, and the audit card's tokens/s
   (draw included) on Qwen3.8-27B. A rough estimate is ~5k tokens/s for one H100, i.e. ~400 M tokens/day; this
   must be measured.
2. **Then 15 %**, when volume exceeds a card. Two conditions:
   - A confirmed failure voids **all** of the hotkey's submissions in the job, audited or not, and they are
     regenerated by others. The void path exists (`write_voided`). Pay is already held for `hold_seconds` (72 min);
     it must cover the void.
   - Ban at the first confirmed failure if the honest failure rate measured in step 1 is ~0 (today: suspect for 24
     h at 100 % audit, ban after 3 failures in 7 days).

   Cheating stays a loss while what a caught miner loses exceeds about `(1 − q)/q ≈ 5.7` times what it saves per
   cheated sample. With the job's work voided and a ban, it is orders of magnitude more. Twenty cheated samples go
   unnoticed with probability 0.85^20 ≈ 4 %.
3. **Logprobs:** top-k logprobs come from the same forward pass as the tokens, so faking them saves nothing. The
   only saving is a cheaper model, which the draw catches. Tolerance on logprobs must be set: fp8 miners are allowed
   and their logprobs differ slightly from bf16. Orders with logprobs may require full precision.

## 5. Supply

- **Shortage signal:** P at the ceiling and work still open for several rounds.
- **Responses:** RL lowers its target speed or raises the budget share; an order extends its window or quotes a
  higher price. For a model nobody serves, the first order also pays for miners to load it, so it needs a higher
  ceiling or a longer window.
- **Avoid the cobweb:** miners renting GPUs take hours to arrive or leave. A ceiling that moves fast makes supply
  swing (high price → influx → crash → exodus → shortage). The ceiling changes slowly, and the history of P is
  public, so miners decide on a trend.

## 6. Risks

| Risk | Answer |
|---|---|
| The marginal miner inflates its ask to raise P | It is served last and loses its volume as soon as cheaper supply covers demand. Measured in §7. |
| One operator with many hotkeys sets P with one and works with the others | It must leave work to its own expensive hotkey; costly when others compete. Measured in §7 (concentration). |
| Lease hoarding | Expiry costs the lease's value. |
| Cheap but slow miners | Deadline per lease (fixed per env, as today). The expiry cost makes a slow miner raise its ask or leave. |
| Ask flapping to probe P | One ask change per round; asks private. |
| Escaping debt with a new hotkey | Registration cost; debt carried. Open question on a first-delivery allowance. |
| Collusion with few operators | Ceiling bounds the damage. Visible as P at the ceiling with work served, not open. |

## 7. Simulation

Replay of the RL run's lease history (controller DB, 2026-10-22 → 10-30, 116 trainer steps) with miner costs
assumed from GPU rental prices and measured throughput. Miner behaviour is assumed, so the dollar levels are not
predictions; the comparisons between rules are what holds.

**Measured** (no assumption):
- 94.6 M trained tokens per step on average;
- per-env prices swung by 10x or more under the current controller;
- 46 % of the cap went unpaid;
- a median of 36 active hotkeys per step, top-1 share 5.9 %;
- envscaler is thin: 21 hotkeys, top-1 share 38 %.

**Compared** (assumed costs):
- **Pay-as-bid** looks cheaper only against naive bidders. Bidders that learn from past prices make it cost 1.4x
  the uniform price, and 1.8x if bidding starts at the ceiling.
- **The uniform price** barely moves when miners bid strategically (+3 %).
- **Ceiling:** start at 1.5-2x the historical price per environment, then tighten toward the observed clearing
  price. If costs are high, a 1x ceiling starves the trainer (80 % served); if they are low, the extra simply
  burns.
- **Main risk: a few large miners set P when supply is tight.**
  - The top hotkey alone adds 6 % to the cost at today's trainer speed, and 46 % at twice the speed.
  - The top-10 merged add 97 %.
  - Thin environments and short rounds are the most exposed.
- **Concentration** rises slightly: top-1 goes from 7 % to 10-12 %, and 70-77 hotkeys are still served.

Decision: uniform price. The open follow-up is a guard against tight-supply manipulation (smoothing P over rounds,
or a share cap per operator), to measure in production before choosing.

## 8. Code map (when decided)

- `rl_leases.LeaseBook.grant`: sort claims by ask instead of shuffling; drop the quota cut.
- `rl_lease_quota`: removed, or kept only as a first-delivery allowance (open).
- `rl_price`: replaced by the round clearing (served asks → P), one per env.
- `rl_ledger`: the price at ingestion becomes the lease's round P; expiry debits; the period cap kept as a guard.
- Corpus and order light leases (`/corpus/lease`): the same grant order and round price, per job.
- A new route for asks (signed by the hotkey, one per env or model), and a public round history.
- Orders: the quote is priced from recent P for the model plus margin, and the ceiling is the order's budget.

## 9. Rollout

1. **Shadow mode, one week:** miners post asks, rounds clear and P is published, while pay stays on the current
   rules. Compare.
2. **Corpus and orders first,** at 100 % audit.
3. **RL** at the next run.

## 10. Open questions

1. Round length for corpus and orders (10 min proposed).
2. A first-delivery allowance for new hotkeys, or debt only.
3. Ask granularity for RL: per env (proposed), or one ask with per-env multipliers.
4. Whether pay-as-bid is ever better for us (§7).
5. When to move from 100 % to 15 % audit: volume threshold and honest failure rate.

## 11. Implementation (v1)

Scope: draw jobs served by the corpus validator and the order control (corpus and batch/eval orders). Off unless the
task's registry params declare it:

```
reliquary tasks set-cap --task-id <task> --max-price-alpha-per-mtoken <ceiling> --pricing auction \
    [--auction-round-seconds 600] [--auction-initial-leases 64]
```

A validator applies it when it wires the job (restart or hot-add); changing it on a live job needs a rewire.

What is coded:

- **`reliquary/corpus/token_auction.py`**: asks, rounds, price moves, leases per round, debts, public view.
  - The round price is fixed when the round opens, at the highest live ask under the target, so a lease's price
    is known when it is granted.
  - Leases go to admitted hotkeys (ask ≤ round price) until the round's leases run out. This is the threshold
    form of "cheapest first": within a round every admitted miner is paid the same, and at each round close the
    target moves one ask down (saturated) or up (leases left while a miner was priced out).
- **Leases** (`corpus/draw_leases.py`):
  - no per-hotkey cap on an auction job;
  - every lease carries its price and round;
  - an expired lease charges its holder in full, a released lease by the share of its time held.
- **Routes** (`validator/corpus_service.py`):
  - `POST /corpus/ask` and `/corpus/jobs/{id}/ask`: signed and single-use, in micro-alpha per million tokens.
  - `GET /corpus/jobs/{id}/auction`: the round, price, ceiling, leases and closed rounds, never an ask or a
    hotkey.
  - Lease answers carry `detail.price_alpha_per_mtoken`.
  - A submission without the hotkey's own lease is refused `lease_required`.
- **Records and verdicts:** an accepted record carries `lease_price` (pool share per token) and `lease_debit`. The
  auditor copies them to the verdict.
- **Settlement** (`corpus_settlement.rewards_for`): a period pays `tokens x lease_price − lease_debit` per hotkey,
  scaled down only past the period's budget (`min(cap, ceiling x tokens)`); the rest burns.
- **Audit queue:** the leases per round follow the audit backlog.
  - The corpus auditor exposes `audit_backlog()`.
  - The order control adds the samples waiting at its executors (`PairedAuditDispatcher.pending_items`).
- **Miner:** `reliquary corpus mine --ask-alpha-per-mtoken X` posts the ask before leasing, every 10 min, and
  again when the validator says it holds none.

Limits of v1:

- **State is in memory:** asks, debts and the target price. A restart drops debts and restarts the price at the
  ceiling for one round. Miners repost asks within 10 min.
- **Debt leaves with the job:** it is taken back from later submissions of the same job only.
- **Agentic miners are not covered:** `corpus mine-agentic` does not post asks yet, so episode jobs should not be
  priced by auction.
- **Split corpus validators are not covered:** a front that serves routes apart from the judge does not wire the
  auction.
- **Audit rate:** start auction jobs at `audit_q = 1` (§4).

## 12. RL (later)

v1 is not wired into the RL controller. Batch and eval orders come first. The auction keeps an RL run fully
async: miners generate continuously, the trainer never waits on a round, and a lease's price is known when it is
granted. The round is only the cadence of price changes, never a synchronisation point.

Three changes are needed before an RL job is priced by auction:

- **Leases per round follow the trainer, not only the audit queue.** In RL the trainer is the bottleneck. The
  signal is the groups waiting for the trainer, plus the audit backlog.
  - Too many leases: groups pile up, fall behind the checkpoint and honest miners fail the lag audit.
  - Too few: the trainer waits.
  - So the lease count is a lag regulator. Target a queue of about `groups per step x tolerated lag`. At
    equilibrium the miners' rate equals what the trainer consumes.
- **No debt when we void the lease.** A lease made stale by a checkpoint swap is not the miner's fault. It ends
  without debt, like our outages are neutral in RL pay today.
- **Short RL leases.** An RL lease lives under the maximum tolerated lag, not 72 min. The 10 min round can stay:
  it is a little longer than a trainer step (~350-420 s), enough for a stable price.

At equilibrium the cheapest miners keep their GPUs full and the most expensive admitted miner fills the gap. The
price settles where supply just keeps up with the trainer. This replaces the RL price rise on trainer wait and
the pay smoothing.
