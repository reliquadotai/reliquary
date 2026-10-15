# RL windows redesign: notes and open decisions

Discussion of 2026-10-13 (user + Claude). Nothing below is decided unless marked
DECIDED. These notes are the starting point for the final design discussion.

## 1. What the production archives show

Source: R2 `reliquary/dataset/window-<N>.json.gz`, windows 47787-48786 (1,000
windows, about 10 days). The run is Qwen3-4B with math, code and logic, fill-closed
v6, `fifo-ingress/v1` selection and `fixed-selected-group/v1` payment.

- **The trainer is the clock.** A window lasts 450 s at the median, of which 351 s
  are picks waiting on the trainer. Every environment has its 112 groups within
  80-130 s.
- **The cycle is longer than the window.** From one window open to the next takes
  777 s at the median, which leaves a 327 s gap of rotation, publication and
  adoption.
- **One environment's supply collapse stalls every environment.**
  - Windows 48078-48200 (~29 h): two math miners left, and the last one supplied
    83 % of math. Admitted groups equalled required groups. Every window doubled,
    from 408 s to 858 s, for about 11 h of excess.
  - 7 windows hit the 1,800 s backstop because code was short.
  - In total, 104 of 999 windows were paced by one late environment.
- **The per-environment price never reacted.** Every environment stayed at the
  0.05 floor for all 10 days. During the math shortage, r ≈ 1.05 fell in the
  `hold` regime: the price rises only when a window times out.
- **Supply is concentrated.** One miner holds 74 % of code, one 70 % of logic, and
  one 48 % of math. There are about 5 real miners.
- **Burned groups are negligible** (about 1 per window per environment).
  Duplicates (`hash_duplicate`) run at about 1 % of math.
- **Pay per token, over the full cycle:** math 0.60, code 0.42 and logic 0.50 α per
  million paid tokens.
  - The fixed pay per group pays code about 30 % less per token than math.
  - Each environment accepts groups during only 8-16 % of the cycle, so a GPU
    doing only RL earns roughly 6-12× less per hour than the per-token rate
    suggests.
- **FIFO selection biases training towards short completions.** Mean rollout
  length by arrival quartile:

  | Environment | Q1 | Q2 | Q3 | Q4 |
  |---|---|---|---|---|
  | math | 322 | 407 | 446 | 450 |
  | code | 532 | 612 | 628 | 644 |
  | logic | 454 | 521 | 534 | 538 |

  The first pick is 20-35 % shorter than the last. Groups arriving after the quota
  fills (the longest ones) are never trained. Both effects act as a hidden length
  penalty.

## 2. Positions taken in the discussion

- **One optimizer step per generated batch** (no PPO minibatches; recent labs do
  this). Expected effects:
  - the run is fully on-policy, so the PPO ratio only carries numeric noise;
  - there are 4× fewer optimizer steps at equal checkpoints, so the learning rate
    goes up: about 3e-6 with warm-up, 4e-6 if stable (√ rule from 2e-6 at 8k
    rollouts).

  Batch size is being worked on in another session.
- **Async (rollouts from older checkpoints) is not for now.** In normal windows the
  trainer, not collection, is the clock. The useful levers are elsewhere:
  - decoupling environments;
  - shrinking the 327 s gap, or letting miners generate during rotation.
- **The env mix should be steered by incentives, against a target share per
  environment**, with supply matched to what the trainer consumes. A fixed
  incentive per environment cannot do that:
  - too high, and we pay for work the trainer cannot use;
  - too low, and the trainer waits.

  The price is the regulator, the quota is the hard cap, and the trainer pace is
  the demand.
- **Decoupling environments (later).**
  - The trainer steps when it has enough tokens in total, after a bounded wait for
    a short environment.
  - The shortfall becomes a debt and that environment's price rises.
  - Per-environment loss weights (Σ w_e·L_e) keep the gradient unbiased, so an
    uneven step adds variance, not bias.
  - Recommended: bounded wait (option A) over never waiting (B).
- **Pay per useful token (later).** Count completion tokens of groups that are
  actually trained: in-zone, EOS-terminated, capped at the maximum length.
  - The August lesson: the throughput tie-break paid for long, reward-0 rollouts.
  - Paying per token removes the miners' interest in short prompts, but not the
    race bias of FIFO selection (section 1).
- **Monopoly on an environment.** A lone miner can push the price up by delivering
  late, then hold it in the deadband.
  - The user's choice: rely on the market (other miners enter), bounded by the cap.
  - To make the market actually react, the reference miner picks environments by
    advertised pay.
  - Rejected for now: a temporary surge that decays; a cap relative to the other
    environments' prices (to revisit with per-token pay if abuse shows up).
- **Deadband 0.8 kept.**
  - Today every price sits at the floor, so the deadband binds nothing.
  - r varies by a factor of 1.1-1.3 from median to p90. Narrowing the 0.8-1.0 band
    would make the price oscillate between the fast rise and the slow descent.
  - Paying less would be a question about the floor, not the deadband.
- **Training time measurement.**
  - Today `training_rounds` = window close − first pick. It includes waits on a
    short environment, so r saturates near 1 exactly when collection is the
    bottleneck.
  - DECIDED for the current change: record per environment how long picks waited
    for its groups while the trainer was ready, and drive the price rise from that.

## 3. Selecting groups without length bias

- **Principle:** if the rule that picks which groups are trained depends, even
  indirectly, on their length, the training data is biased.
  - FIFO favours short completions: they finish first.
  - The old per-group throughput rule, tokens ÷ time since window open, favoured
    long ones. A miner generates in vLLM batches, so a long and a short group of
    the same batch arrive together, and the long one shows 10× the "throughput".
    This was measured in August: late winners carried 4× the tokens of early ones.
- **Prompt selection by miners is the subnet's core and stays.** Assigning prompts
  is ruled out.
- **Proposal: an announcement nonce times each group.**
  1. The miner freely picks a prompt and announces it before generating.
  2. The validator timestamps the announcement and returns a fresh nonce.
  3. The forced-seed uniforms derive from (window randomness, prompt, nonce), so
     the group cannot exist before the announcement.
  4. The group's true duration = hash precommit time − announcement time.
  5. Its rate = tokens ÷ duration, which is length-neutral.
- **Why the proposal resists gaming:**
  - pre-generating is impossible, because the nonce is unknown until the
    announcement;
  - announcing late is impossible for the same reason, and announcing early only
    lowers one's own rate;
  - choosing short prompts gains no rate and, with per-token pay, no money; picking
    in-zone prompts is what pays;
  - big batches lower the per-group rate. This is a trade-off to own: the rule
    rewards per-group responsiveness over node throughput.
- **Residuals to correct in the formula:**
  - a fixed per-group overhead (RTT, prefill, queueing) slightly favours long
    groups: fit duration ≈ a + tokens/b per window and rank on the residual;
  - proximity to the validator is worth milliseconds against seconds of
    generation;
  - the end time is taken at the hash precommit, so payload size does not enter
    the duration.
- **Mandatory condition: no first-come filling.** A long group arrives later in
  absolute time even at an equal rate. Candidates are collected over a period long
  enough for a maximum-length group, then all of them are ranked by rate. This can
  overlap the trainer's current step, which takes 351 s against 80-130 s of
  collection.
- **Concurrent announcements: option A, non-exclusive (preferred).**
  - Two miners may generate the same prompt with different nonces. If both deliver,
    the better rate wins.
  - Nothing can be squatted. That matters because a miner can hold 100 keys, so a
    per-hotkey limit or an abandonment cost does not stop squatting under
    exclusivity.
  - Proximity is irrelevant: ties are decided by generation rate, not by latency.
  - Cost: duplicate work, about 1 % today. It may grow as miners converge on the
    same good prompts, so measure it.
  - Anti-spam: a cap of open announcements per registered UID, plus expiry.
  - Announcements stay private until the window seals, so nobody copies another
    miner's selection.
- **Rejected: option B, exclusive reservation.** Squatting with many keys, and the
  validator's nearest miner wins every race on a popular prompt.
- **Possible add-on: a "taken" hint for reliable miners only.** When a prompt is
  already announced by a miner with a good delivery record, a second announcer is
  told "taken" and usually picks another prompt. Keys without a delivery record
  never trigger the hint, so fresh sybil keys divert nobody. Advisory only, never
  blocking. To enable if duplicates grow.
- **Raising the registration burn** makes sybil keys costlier, but it is also an
  entry barrier for honest miners, which the market remedy above relies on. It is a
  separate lever. Penalising multi-key operators is unreliable: keys cannot be tied
  to an operator with confidence (coldkey, IP and timing are all evadable).

## 4. Open questions for the final design

1. The cap of open announcements per UID, the expiry, and whether an abandonment
   rate matters at all under option A.
2. How long the collection period lasts, and how it overlaps the trainer step.
3. The ranking formula: overhead correction, per-group rate versus node
   throughput.
4. Pay per token: which tokens count, pay for out-of-zone and exploration groups,
   and how pay interacts with the rate ranking.
5. How per-environment quotas, the price controller and the decoupled step fit
   together.
6. Multi-turn episodes (SWE, TMax): duration includes sandbox time, so tokens ÷
   duration no longer measures the miner. Subtract the signed session time or use
   another criterion.
7. Announcements made just before a checkpoint switch.
8. Protocol change: the nonce in forced-seed derivation, one more round trip per
   group, and the deployment order.
9. Reducing the 327 s gap between windows (this decides how much of the cycle
   miners can work).
10. Checking whether miners already pick short or easy prompts. Compare chosen
    prompts to a uniform draw over the same dataset range.
