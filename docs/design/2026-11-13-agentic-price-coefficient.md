# Agentic price coefficient: measure first

Date: 2026-11-13. Status: design, ON HOLD. A platform patch exists (local branch
`feat/agentic-price-coefficient` of reliquary-platform, uncommitted); no Catalyst code change. The decision waits
for the measurements below.

## Observation

Two agentic eval orders (Qwen/Qwen3.5-9B, auction pricing) ran on 2026-11-13: SWE `6e4e20e3` (100 x 4) and TMax
`7bf9e45f` (200 x 4). Their public auction views showed the price at its ceiling (0.6466 vs 0.6598 alpha per
Mtoken), a budget of 64 leases with only 10 and 4 held, and 4 miners working; about 11 TMax and 2 SWE episodes
were done in the first hour. The gateways had spare sessions.

## Hypotheses

1. **Price.** An episode yields fewer verified tokens per miner GPU-hour than a single-turn request: the GPU waits
   on tools, holds the KV cache across turns and prefills every observation. At a per-token ceiling sized for
   single-turn work, an agentic lease earns less per hour, so miners offer above the ceiling or do not offer.
2. **Not yet joined.** Agentic jobs need the episode harness and a sandbox client; few miners run them yet. A higher
   price would not help until they do.
3. **Sandbox.** Sessions are refused (fleet full, open rate, registration) before the price matters.

Each predicts a different measurement:

| Hypothesis | Tokens/h per miner, agentic / single-turn | Offers on agentic jobs | Sandbox refusals |
| ---------- | ----------------------------------------- | ---------------------- | ---------------- |
| Price      | well under 1                              | many above the ceiling | few              |
| Not joined | any                                       | few hotkeys offer      | few              |
| Sandbox    | any                                       | under the ceiling      | many             |

## Measurement (`scripts/measure_agentic_supply.py`)

Read-only, prints JSON. Its outputs are not committed.

- **Throughput** (`--job`, repeatable): for each job, the passed records' metadata (hotkey, `received_at`,
  `token_count`) from the bucket. Per hotkey, verified tokens per active hour: records are split into sessions
  at gaps over `--idle-gap` (1800 s); a session's first record is left out (its work began before the session
  did). Per model, the median miner rate of agentic and single-turn jobs and their ratio. Compare jobs of the
  same model (the 9B evals against a single-turn eval or generation job on Qwen3.5-9B). One hotkey may run
  several GPUs: compare medians, and read the miners' declared setups (`POST /miner/setup`) when a ratio looks
  off.
- **Offers** (`--auction-state` with `--api`): each live offer of the control's auction state
  (`RELIQUARY_AUCTION_STATE_DIR`, one file per job), converted to alpha per Mtoken with the job's ceiling from
  `GET /corpus/pay`, and flagged when above it. An offer above the ceiling gets no lease (`ask_above_price`); a
  hotkey with no offer gets none (`no_ask`). The job's public auction view (`/corpus/jobs/<job>/auction`) is
  printed beside it.
- **Sandbox refusals** (`--log`, repeatable): the order control's log (`docker logs <control>`), lines
  `sandbox session <what> refused for <hotkey>: <reason>`, counted by what and reason and by hotkey prefix. The
  control status also carries `quota_refusals_fleet_full` over the last ten minutes
  (`reliquary/validator/order_sandbox_status.py`).

Gap: lease refusals are answered to the miner but not logged or counted. If the offers snapshot is not enough, a
per-reason lease refusal counter in the job's auction view is the smallest addition.

Example, on the control host:

```
python scripts/measure_agentic_supply.py \
  --job order-eval-6e4e20e3-... --job order-eval-7bf9e45f-... --job <single-turn job, same model> \
  --auction-state "$RELIQUARY_AUCTION_STATE_DIR" --api https://api.reliqua.ai \
  --log control.log > /tmp/agentic-supply.json
```

## If the price is the limit: the coefficient

One source of truth: the platform's quote. Today the platform prices a subnet order's token ceilings in USD
(`subnet_token_prices`, `worker/token-prices.ts`) and sends them with the job (`max_price_usd_per_mtoken`); the
admin converts them once to alpha at the day's rates (`admin/service.py` `token_price`), and the auction's
`ceiling_alpha_per_mtoken` is that converted output ceiling (`corpus/token_auction.py` `auction_params`). Order
cap sizing (`admin/cap_sizing.py`) scales with the same ceiling. So a factor applied in the quote reaches the
miners' ceiling without any Catalyst change, and the public `/corpus/pay` view shows the USD it came from.

The platform patch (held):

- `EVAL_AGENTIC_PRICE_FACTOR` (default 2, at least 1) multiplies the input and output ceilings, and the flat
  per-token price when no price row applies, for orders whose set plays episodes. The quote records the factor
  (`pricing.agentic_price_factor`, `token_ceilings.agentic_factor`). The refund of unverified tokens uses the
  multiplied price, so the customer pays the factor only on tokens delivered. Single-turn quotes are unchanged.
- Gateway price made explicit: `GATEWAY_HOURLY_COST_USD` (0.30, the Targon VM's rent) and `GATEWAY_PRICE_MARKUP`
  (1); the quote charges cost x (1 + markup) = 0.60 USD per gateway-hour as before, then the order margin. The
  fleet's spend projection and its accounting before a VM reports its price now use the cost (0.30), not 0.60.
  `GATEWAY_HOURLY_PRICE_USD` is gone: check the deployed Worker's settings before deploying.

Setting k from data: k = median single-turn tokens per miner-hour / median agentic tokens per miner-hour, same
model, from the throughput measurement above; recheck after each change of harness or sandbox.

## Observation tokens

An episode record pays its model tokens only (`token_count` = the turns' spans); it carries no
`prompt_token_count`, and an auction job never pays prompt tokens (`corpus_period_settlement.py`). Pricing
observations as input would need episode records to count them and the auction settlement to pay input, a
larger change than the factor. Recommendation: do not price them separately; their prefill cost already lowers the
measured agentic tokens per hour, so a k set from that measurement covers it.
