# GPU check: the forced-draw audit of a multi-turn suite episode

Status: test plan, 2026-11-13. Not run: it needs a GPU host with the order's checkpoint.

Branch `feat/suite-episode-orders` lets an eval order play a suite set (a replayed tool world: telecom,
EnvScaler, AutomationBench) under `sampling.draw = gumbel_max`. Every piece is unit tested on CPU with a
scripted engine. What CPU tests cannot show is that the draw a miner's vLLM engine follows over a whole
episode is the draw the auditor recomputes from one forward over the stored trajectory. Single-turn draws and
signed SWE episodes are measured; a suite episode differs in three ways that this check covers:

- observations are rendered by `reliquary-chatml-tools-v1`, not the `renderers:qwen38` template;
- the draw position runs over model tokens across turns (`base_offset` = model tokens so far), with tool
  observations between spans that take no position;
- an episode may end as a `context_length` cut, or with a turn cut at `max_tokens_per_turn`.

## Setup

- One H100/H200 host, a Qwen3.5-family checkpoint (the smallest pinned one; Qwen3.5-9B is enough).
- Catalyst at `feat/suite-episode-orders`, the env wheel `reliquary-telecom-solo` 0.2.0a1 (and later
  `reliquary-automationbench`) at the registry's manifest pin.
- A local order control and miner on the same host (no production): the admin writes the order into a local
  bucket, as `tests/unit/test_order_suite.py` does, then `reliquary corpus mine-agentic` against it.

## Runs

1. **Honest, T = 1.** Telecom benchmark set, 16 tasks x 4 samples, `max_turns` 40, `max_tokens_per_turn`
   4096, `max_total_tokens` 32768. Expect every sample accepted by intake, every audit verdict passed.
2. **Honest, warped.** Same, `temperature` 0.7, `top_p` 0.95 (the orders customers send).
3. **Cuts.** Same tasks with `max_total_tokens` 6144, so most episodes end as `context_length`; and
   `max_tokens_per_turn` 256 so some turns end at their cap. Expect cuts accepted, graded 0, audits passed.
4. **Long episode.** AutomationBench (when its wheel exists): episodes of 30-50 turns and 40-60k tokens, the
   audit forward at its longest.
5. **Negative controls.** (a) A miner with a re-rolled draw on one turn (seed offset +1 from turn 3);
   (b) the honest trajectory with one observation edited (must be refused at intake by the replay, never
   reaching the audit); (c) another checkpoint (the 4B) mining the 9B order.

## What to record (per run)

- Per sample: positions, stochastic positions, exact share, max miss gap, far misses, last-token gap
  (`draw_audit.diagnostics_to_wire`), against the judge's floor (0.98), grace (8), gap tolerance (4.0) and
  terminal gap (3.0).
- Per turn: the share of misses by turn index, to see whether drift grows across turns (the forward over a
  long trajectory versus the miner's per-turn prefill with prefix cache).
- Audit seconds per episode and tokens per second, to size executors for long episodes.

## Pass criteria

- Runs 1-4: no honest sample refused; exact share >= 0.99 at p1; no far miss concentrated on the first token
  after an observation (would mean the observation positions leak into the draw count).
- Run 3: every `context_length` sample accepted at intake and graded `reward 0`; a turn cut at its cap passes
  the terminal-gap rule (its last token is a drawn one, not an end token).
- Run 5: (a) refused by the draw judge and charged; (b) refused at intake (`bad_observation`); (c) refused,
  match under the 0.90 charge floor.

## If it fails

- Misses right after each observation: check `GenerateEngine._model_tokens` against the auditor's span
  position count (`row_diagnostics`), and the renderer's assistant prefix tokens (they must be prompt, not span).
- Drift growing with turns only: compare a forward over the full trajectory with per-turn prefills on the
  same host (prefix caching on/off) before touching the judge.
