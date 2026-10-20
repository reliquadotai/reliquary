# One harness definition for miners, validators and gateways, then surface variants: design

Status: FOR VALIDATION (2026-10-20). Audit behind it: `~/cc-data/harness-audit/REPORT.md` (inventory,
incidents, options). Branches: Catalyst `feat/harness-single-source` (from feat/rl-async
a5df544f + origin/feat/rl-async 0af18807), reliquary-sandbox `feat/reliquary-harness` (from
fc2fa64, the commit Catalyst pins).

Tags: **DECIDED** (the user agreed), **DONE** (on the branches, tested), **PROPOSED**
(recommended, awaiting the user), **PENDING** (waits on a measurement).

## 1. Decisions

1. **Option B** (DECIDED 10-19). Our own episode loop. verifiers stays only as the taskset
   loader (`reliquary_swe`, the sandbox bridge) and, for eval and prime-rl exports, behind the
   compatibility harness `bash-vf-b2e4e81`, whose spec reproduces today's bytes.
2. **One package, `reliquary-harness`** (DECIDED), in the reliquary-sandbox repository as its
   own distribution (`harness/`, module `reliquary_harness`), pure code, no dependency
   (`renderers==0.1.11` for the Qwen renderer, extra `qwen`, imported inside functions; the
   loader refuses any other `renderers` release, see section 6). It is a separate
   distribution so that installing a new harness commit does not move `SANDBOX_COMMIT`.
3. **Pinned by digest in the contract** (DECIDED, step 4). A process whose installed digest
   differs refuses to serve or to mine. It never bans.
4. **Steps 1 to 3 change no token** (DECIDED, DONE): the goldens frozen from the old code
   pass unchanged.

## 2. What landed (steps 1 to 3)

| Step | Where | What | Evidence |
|---|---|---|---|
| 0 | Catalyst `6c6d331d` | Goldens frozen from the old code, for two tokenizers (qwen38 `cd79c4a4…`, Teutonic `422fa182…`). (a) The validator's render: SWE signed bash+edit, a `context_length` stop on a capped turn, SWE corpus with the network notice, TMax bash only, 13 real model turns from smoke3, EnvScaler v2 ChatML. (b) The miner's loop at the generate endpoint: verifiers' real bash harness under its train client, against the real gateway over FakeBoxes, in six scenarios (honest, `max_turns`, `episode_closed`, `context_length`, capped turn, bash only). | `tests/unit/test_harness_goldens.py`, `tests/unit/test_harness_episode_parity.py` (`RUN_VERIFIERS_BRIDGE=1` replays verifiers: identical) |
| 1 | sandbox `753b208` | `reliquary_harness`: `spec` (HarnessSpec, canonical JSON, digest, `digest_refusal`, `BASH_VF_B2E4E81`), `qwen` (QwenTurnRenderer moved), `engine` (one turn, as the train client sends it), `loop` (the episode loop and its stops), `variant` (the seed draw). The verifiers files it reproduces are vendored (MIT) and pinned by sha256. | `harness/tests/`: parity with upstream (never skipped), goldens, loop, spec |
| 2 | Catalyst `c31bcdb3` | `agentic_swe.py` loses its copies (prompt, tools, filter, notice, QwenTurnRenderer). It re-exports the spec under the old names. `HARNESS_DIGEST` pins the spec, and `harness_refusal` refuses SWE and signed jobs on another digest. | goldens identical; `test_harness_pin.py` (never skipped) |
| 3 | Catalyst `2a3a4bea` | `miner/harness_episode.py`: `HarnessEpisodeRunner` replaces `SandboxEpisodeRunner` for signed corpus and RL episodes (`SignedSweEpisodeRunner._default_runner`). verifiers no longer runs the miner's loop. | the six recorded scenarios are reproduced exactly at the generate endpoint (prompt ids, sampling params, session, stop, signed calls, final, refusals) |

What the miner's loop reproduces, measured against verifiers b2e4e81 (`loop.py`):

- the initial render: system prompt and tools from the spec, `add_generation_prompt`;
- one `X-Session-ID` per episode, reported through `on_trace` before the first turn, so the
  forced draw binds before generating;
- sampling `{temperature, top_p, max_tokens}`, then generate's own `stop_token_ids`,
  `logprobs=1` and `skip_special_tokens=False`;
- `routed_experts_prompt_start = len(previous prompt + completion) - 1` on bridged turns;
- the `max_model_len` preflight, which gives `context_length`;
- before each model call, `max_turns` first, then `episode_closed`. The turn that reaches
  `max_turns` still has its calls sent;
- a final on a call stops the turn at once;
- a turn with no call gives `agent_completed`;
- the call filter and call ids of `response_from_generate`;
- the bridge with `{"role": "tool", "tool_call_id": "call_<k>", "content"}`.

Known differences, none of which reaches an admissible episode:

- When the renderer cannot bridge, verifiers re-rendered the whole conversation; the loop
  raises `BridgeFailed`. The validator refuses such an episode anyway (`next_prompt` is None,
  so `bad_observation`).
- `--harness-env` no longer reaches anything: there is no local harness program.
- Error texts differ: they are logs, never submitted.

Not yet done in step 3: the legacy Docker replay corpus miner (`miner/agentic_episode.py`,
verifiers with a restricted runtime) still runs verifiers. It is legacy, and goes in step 7.

Deploy precondition: Catalyst's `pyproject.toml` depends on
`reliquary-harness[qwen] @ git+…reliquary-sandbox.git@39b21cf…#subdirectory=harness` (pushed to
the public repository). The pinned commit must be on the public repository before any image build. The main image installs
pyproject. The CPU executor image (code grader) imports no agentic code, so it needs nothing.

## 3. Step 4: the harness in the contract (DONE on the branches)

**Corpus jobs.** `episode.harness_spec = {id, version, digest}` (`corpus/job.py`). It is
optional, and written only when set, so every existing manifest keeps its contract bytes and
its `contract_sha`. Present, it enters the contract.

- The shape is checked at parse, and `job.py` still imports nothing.
- Whether the binary serves the pin is `agentic_swe.harness_pin_refusal`, first in
  `episode_support_refusal` and `sandbox_support_refusal`. So the miner refuses to mine and the
  validator refuses to resolve the job.
- A corpus job may only pin a harness without variants: it has no validator seed to draw one
  from.

**RL episode envs.** `rl_episode_envs.<env>.episode.harness = {id, version, digest}`
(`service_contract._validate_episode`, which also serves the window order). It is optional:
absent means `bash-vf-b2e4e81@1`. `EpisodeEnvPolicy.harness_pin()` carries it.

- The miner's `build_episode_generator` and the validator's `load_episode_environment` refuse
  an env whose pin this build does not serve. The env is not loaded: a refusal, never a ban.
- The window path (`episode_mining._episode_policies`) refuses a pin with variants.

**Pins.** `agentic_swe.HARNESS_DIGESTS` lists every spec this build serves at the digest it was
built against: `bash-vf-b2e4e81@1`, `bash-surface@1` and `mini-swe@1`. `harness_refusal` checks the installed
package against them.

**Record 0 is unchanged** (DECIDED in implementation, against the earlier PROPOSED). The harness
only decides tokens, and the validator renders and compares every token itself. The variant is
drawn by both sides from the lease (section 5). A miner that plays another harness or variant
produces other tokens and is refused at admission. So signing the harness in record 0 would add
no check the validator does not already make. It would only add a sandbox protocol change and a
coordinated gateway release. The gateway never learns the harness: it runs canonical tools.

**Release**: no sandbox release.

- Miners and validators need a Catalyst build that serves the pinned spec, that is, one with the
  same reliquary-harness commit.
- Enabling variants is a registry change by the operator: add `harness` to an env's
  `rl_episode_envs` block. It is refused by any binary that does not serve that digest.

## 4. Step 5: one observation limit (coordinated release, NOT DONE)

What the model sees for a bash output today depends on the path:

| Path | Limit | Where |
|---|---|---|
| Signed gateway (RL, signed corpus) | 64 KiB, a per-gateway operator setting | sandbox `settings.py:123` `episode_output_cap_bytes` |
| Legacy Docker replay (validator) | 4 MiB of characters | `corpus_grade_protocol.py:29`, `agentic_replay.py:126` |
| reliquary_terminal's patched program | 1 MiB, 180 s per command | envs `reliquary_terminal/harness.py:32-38` |
| `environment.toml` of the bash envs | 16 KiB (8 KiB telecom) | envs, unread by the signed path |
| Family A runner (suite, EnvScaler) | 64 KiB, which raises instead of truncating | `agentic/runner.py:56` |

**Proposal** (PROPOSED):

1. The spec owns the limit: `observation.max_output_bytes` (and the per-call timeout default).
   - For `bash-vf-b2e4e81@1` the value is 65536. That is what every RL and signed-corpus
     episode already saw, so no token changes.
   - The rendering stays `reliquary-observation/1`: the output is cut by the gateway and
     flagged `truncated`, and the note is added by `render_observation`.
2. The gateway reads the cap of the harness named in record 0, never its own setting.
   - `episode_output_cap_bytes` becomes the upper bound the machine accepts.
   - A machine whose bound is below the spec's refuses that harness at startup (capacity
     report) and at open.
   - Record 0 signs the cap applied (`budgets.output_bytes`), and the validator checks it
     equals the spec's.
3. The legacy replay path renders through the same function with the same cap (step 7), until
   it is retired. `MAX_OBSERVATION_CHARS` then only bounds the wire.
4. The bash envs' `environment.toml` `max_observation_bytes` and reliquary_terminal's patched
   program stop mattering for signed episodes.
   - They are removed from the bridged path's options, and the patch is kept only for the
     prime-rl/eval use of the env.
   - The env catalogue entry's `options` must not carry an observation limit (validated in
     `validate_catalogue`).
5. Family A keeps raising at its limit until step 6 moves it into a spec. It is a different
   contract today (an over-limit observation ends the episode), and changing it is a manifest
   regen.

`observation.py` moves from `reliquary_sandbox` into `reliquary_harness` in this step. A
re-export stays in `reliquary_sandbox.observation`. The sandbox distribution then depends on
reliquary-harness at the same commit, which is a single repository release.

**Tests**: truncation goldens at the cap, one byte under it and one byte over it, both
streams; the gateway refusing a lower bound; record 0's signed cap checked by the validator.

## 5. Step 8: surface variants (DONE on the branches)

**Spec** (`reliquary_harness.spec`). A spec's axes change what the model is shown, never what
runs:

- `tools`: a surface name and description per canonical tool. The parameters stay canonical.
- `system_prompt`: one fragment per tool, where `{bash}` and `{edit}` stand for the surface
  names.
- `observation`: `plain`, or `exit-code`, which puts the signed exit code on a line before the
  gateway's rendering. Refused calls are never decorated.

`bash-surface@1` has 2 tool namings (`bash`/`edit`, then `shell`/`str_replace`) × 2 prompt
phrasings, so 4 variants. Its variant 0 renders `bash-vf-b2e4e81` token for token. The
observation axis is implemented and tested end to end (`test_harness_variants_e2e`), but it is
not in `bash-surface@1`: ship few variants first, and measure.

**Draw** (`agentic_swe.lease_variant`). The draw is
`sha256(len‖nonce ‖ len‖"rl-lease:{lease_id}" ‖ len‖prompt_idx ‖ len‖digest) mod 4`.

- The lease's draw nonce derives from the validator's secret, so the miner cannot choose the
  variant.
- Both sides compute it alone.
- All 16 rollouts of a lease share it, so a group's advantages compare policies, not surfaces.

**Reading back** (`QwenTurnRenderer`).

- The calls read back (`tool_calls`, `calls_of`) carry the canonical names. So `plan_call`, the
  gateway and the transcript only ever see `bash` and `edit`.
- The assistant message keeps the written names, because it must re-render to its span (the
  canonical round trip and export).
- A canonical name the variant hides reads as `?bash`, a tool nobody offers (`not_offered`). So
  a hidden name is never executed.

**Wiring**:

| Side | What changed |
|---|---|
| Miner | `EpisodeLeaseGenerator.play_lease` draws the variant, then uses its renderer for the admission precheck. `LeaseEpisodeRunner(harness=, variant=)` passes it to `HarnessEpisodeRunner(spec=, variant=)`, which renders with it and decorates observations. |
| Validator | `LeaseEpisodeChecker` draws the variant per lease and grades with that variant's renderer and prompt (`renderer_of`, one per variant). `signed_parse` and `episode_admission` render signed records in the renderer's observation format. |

**Tests**:

- Per variant: prompt and round trip goldens (`harness/tests/goldens/surface-variants.*`).
- Variant 0 equals vf.
- The miner plays each variant against the real gateway. The gateway sees canonical tools only.
  The validator's parse admits the episode under that variant and refuses it under any other
  that renders differently.
- `exit-code` is read back only as such.
- The draw is a function of the lease alone, and all 4 variants are reached.

**To measure before turning it on** (PENDING): `bad_turns` and the valid-call rate per variant on
an RL smoke. Telecom's 0/38 with unfamiliar markup is the warning. Turning it on means adding
`harness: {bash-surface, 1, <digest>}` to the SWE/TMax `rl_episode_envs` blocks.

## 6. Choosing the harness, and `mini-swe@1` (DONE on the branches)

**At task creation.** `reliquary tasks create --rl-async … --rl-episode-envs envs.json --harness
swe_v1=mini-swe@1,tmax_v1=bash-surface@1` writes `{id, version, digest}` into each named env's
`episode.harness` (`cli.main.with_episode_harnesses`). The digest is the one this build serves,
so a typo or a stale build fails at creation. An env not named keeps the compatibility harness.
One harness per env; a weighted mix per env is open (section 9).

**`mini-swe@1`** (`reliquary_harness.mini`, texts from mini-swe-agent 04d809ce, MIT): no tool
schema in the prompt; the model writes one command in a ```` ```mswea_bash_command ```` block.

- Exactly one block is one canonical `bash` call. The gateway runs it as any other call.
- `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` ends the episode without running (the gateway
  grades the box; there is no output to read a marker from).
- Zero or several blocks: a local call to `?format` (or `?length` for a turn cut by its token
  cap), answered with mini's error text; the episode goes on. These are tools nobody offers,
  so the gateway never runs them, and the validator reads them back the same way.
- The observation is mini's `<returncode>…</returncode>\n<output>…</output>` as a user
  message, head and tail past 10000 characters; a timed-out or cut command shows the gateway's
  flags in `<exception>`, not a Python exception.
- A corpus job cannot pin it: corpus jobs play `bash-vf-b2e4e81@1` only.

**Smoke lessons** (2026-10-20, Qwen3.5-2B on one H100, forced draw, real gateway):

- **The renderer release is part of the harness.** A fresh install took `renderers` 0.1.12
  (Catalyst depended on reliquary-harness without its `qwen` extra, so only verifiers'
  `renderers>=0.1.10` chose it). 0.1.12 refuses to bridge a turn closed inside its reasoning,
  which mini answers with a format error and goes on: every SWE lease was given up with
  `BridgeFailed`. Fix: Catalyst depends on `reliquary-harness[qwen]`, and `build_renderer`
  refuses any release other than 0.1.11. The digest covers our sources only, so without that
  check another release changes tokens silently. Production images already locked 0.1.11.
- **A foreign format costs whole leases.** A lease is 16 forced rollouts and one refused rollout
  gives it up. The 2B model, unused to mini's format, sometimes writes a fake output and opens
  its own next turn (`<|im_start|>` inside the span): `bad_turns`, exactly what that check is
  for. Measure the refused-rollout rate per harness before opening a harness to miners; a model
  trained on the format makes it rare.

## 7. Remaining migration

Step 5 needs a coordinated release. Step 6 needs a manifest regen and step 7 an
executor coordination.

| # | Step | Release | Gate |
|---|---|---|---|
| 4 | Contract triplet (corpus `harness_spec`, RL `harness`), refusals: **DONE** | Catalyst build only | `test_harness_contract`, the 6 scenarios |
| 5 | One observation limit, `observation.py` into the harness | **coordinated** | truncation goldens; 64 KiB keeps every token |
| 6 | Family A (ChatML, EnvScaler, telecom) as specs; `renderers.py` delegates | manifest regen (`scripts/regen_episode_manifests.py`) | `test_chatml_episode_consensus.py`, the EnvScaler golden |
| 7 | Legacy Docker replay: `TOOL_PROGRAM` onto `format_observation`, then retired; the legacy miner off verifiers | executors | replay tests |
| 8 | Variants: **DONE** (off until a block names `bash-surface@1`) | Catalyst build + registry block | `test_harness_variants_e2e`, an RL smoke per variant |
| 9 | Weighted harness mix per env (draw per lease, as variants), metrics per harness | Catalyst build + registry block | to design |

Rollback for step 3, before any release: reverting the step 3 commit restores verifiers'
loop (`_context` and `_default_runner`), with the same tokens. Nothing else depends on it.

## 8. Validation evidence

**Without a GPU** (VPS, one file at a time, `RELIQUARY_HARNESS_TOKENIZERS` set to both
tokenizers):

- reliquary-harness: `test_spec`, `test_qwen_goldens`, `test_upstream_parity` (never skipped),
  `test_variants`, `test_loop`, `test_mini` (with the reasoning-closed turn and the renderers
  release refusal): green.
- Catalyst: `test_harness_goldens` (13), `test_harness_episode_parity` (13, plus 12 that replay
  verifiers when `RUN_VERIFIERS_BRIDGE=1`), `test_harness_episode`, `test_harness_pin`,
  `test_harness_contract` (12), `test_harness_variants_e2e` (5), `test_harness_mini_e2e`,
  `test_rl_async_episodes` (17), `test_episode_inertness`, `test_episode_protocol`,
  `test_episode_group_miner`, `test_rl_episode_miner_side`, `test_service_episode_wire`:
  green. The goldens frozen from the old code are unchanged byte for byte.

**RL smokes** (Qwen3.5-2B, async RL, forced draw, real signed gateway over runc, local S3,
one miner, one validator, one trainer; 2026-10-20):

| Run | Harness | Leases submitted | Admitted by the validator | Audits (forced redraw) |
|---|---|---|---|---|
| A (H100 PCIe) | `bash-vf-b2e4e81@1`, SWE + TMax | 8 | 8 | all passed, min match ≥ 0.978 |
| hBC (same box) | TMax `bash-surface@1` | 5 | 5 (variants 1, 2, 3) | 4 passed, ~0.99 |
| hBC | SWE `mini-swe@1` | 0 | — | all given up: `BridgeFailed`, renderers 0.1.12 (section 6) |
| hC (2× H100 SXM, 6a8096d7 + 39b21cf) | TMax `bash-surface@1` | 15 | 15 (all 4 variants; 4 in zone) | 11 passed, 0 failed, min match 0.979 |
| hC | SWE `mini-swe@1` | 12 | 11 (3 in zone), 1 refused (below) | 8 passed, 0 failed, min match 0.981 |

In hC the trainer took one step on 64 rollouts (4 in-zone groups, one of them a mini-swe SWE
group). On the miner, 2 rollouts out of the roughly 460 played (29 leases) failed the shared
admission check, and each gave up its lease. One was a mini-swe rollout refused for
`bad_turns`: the 2B model wrote a fake output, then `<|im_start|>assistant` inside its span.
The other was `bad_termination`. Both are the checks working, not a harness defect.

The one refused SWE lease (`episode_transcript`) was the smoke's storage, not the harness.
Under a load average near 100, a client aborted its PUT mid-body ("unexpected EOF"). The
local S3 (SeaweedFS) then committed an empty object, here the machine record. The validator
lost the machine's public key, 76 s before that lease's check. After that, the controller's
heartbeat, which reads before it writes, could not repair the record. R2 refuses an upload
shorter than its declared length, so production does not see this. The smoke re-registered
the machine by hand.

Not yet measured: `bad_turns` per variant on a model trained in the format, and any harness
on the production 27B/Teutonic models.

## 9. Open points

- **The digest covers source bytes**, so a comment edit in `qwen.py` is a new digest, hence a
  new contract (as `implementation_sha256` does in `registry.py`). Alternative: hash the
  normalized AST. PROPOSED: keep bytes, and batch doc edits with real changes.
- **The goldens need the tokenizers on the host.** They skip by name without them. CI should
  provide the two tokenizer files (about 13 MB), or a vendored `tokenizer.json.gz`.
- **renderers 0.1.11's parameter whitespace stripping** is still worked around by
  `span_is_canonical`. Vendoring the Qwen3.8 renderer into the package would freeze it; that
  is a new spec version, not this one.
- **A weighted mix of harnesses per env.** Same draw as variants (lease nonce), over
  `[(pin, weight)]`; validator and miner compute it alone. With it, metrics per harness
  (admission, `bad_turns`, reward) in the controller's logs.
- **A refused episode's reasons are not logged by the RL lease checker.** `LeaseEpisodeChecker`
  logs only the stage (`episode_transcript`), and for variant 0 an admitted lease logs nothing.
  Log the refusal's detail and the harness on every lease.
- **A machine record that is not JSON stays broken**: `_update` raises before writing. Only a
  store that commits truncated bodies produces one, but a heartbeat could rewrite it from its
  registration.
- **The sandbox bridge still imports verifiers to load tasksets.** Moving taskset loading off
  verifiers is a separate change.
