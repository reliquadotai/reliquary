# One harness definition for miners, validators and gateways, then surface variants: design

Status: DRAFT (2026-10-19). Audit behind it: `~/cc-data/harness-audit/REPORT.md` (inventory,
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
   (`renderers==0.1.11` for the Qwen renderer, imported inside functions). It is a separate
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
`reliquary-harness @ git+…reliquary-sandbox.git@753b208…#subdirectory=harness`. That commit
must be pushed to the public repository before any image build. The main image installs
pyproject. The CPU executor image (code grader) imports no agentic code, so it needs nothing.

## 3. Step 4: the harness in the contract (coordinated release)

**Contract** (PROPOSED). `episode.harness` becomes `{"id", "version", "digest"}` in place of
`"bash"` (`corpus/job.py:93-121, 505-545`). The digest enters `contract_sha`.

- The `renderer` field is then derived from the spec, and is checked equal when present.
- So is the `verifiers` field. It stays in the contract only while the legacy replay path
  exists, because that path runs verifiers.
- An old job holding `"bash"` keeps parsing as `bash-vf-b2e4e81@1` with no digest check, so
  existing contracts and their `contract_sha` do not move.
- A new job must carry the object.

**RL profiles** (PROPOSED). The `rl_async_episodes` block of a signed episode env
(`validator/rl_async_episodes.validate_episode_block`, served by `GET /rl/episodes`) carries
the same triplet. `async_episode_policy` exposes it. `build_episode_generator` checks it
with `digest_refusal` before loading the env, and so does the validator when it builds the
env.

**Record 0** (PROPOSED, sandbox protocol change).

- The open body gets `harness: {id, version, digest, variant}` next to `tools_version`.
  The gateway takes it from the session token (section 5) and signs it.
- A gateway that does not hold that digest refuses the open, with a new signed open-refusal
  reason `harness_unsupported` (a retryable 503 on the miner side, like the full-machine
  refusal).
- `transcript_refusal` (miner) and the signed intake/admission (validator) require record 0's
  triplet to equal the contract's. Otherwise the reason is `sandbox_transcript` with
  `why: harness`. It is refused, consumes nothing and never bans.

**Refusal, never ban** (DECIDED): every mismatch is a "cannot serve":

| Where | On a digest mismatch |
|---|---|
| Miner | Refuses to mine the job (`episode_support_refusal`, already wired). |
| Validator | Refuses to resolve the job (`corpus_service.py:486-488`, already wired), so no intake. |
| Gateway | Refuses the open. |

**Release**: miner, validator and gateway move together, like `SANDBOX_COMMIT` today. The
order is: gateway image with the new harness commit, then the validator and miners (Watchtower
on main after the user's go), then jobs re-issued with the triplet.

**Tests**:

- `test_corpus_binding_golden.py` regenerated for the new contract field. An old contract's
  sha must stay the same.
- A refusal test per side.
- The six recorded scenarios still pass.

## 4. Step 5: one observation limit (coordinated release)

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

## 5. Step 8: surface variants drawn from the validator's seed

**Axes** (PROPOSED). Each axis is a finite list in the spec, so it is part of the digest.
Adding a value is a new spec version.

- `tool_names`: (surface name, description) per canonical tool. For example `bash` becomes
  `bash` / `shell` / `run_command`, and `edit` becomes `edit` / `str_replace` /
  `replace_in_file`. Parameter names stay canonical in a first version: renaming them doubles
  the parse risk for little gain.
- `system_prompt`: N phrasings of the two fragments. Start with N = 3: today's, plus two
  rewordings of the same meaning.
- `observation`: M renderings of a signed record. M = 2 at first:
  - today's (`reliquary-observation/1`);
  - a header form, `exit code: N` followed by the output and the notes. The record already
    signs `exit_code`, so the validator can render it.
- `renderer`: not an axis for now. It would change the chat template under a trained policy.

Starting cardinality: 3 × 3 × 2 = 18 variants, but **ship 2 to 4 at first** (PENDING, measure
`bad_turns` and the valid-call rate per variant). Telecom's 0/38 with unfamiliar markup
(MEM:project_telecom_chatml_breaks_tools_2026_09_20) is the warning.

**Draw** (`reliquary_harness.variant`, DONE as a function, PROPOSED as a protocol). The draw
is `variant_index = sha256(len‖seed ‖ len‖engagement ‖ len‖task_index ‖ len‖digest) mod
|variants|`, never `hash()`. Its inputs:

- `seed`: the validator's lease draw nonce (RL) or the job's drand round (corpus). The
  validator knows it before the episode opens, and the miner cannot choose it.
- `engagement`: `rl-lease:{lease_id}` (RL) or `corpus:{job_id}:{prompt_index}`.

**One variant per group** (PROPOSED). The RL engagement deliberately excludes the rollout
index, so a lease's 16 rollouts play the same variant. Their advantages then compare policies,
not surfaces. Groups across leases and prompts cover the variants.

**Who learns it, and who checks it** (PROPOSED). The variant goes through these hands:

1. The validator computes the index at grant and puts it into the session token claims (a new
   field, `harness_variant`, signed by the validator).
2. The gateway reads it at open and signs it in record 0 (section 3).
3. The miner renders with record 0's variant.
4. The validator's admission recomputes the index and refuses on a mismatch (`bad_variant`).
   During a burn-in it is inconclusive and does not count toward probation, as in
   MEM:project_next_rl_run_branch_2026_10_05.

**Surface to canonical mapping** (PROPOSED).

- `parse(spec, variant, completion)` maps surface tool names to canonical ones before
  `plan_call`. The gateway only ever sees `bash` and `edit`, so transcripts stay canonical and
  `verify_transcript` is unchanged.
- An unknown surface name, including a canonical name the variant does not show, stays a call
  to an unoffered tool (`not_offered`), as today.
- `render_observation(body, variant)` picks the rendering. The validator renders the signed
  record with the same variant and compares tokens, as now.

**Tests**:

- Golden per (spec version, variant, tokenizer).
- `parse(render(action)) == action` for every variant, including indentation and an opened
  `<think>`.
- Two hosts replaying one transcript give zero token difference.
- The draw's distribution and its stability across Python versions.
- An RL smoke with several variants, measuring `bad_turns` per variant.

## 6. Remaining migration

Steps 4, 5 and 8 need a coordinated release. Step 6 needs a manifest regen and step 7 an
executor coordination.

| # | Step | Release | Gate |
|---|---|---|---|
| 4 | Contract triplet, record 0 signs it, gateway refusal | **coordinated** (gateway, validator, miner) | binding goldens, refusal tests, the 6 scenarios |
| 5 | One observation limit, `observation.py` into the harness | **coordinated** | truncation goldens; 64 KiB keeps every token |
| 6 | Family A (ChatML, EnvScaler, telecom) as specs; `renderers.py` delegates | manifest regen (`scripts/regen_episode_manifests.py`) | `test_chatml_episode_consensus.py`, the EnvScaler golden |
| 7 | Legacy Docker replay: `TOOL_PROGRAM` onto `format_observation`, then retired; the legacy miner off verifiers | executors | replay tests |
| 8 | Variants | **coordinated** | section 5 |

Rollback for step 3, before any release: reverting the step 3 commit restores verifiers'
loop (`_context` and `_default_runner`), with the same tokens. Nothing else depends on it.

## 7. Open points

- **The digest covers source bytes**, so a comment edit in `qwen.py` is a new digest, hence a
  new contract (as `implementation_sha256` does in `registry.py`). Alternative: hash the
  normalized AST. PROPOSED: keep bytes, and batch doc edits with real changes.
- **The goldens need the tokenizers on the host.** They skip by name without them. CI should
  provide the two tokenizer files (about 13 MB), or a vendored `tokenizer.json.gz`.
- **renderers 0.1.11's parameter whitespace stripping** is still worked around by
  `span_is_canonical`. Vendoring the Qwen3.8 renderer into the package would freeze it; that
  is a new spec version, not this one.
- **The sandbox bridge still imports verifiers to load tasksets.** Moving taskset loading off
  verifiers is a separate change.
