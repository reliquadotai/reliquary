# Corpus jobs on suite episode environments

2026-10-13. Status: implemented on the validator side (`feat/corpus-suite-episodes`).
The miner side follows this document.

## What it is

A corpus (SFT distillation) job whose unit is one multi-turn tool episode of
an episode environment already in the registry (`reliquary.environment.registry`,
`interaction_mode="episode"`, ChatML renderer). The first ones are
`reliquary_telecom_solo_v1` and `reliquary_telecom_solo_v2`. Any later
environment works once it is registered, as long as its renderer is
`reliquary-chatml-tools-v1`.

SWE episode jobs (`harness: "bash"`, `reliquary-swe`) are unchanged.

## Manifest

```json
"prompt_source": "<env name>",
"renderer_id": "reliquary-chatml-tools-v1",
"sampling": {"n": 1, "max_new_tokens": <episode.max_tokens_per_turn>, ...},
"slots_per_prompt": 2,              // >= 2
"filter": null,
"prompt_order": "free",
"episode": {
  "env": {"package": "reliquary-suite", "name": "<env name>",
          "version": "<the spec's environment_manifest_sha256, 64 hex>", "split": "train"},
  "harness": "suite",
  "renderer": "reliquary-chatml-tools-v1",
  "max_turns": 40,                  // 1..64, and at most the environment's own max_turns
  "max_tokens_per_turn": 4096,
  "max_total_tokens": 60000         // prompt + trajectory, <= 60000
}
```

- A suite episode has no `verifiers`, no `num_images` and no
  `replay_fraction_failed`. A manifest carrying any of them is refused.
- `harness` and `package` go together: `bash` runs `reliquary-swe`, `suite`
  runs `reliquary-suite`.
- The `_check_episode_job` rules still apply: `n == 1`, `slots_per_prompt >= 2`,
  no filter, `prompt_order == "free"`,
  `max_new_tokens == max_tokens_per_turn`, `renderer_id == episode.renderer`.
  One rule is added: `prompt_source == episode.env.name`.
- Where the job is resolved (`jobs create`, a validator serving it, export),
  the environment must be in the registry and be an episode environment. Its
  `environment_manifest_sha256` must equal `episode.env.version`, its
  `renderer_id` must equal `episode.renderer`, and that renderer must be ChatML
  (`suite_episode.suite_spec_refusal`). Each failure is refused with a message
  that names the field.
- `episode.max_turns` above the environment's own `max_turns` is refused,
  never capped (`SuiteSource`, run by `jobs create` and by a serving
  validator). The turn limit is therefore `episode.max_turns`, which equals
  `min(episode.max_turns, environment.max_turns)` as in RL.
- Binaries older than this change refuse the manifest
  (`episode is missing: verifiers, replay_fraction_failed`), so a job with
  this harness fails closed on them.
- Range guards (`reliquary.eval.sets`):
  - `reliquary_telecom_solo_v2` serves corpus jobs inside `[0, 1220)` only.
    That is its train SFT share, from `environment.toml [data]`. RL keeps
    `[1220, 1713)` (`RL_ONLY_RANGES`).
  - `reliquary_telecom_solo_v1` serves no corpus job, because its train split
    still holds the tau2-bench benchmark tickets.
  - Enforced at `jobs create` and again when a validator builds the job's
    intake (`build_suite_intake`), so an edited manifest is not served.
- An environment that no profile declares (telecom v2 today) gets its contract
  body from the registry spec and the job's episode bounds
  (`agentic_source.suite_episode_body`). This body is for corpus contracts
  only: RL on such an environment still needs a reviewed catalog and profile
  entry.

## Miner

For a job and a `prompt_index` in `[prompt_start, prompt_start + prompt_count)`:

1. `environment = registry[episode.env.name].create()` (train split) and
   `task = environment.get_task(prompt_index)`.
2. `renderer = renderer_for(episode.renderer, encode)`. The prompt tokens are
   `renderer.encode_initial(task)`, and `rendered_prompt = renderer.initial_text(task)`.
3. `seed = episode_seed(job_id, prompt_index)`. This is
   `int(sha256(f"{job_id}:{prompt_index}").hexdigest()[:16], 16)`, the
   first 8 bytes read big-endian, as in RL. Import it from
   `reliquary.corpus.suite_episode`.
4. Run `EpisodeRunner(renderer=renderer, max_turns=min(episode.max_turns, environment.max_turns))` with
   the model as the policy. Each turn is generated up to `<|im_end|>`, which
   stays in the turn's tokens, or up to `max_tokens_per_turn`. Generation
   continues from the exact running token list (append-only).
5. Submit the same trajectory format as SWE
   (`protocol/corpus_submission.py`):
   - `tokens`: every token after the initial prompt, up to the end of the
     last assistant span. What the runner appends after it (the final suffix,
     or an observation after a last call at the turn limit) is not sent.
   - `turns`: `[{start, end, proofs}]` in those coordinates. The first turn
     starts at 0.
   - `final_diff`: `""`.
   - `stop`: `"agent_completed"` when the environment ended the episode, or
     `"max_turns"` when the turn limit did. `"context_length"` is refused.

`encode` and `decode` are the fast tokenizer's backend with no special tokens
added or skipped (`suite_episode.TokenizerCodec`), as RL replay workers code.

## Validator

### Intake

`validator/suite_intake.py` is the twin of `EpisodeIntake`. It runs these
checks in order:

1. Shape, ownership, `final_diff == ""`, and `stop` in `SUITE_STOPS`.
2. `rendered_prompt` equals `initial_text(task)`.
3. Vocabulary.
4. The generic turn checks:
   - spans, short turns and budget
   - each span is one turn (`suite_episode.suite_span_refusal`, which applies
     the SWE rule `trajectory_parse.check_span_tokens`): a stop id
     (`<|im_end|>` or `<|endoftext|>`) only as the span's last token, and no
     turn markup (`<|im_start|>`, and the tool-response tags where they are
     single tokens) anywhere in it, nor spelled out as plain text. Without
     this, one span could hold a call, a made-up tool response and a final
     answer, all of it accepted by the replay, paid and trained as the
     model's. Miners precheck with the same function.
   - termination: a turn before the last ends on `<|im_end|>` or exactly at
     the cap, and the last turn may also end on `<|endoftext|>`, so
     `<|endoftext|>` mid-episode is `bad_termination`
   - proof shape
5. The replay. It reconstructs what RL admission's `replay_tokenized_episode`
   reconstructs, with the turn limit
   `min(episode.max_turns, environment.max_turns)`. A miner that replays with
   `replay_tokenized_episode` itself, which uses the environment's limit,
   reads the same result over the submitted tokens. That function is not edited
   because it is in the builtin environments' pinned implementation digest.
   The replay runs in a spawned, CPU-only process pool that holds the
   checkpoint tokenizer. The 30 s wall deadline starts when a worker starts
   the replay, not while it waits in the queue. The route's worker thread
   waits on the pool, never the event loop. The intake compares tokens up to
   the end of the last span (prompt included), the spans (and so the number
   of actions), and the termination against `stop`.

   The intake also refuses an observation rendered into the trajectory that
   contains `<|im_end|>`, `<|im_start|>` or `<|endoftext|>`
   (`suite_episode.observation_markup`). The renderer writes observation text
   verbatim, so a marker in it becomes the real token, which would give a turn
   nobody generated. An environment can echo an argument, and a JSON argument
   can spell a marker with escapes that no span check sees: telecom's
   `set_apn_settings` does both. The export runs the same check and never
   writes such a tool message.

A mismatch is an intake refusal (`bad_observation`, `bad_turns` or
`bad_stop`). It costs no slot and carries no sanction. The replay's
`task_id`, `reward` and `success` go into `IntakeFacts`.

`ReplayUnavailable` means the route answers 503, never a refusal. It is raised
in three cases:
- A replay ran past its deadline. A replay is about 30 ms of work, so 30 s
  means the host is starved.
- A worker died twice in a row. A broken pool is rebuilt and the replay
  retried once, which is logged as an error.
- There was no answer within 180 s, queue included.

The deadline is a flag set by the timer itself. An environment that catches
every exception (telecom does) cannot turn the deadline into an observation.

After a deadline, or after a started replay that never answers, the pool is
retired, its workers are killed and the pool is rebuilt. Each worker runs
under a 2 GiB `RLIMIT_AS`. Intake checks run on their own bounded threads (8),
never on the loop's default executor, which the ledger turns use.

A turn before the last that ends on a stop id other than `<|im_end|>` is
`bad_termination`, even when it ends exactly at the cap.

### Grading

`validator/suite_grading.py`: `LocalEpisodeGrader` subclasses `CorpusGrader`
and keeps `ready`, `oldest_unready_received_at`, `enqueue`, `run` and
`status`. It replays each stored record again, which is deterministic and
survives a restart, and writes a `reliquary/corpus-grade/v1` document with
these fields:

- `status: "ok"`
- `graded_success`: the environment's success
- `instance_id`: the task id
- `replay: {status: "ok", certified: true, drawn: true, ...}`
- `replay_certified: true`

A failed audit is graded `audit_failed`. A record that no longer replays is
graded `unreplayable`, which certifies nothing and sanctions nobody.

A replay past its deadline is not graded and is retried at the next rescan,
because intake already replayed it within the deadline, so the host was
starved. After 3 misses it is graded `unreplayable`, so one submission never
holds every period. `ReplayUnavailable` is retried the same way. The grader
runs as many grades at once as the pool has workers.

Suite jobs have no grade-executor pin, no replay lease check, no
`grade_refusal`, and no quarantine hold or regrade (`remote_graded`).
Settlement and export filtering are unchanged.

### Audit

The TOPLOC audit is unchanged because it is environment-agnostic.

## Export

`reliquary jobs export JOB --sft` (graded successes only, as for SWE) writes
one row per certified trajectory (`delivery.suite_episode_row`). The rows are
rebuilt by replaying with the job's checkpoint tokenizer and the installed
environment:

- `tokens`, `assistant_mask`: the training form that `reliquary-post-training`
  reads as is (`kind: episode`). It also reads `prompt_index`, `task_id`,
  `graded_success` and `replay_certified`.
- `tokens_transform: "chatml-single-im-end-v1"`. The model's turn keeps the
  `<|im_end|>` it stopped on, and the ChatML renderer's observation text opens
  with its own `<|im_end|>\n`, so the proven transcript holds
  `<|im_end|><|im_end|>\n`. The export drops the observation's copy, which is
  never in the mask. A turn cut at the cap keeps it, because it is that turn's
  only terminator. After a last turn that ends on `<|im_end|>`, the export
  appends the template's `\n`, unmasked. The RL renderer is not changed,
  because it is consensus.
- `+final-endoftext-as-im-end-v1` is added to `tokens_transform` only on the
  rows it applies to. A last turn that stopped on `<|endoftext|>` has that
  token replaced by `<|im_end|>` (still masked) and then gets the `\n`, so the
  row is the template's transcript and the model learns to stop on
  `<|im_end|>`. No other span token is ever changed.
- `messages`: system (the episode guidance), user (the task), then for each
  turn an assistant message with `reasoning_content`, `content` and
  `tool_calls` (`{type: "function", function: {name, arguments: <object>}}`),
  followed by the `tool` messages (anything else the environment said becomes
  `user`) up to the last turn.
- `tools`: `[{type: "function", function: <tool>}]`.
- `reward`, `turns`, `stop`.

For a well-formed episode, `tokens ==
encode(apply_chat_template(messages, tools=tools))` under the Qwen3.5 template
(tested).

The Parquet `deliver` path refuses suite jobs.
