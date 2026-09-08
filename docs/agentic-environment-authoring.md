# Reliquary Episode v1 authoring

Reliquary Episode v1 is the single multi-turn format for every new native tool
environment. The consensus object is Reliquary-owned; MCP and Prime Verifiers
are interoperability boundaries, not reward or replay authorities.

## The three environments shipped in the first suite

1. `reliquary_stateful_tools_v1` — SQLite-backed customer, order, refund and
   support workflows across address-update, refund and support-note families.
   This is the flagship and first training candidate.
2. `reliquary_retrieval_tools_v1` — frozen search/open/cite workflows. This is
   the cheapest transfer-oriented second lane, with single-evidence,
   multi-hop-alias and revision-resolution tasks.
3. `reliquary_workspace_tools_v1` — isolated one-file repair with deterministic
   tests and final-state checks across four basic bug families. This is the
   higher-cost curriculum lane.

All three use the same `EpisodeEnvironment` methods, action JSON, renderer,
runner, replay, signed metadata, assistant-span mask and qualification tool.

## Add a new environment

Create `reliquary/environment/agentic/envs/<name>_v1/` with an environment class
that implements:

```python
name: str
validator_authoritative_reward = True
max_turns: int
__len__()
get_task(index)
reset(task, seed)
step(task, state, action)
grade(task, state, trace)
close(state)
```

Keep task creation deterministic from the integer index. Put hidden initial
state, expected invariants and `reference_actions` in `EpisodeTask.private`;
none of it is rendered to the model. A step returns only JSON-safe tool events.
Grade final state and explicit invariants, never prose style.

The episode-level training reward is binary: every required check must pass for
reward `1.0`; any failed check produces `0.0`. Keep weighted checks as verifier
diagnostics so operators can identify which capability failed, but do not feed
partial credit into GRPO. A task generator should expose several meaningful
families through `task.metadata["family"]` instead of producing cosmetic
variants of one template.

An invalid JSON action or unknown tool is an immediate terminal binary failure.
Once `no_invalid_tool_calls` has failed, success is impossible, so continuing to
sample would only consume GPU tokens without changing the outcome. New binary
environments should use the same fail-fast rule unless their reward contract
explicitly makes recovery from malformed actions possible.

Then:

1. Add the environment package to `agentic/envs` and `BUILTIN_EPISODE_ENVIRONMENTS`.
2. Add one immutable `EnvironmentSpec` with `interaction_mode="episode"`.
3. Commit a canonical manifest. Bind its schema, checked-in golden fixture and
   complete implementation-file set by SHA-256, then bind the canonical
   manifest SHA-256 in both the registry and a new opt-in protocol profile.
   Never change an activated v1 generator or verifier in place.
4. Add accepted, rejected, collateral-mutation and family-diversity tests.
5. Add fixed indices `0`, `1`, `2` and `7` to the environment's JSONL golden
   fixture and verify them with `tests/unit/test_episode_goldens.py`.
6. Run CPU qualification and cross-host corpus comparison.
7. Run model shadow collection on a GPU host before allowing optimizer steps.

## Canonical wire behavior

The model emits exactly one JSON object per turn:

```json
{"tool":"tool_name","arguments":{"key":"value"}}
```

or:

```json
{"final":"answer"}
```

The renderer appends each action and validator-produced observation without
re-tokenizing the existing prefix. The signed v8 metadata binds actions,
assistant spans, observation digests, termination, final state and semantic
trace digest. Admission reconstructs the complete token transcript and replays
the environment. Training receives only the assistant spans copied into a
private field after that replay succeeds.

The machine-readable metadata schema is
`reliquary/environment/schemas/episode-v1.schema.json`.

`scripts/emit_episode_determinism.py` emits tokenizer-neutral reference and
rejected trajectories. CI regenerates a larger 64-task corpus on Ubuntu 22.04
and 24.04 and requires byte-identical output. Environment manifests are checked
at registry construction, so a changed source file, schema, fixture or manifest
without the corresponding contract update prevents the process from starting.

## Interoperability

- `agentic/adapters/mcp.py` converts public MCP tool-list/tool-call shapes.
- `agentic/adapters/prime_v1.py` exports JSON-native Task/Trace material for a
  pinned Prime Verifiers v1 integration.

Adapters must not bypass Reliquary replay, manifests, signatures or reward
checks. External packages should be pinned by exact commit in the deployment
manifest rather than added to the core runtime dependency set.
