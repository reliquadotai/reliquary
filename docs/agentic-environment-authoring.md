# Reliquary Episode v1 authoring

Reliquary Episode v1 is the single multi-turn format for every new native tool
environment. The consensus object is Reliquary-owned; MCP and Prime Verifiers
are interoperability boundaries, not reward or replay authorities.

## The three environments shipped in the first suite

1. `reliquary_stateful_tools_v1` — SQLite-backed customer, order, refund and
   support workflows. This is the flagship and first training candidate.
2. `reliquary_retrieval_tools_v1` — frozen search/open/cite workflows. This is
   the cheapest transfer-oriented second lane.
3. `reliquary_workspace_tools_v1` — isolated one-file repair with deterministic
   tests and final-state checks. This is the higher-cost curriculum lane.

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

Then:

1. Add the environment package to `agentic/envs` and `BUILTIN_EPISODE_ENVIRONMENTS`.
2. Add one immutable `EnvironmentSpec` with `interaction_mode="episode"`.
3. Commit a canonical manifest and bind its SHA-256 in a new opt-in protocol
   profile. Never change an activated v1 generator or verifier in place.
4. Add a reference-trajectory test and run the CPU qualification command.
5. Run model shadow collection on a GPU host before allowing optimizer steps.

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

## Interoperability

- `agentic/adapters/mcp.py` converts public MCP tool-list/tool-call shapes.
- `agentic/adapters/prime_v1.py` exports JSON-native Task/Trace material for a
  pinned Prime Verifiers v1 integration.

Adapters must not bypass Reliquary replay, manifests, signatures or reward
checks. External packages should be pinned by exact commit in the deployment
manifest rather than added to the core runtime dependency set.
