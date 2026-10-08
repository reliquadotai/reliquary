# Agentic corpus execution

An episode job is defined by its immutable manifest and served task contract.
The contract pins the checkpoint, renderer, environment artifacts, token
budgets, and proof rules used by the producer and verifier.

The native `corpus mine-agentic` command selects one explicit job, checks its
served contract and checkpoint fingerprint, and produces signed trajectories.
Its generation runtime captures the actual decode activations and builds a
proof for each assistant span. Token and span bounds are validated before
admission; the verifier checks the declared checkpoint and proof contract.

Before it starts an episode the miner reads `GET /corpus/jobs/<job>/open`
and passes over the prompts of its walk that have no slot left, so an episode
is not spent on a `prompt_full` refusal; see
[corpus-task-launch §4.1.1](corpus-task-launch.md#411-which-prompts-still-have-a-slot-open).

The worker requires the selected environment artifacts and a qualified
inference runtime. The miner uses its own hotkey; controller, storage, and
cold-wallet credentials are not part of the miner interface.

Completion, proof verification, grading, delivery, and incentive settlement
are separate outcomes. Confirm each required outcome from its native record.
A zero-cap task carries no task incentive.

Service shutdown awaits owned background work and finalizers. See the
[split-process runtime reference](../design/2026-10-02-corpus-split-processes.md)
for process roles and shutdown grace configuration.
