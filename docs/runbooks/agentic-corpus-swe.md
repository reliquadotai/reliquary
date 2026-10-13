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

## Suite episode jobs

Some episode jobs do not run a SWE task in Docker. Their manifest says
`episode.harness: "suite"`, and `episode.env.name` names an episode
environment registered in Reliquary (the first one is
`reliquary_telecom_solo_v1`). The model calls the environment's tools, one call
per turn, until the environment says the episode is over or the turn limit is
reached. These are the same episodes the RL miner runs.

To mine one:

1. Install the environment's wheel, at the version the job pins. For telecom:

   ```bash
   pip install "git+https://github.com/reliquadotai/reliquary-environments@0bfffb4#subdirectory=environments/tool_use/reliquary_telecom_solo"
   ```

2. Run the same command as for any episode job:

   ```bash
   reliquary corpus mine-agentic --validator-url <url> --job-id <job>
   ```

You do not need Docker, verifiers or renderers, and the miner opens no network
port. Before it starts, the miner checks that the environment is registered in
your Reliquary, that its manifest digest is the one the job pins in
`episode.env.version`, and that the installed wheel matches it. If one of these
fails, it stops and says what to install or update.

What the miner submits, and what it drops:

- An episode the environment ends is submitted with stop `agent_completed`;
  one cut by the job's turn limit with stop `max_turns`.
- An episode that goes past the job's token budget is dropped. So is one where
  a turn other than the last ends on `<|endoftext|>`: the validator refuses it.
- Before signing, the miner replays the trajectory exactly as the validator
  will. Only a trajectory whose replay matches is submitted; the others are
  counted as `precheck_refused:<reason>` in the final counts.
