# Native Affine held-out evaluation

This lane runs one original, authority-signed MiMo `role: evaluate` job in a
bounded allocation. It keeps Affine's native source, checkpoint descriptor,
held-out suite, common random draws, final-channel grader and original expiry.
It does not mine, train, upload to Affine, sign a native receipt or write chain
state. Independent SN81 benchmarks use [AFFINE_EVALUATION.md](AFFINE_EVALUATION.md)
and have a different sampler and comparison identity.

The reviewed report contract is `mimo-heldout-evaluation-v1` at upstream revision
`ebf4f5f67dd34e3a13c6c275e1b9e3fd3c9766cd`. Native execution still performs the
upstream release-policy, complete source-closure, runtime and job admission.
An authenticated job does not qualify Reliquary hardware or establish numerical
agreement with another host.

## Frozen intake

The owner-private preparation JSON has exactly these fields:

| Field | Meaning |
| --- | --- |
| `schema` | `affine-native-evaluation-intake/v1` |
| `assurance` | `native-mimo-evaluation`, or `fixture` for synthetic checks |
| `authority` | Independently trusted Affine Ed25519 public key |
| `jobEnvelope` | Original signed evaluate job, including its signed opening |
| `sourceDirectory` | Reviewed source export, without Git state or bytecode |
| `sourceRevision` | Reviewed report-contract revision above |
| `checkpointDirectory` | Dedicated local export matching every signed file hash |
| `limits` | Bounded allocation limits below |
| `review` | Exactly `sourceApproved`, `checkpointApproved`, `heldoutApproved`, all `true` |

All paths are absolute, owner-private and outside repositories. Preparation
authenticates both signatures, refuses mining/training/upload capabilities,
checks the original suite and its separation from mining, and freezes the
entire supplied source inventory. The revision identifies the reviewed report
contract; the original job's exact file inventory identifies executed code.
The complete installed `subnet/**/*.py` map
must equal the signed job's map. Checkpoint files must be safe local model files
whose bytes match the signed descriptor. The native evaluator file must match
the reviewed contract; fixture source is never executed.

`limits` contains `maxSeconds`, `finalizationReserveSeconds`, `maxLogBytes`,
`maxOutputBytes` and `maxWorkspaceBytes`. The native report is bounded to 12 MB.
The exposed integer `nativeJobExpiresAt` floors the signed timestamp; it can
shorten an allocation fractionally but never extends the original job. A fresh
signed job must come from Affine when that original lifetime is insufficient.

Preparation writes `job.private.json`, the unchanged canonical signed payload in
`native-job.private.json`, and a frozen `source/` export. Five immutable pins
bind the job, source, runtime, signed inputs and checkpoint inventory. The
preparation result leaves native admission, hardware qualification and activation
unverified. It performs no provider request or billable execution.

## Allocation and result delivery

The host mounts the prepared bundle and checkpoint read-only, with a separate
private writable workspace. Native execution requires CPython 3.12, an
unprivileged Linux container with networking disabled, H200/SM90, and the exact
package versions signed in the job. The host supplies a lease assertion and
owns the independent container deadline, memory limit and enforced disk quota.

The child invokes the unchanged `subnet.backend_jobs.execute` evaluate path
with the local checkpoint, `allow_public_mimo=True` and an allocation guard.
The upstream enabled signed release policy remains authoritative. The wrapper
does not turn that local flag into release authority. This intake supports
bundled native task data. Openings requiring separate task-asset capabilities
are refused before allocation until their frozen intake is qualified. It never
substitutes another sampler or silently changes
the original suite, horizon or runtime.

The wrapper bounds time, log growth and observed workspace size, revokes its
local guard on cancellation or lease loss, and stops the child process group.
Observed size checks complement the host's disk quota. An interrupted attempt
preserves its journal and requires reconciliation and a fresh workspace; it is
not silently reexecuted.

The report checker verifies job/source/runtime/checkpoint identities, the frozen
dataset identity, ordered task/repeat coverage, token bounds and first-EOS
semantics. It preserves correct, completed-wrong, unresolved and infrastructure
outcomes separately. Infrastructure failures require `status: error` and
`mean_reward: null`; they never become invented wrong answers. A valid report
may therefore describe an unsuccessful evaluation.

The owner-private ZIP contains exactly `native-report.json` and
`evaluation-result.json`. The result binds its byte hashes and keeps native
parity, GPU qualification, operator acceptance and chain transactions false.
Report structure is checked; the checker does not recompute model outputs or
independently establish their truth. Platform delivery must still verify the
approved host receipt, owned stored bytes and final account settlement.

The core module exposes `prepare`, `load_bundle`, `native_mounts`, `run`,
`verify_report`, `verify` and `export`. Fixture execution and native execution
cannot be interchanged. `run(max_seconds=...)` can shorten its allocation; it
cannot extend prepared limits or the original signed job lifetime.

The operator CLI is `python -m reliquary.integrations.affine_cli native-evaluation`.
Use `prepare --config "$INTAKE" --out "$BUNDLE"`, `check --bundle "$BUNDLE"`,
then `run --bundle "$BUNDLE" --workspace "$WORKSPACE" --checkpoint "$CHECKPOINT"`.
`verify` checks the saved report; `export --out "$ARCHIVE"` writes a new ZIP.
Run, verify and export require the same bundle and workspace. `--fixture` is
explicit for synthetic runs. The connected platform adapter additionally owns
the broker lease, host deadline, receipt signing, delivery and settlement.

```sh
python -m pytest --noconftest tests/unit/test_affine_native_evaluation.py -q
```

These CPU checks establish signature, isolation, deadline, failure-report and
delivery contracts. Actual model execution, paired native result comparison,
GPU image qualification, customer funding and production activation remain
separate acceptance steps.

An optional offline oracle check accepts `AFFINE_NATIVE_EVALUATOR_FILE`, the
reviewed original `subnet/mimo_evaluation.py` saved outside Git. It checks that
file's pinned hash and executes its suite admission, evaluator and result
validator with a scripted CPU runtime and grader. Its correct/wrong/capped/error
rows must also pass this bridge's independent report checker. This checks report
semantics without claiming real sampling mathematics or model execution.
The dedicated native contract CI job downloads that exact pinned revision and
runs the oracle alongside fixed32 coverage, partial-report rejection,
cancellation, lost-lease and interrupted-attempt retry checks. It uses no model
weights or GPU runtime.
