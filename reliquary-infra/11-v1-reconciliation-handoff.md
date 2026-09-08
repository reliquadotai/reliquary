# Romain — V1 infrastructure reconciliation

Date: 2026-09-08. This is a code and qualification handoff, not a production
activation or a claim that the executor/signer hosts have been provisioned.

## Branch and source of truth

- Destination: `integration/reliquary-v1-final`, [draft PR #224](https://github.com/reliquadotai/reliquary/pull/224).
- V1 input: `589fd436b55d2f9c43b40b173f28327ddd56148e`.
- Infra input: `codex/remote-cpu-sandbox-prototype`, `780df1c7cc62bcda0aaf37171644d546c0f49b29`.
- Common ancestor: `c0b01d1a4bf34a89d0074bdd4bb6460b6ebf5cc0`.
- Infra conversation digested: `01a02389-6122-7e92-8822-ef91364f2263`, from the original architecture research through the three-role package and final status correction.

All eleven infra commits are retained by a merge. There were three textual
conflicts in `validator/checkpoint.py` and `validator/service.py`; they were
resolved against the current V1 invariants, with independent read-only reviews.

| Infra commit | Retained contribution |
| --- | --- |
| `ef83500` | Remote CPU executor boundary |
| `3c837ed` | Hardened shadow control edge |
| `0cc4448` | CPU executor packaging and containment |
| `3a0d40d` | Executor build dependency correction |
| `27dd386` | Build validation |
| `a4cf665` | Control playbook lint corrections |
| `ed2aaf4` | Separate semantic signer |
| `c5e820d` | Ephemeral Bittensor home |
| `8330033` | Listener-aware health |
| `2af379e` | Qualification mTLS identity |
| `780df1c` | Inventory resolution from the deployment wrapper |

Romain's five latest V1 commits remain intact: `d10ae29` and `98b04ad`
(pinned, gated EnvScaler corpus), `e582c0d` (V6 deployment documentation),
`8424b83` (profile-derived test fixtures), and `589fd43` (V6-specific CI).
The checkpoint-epoch retirement in `10c2a4d` remains intact. Executor protocol
v2 and signer protocol v1 are separate internal APIs, not economic profile IDs.

## What is integrated

- Trusted control retains expected answers and scoring. The remote CPU agent
  receives bounded code/call arguments over mTLS and runs disposable gVisor/KVM
  sandboxes. Local execution remains the default; remote shadow mode has no
  scoring authority.
- Signer owns only checkpoint signatures, set-weights and serve-axon operations,
  with a separate mTLS trust domain and durable replay journal. There is no
  arbitrary-byte signing API. No real wallet or chain operation was used for
  reconciliation tests.
- V1 canonical checkpoint repository/OID validation, monotonic floor, exact
  retries and the fill/adoption barriers are preserved. Signer calls run off
  the event loop. A local resume directory is never signed or advertised.
- Existing V1 environment adapters, retention, control hardening and historical
  generation profiles are retained. EnvScaler and the external stateful wheel
  remain outside active profiles.
- The three playbooks, immutable artifact builders, role validators and
  snapshot edge are now in the same branch as V1.

## Corrections made while reconciling

1. Worker-capacity wait no longer consumes a candidate's execution allowance.
   A batch-level deadline becomes an infrastructure error, not a false negative
   training label. A genuine per-case execution timeout remains a candidate
   failure.
2. Cancelling an executor HTTP task retains its capacity slot until the actual
   sandbox work finishes; cancelled queued work releases its own slot.
3. Restart can recover an unfinished checkpoint-signature request. Completed
   retries remain cached; pending/uncertain chain writes cannot replay.
4. Both Compose services read the configuration paths actually installed by
   Ansible, rather than looking for missing sibling `.env` files.
5. Builds bind to committed Git source, including staged-change detection and
   exclusion of untracked/ignored files from the Docker context. Evidence
   checksums remain valid after moving an artifact bundle.
6. Infra tests are included in normal CPU CI; the V6 job also exercises the
   new signer/executor seams. Linux CI builds both roles at the PR source SHA,
   verifies a moved bundle and checks that tampered evidence is rejected.

## Validation and reproduction

Validation results are recorded in PR #224 and the delivery message. Historical
August results and artifacts at `780df1c` do not qualify this new source.

```sh
python -m pytest -q --ignore=tests/gpu --ignore=tests/integration/test_grader_e2e.py
```

The dedicated V6 command is maintained in `.github/workflows/validator-tests.yml`.
`Infrastructure packages` builds the two Linux/amd64 images without running any
hostile workload or starting a deployed service. Builds must be repeated for
the exact source revision intended for installation; pin both the resulting
image ID and grader-runtime ID. The grader worker changed relative to V1, so
control/executor runtime identities must agree before selecting remote mode.

## Next steps for Rom

1. Review the merge and its green CPU, V6, determinism and package checks.
2. Confirm the current host inventory and SSH access separately. Operational
   addresses, administrator keys and private inventory remain outside Git.
3. Build or retrieve the exact qualified artifacts, prepare the independent
   mTLS leaves and provision the dedicated executor/signer hosts when available.
4. Qualify the executor with the malicious corpus, real KVM kernel, overload,
   restart and at least 10,000 representative shadow batches with zero
   deterministic divergence. New request/output size bounds need corpus
   qualification; unit tests alone do not establish historical parity.
5. Qualify signer restart/identity/policy with a test hotkey first. Real hotkey
   transfer and chain writes require their explicit activation procedure.
6. Implement and qualify the remote GPU proof boundary around the current
   scheduler, then the wallet-free authoritative validator deployment.
7. Qualify fill-closed mid-window crash/restart and exact GPU end-to-end
   checkpoint publication/adoption. Retain current production as rollback until
   the separate cutover, monitoring and backup gates pass.

The pre-existing V1 backlog remains separate: historical non-epoch episode
schema-3 reads are currently rejected; the retired epoch flag is silently
ignored; the dormant release capability bundle and older reconciliation report
still describe ticketed lanes. The infrastructure merge does not repair or
reactivate that retired market. External Logic/Math/Code releases and dependency
PR #221 are not imported by this merge.

No remote proof worker, GPU-provider automation, production activation, server
purchase, main-branch merge or main-branch push is part of this reconciliation.
