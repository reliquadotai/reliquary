# Reliquary V1 final reconciliation — 2026-09-05

This candidate combines V1 checkpoint/market/durability work with the dormant
embedded environment suite, the pinned external-wheel adapter, and missing
security/control/retention fixes. It preserves historical generation contracts
and immutable checkpoint identities. It does not activate a profile or deploy
a service. Keep the PR draft pending hardware qualification.

## Scope and graph evidence

- Final branch: `integration/reliquary-v1-final`.
- Target **MAIN**: `4544394feecc312ebbfb30f7e0815fe62d034b33`.
- Selected base **V1**, PR #220: `71672e14ac2bef54e7be2177e9271c7cd2295c8e`.
- **B**: `68f0047b0bae8159575252e19ea49bec87b4cc8f`.
- **A**: `93c0e911269311f3bae7a25546e5327b7ccbca9d`.
- **C**, older main: `c0b01d1a4bf34a89d0074bdd4bb6460b6ebf5cc0`.

The final branch did not exist locally or remotely. It was created from V1;
merging MAIN reported already up to date. MAIN is an exact ancestor, including
the recent #217 and #219 squash merges. No main-only change was omitted.

Three independent read-only audits covered PRs, branches/worktrees, and
environments. One editor performed all integration. Fetch covered origin
branches, tags and all PR heads, including #216 whose origin branch is absent.
PR heads use the separate `refs/reconciliation-pr/*` namespace to avoid old
local PR-ref namespace collisions.

For every local/origin branch, the audit examined reachable commits with
`git rev-list --since-as-filter=2026-08-29T00:00:00Z`, rather than filtering only
tip dates. It used `merge-base --is-ancestor`, `merge-base`, `git cherry`,
`range-diff`, and real file/test diffs to resolve cherry-picked equivalents.
There were 20 original recent refs representing 15 branch names, plus
`origin/HEAD`; nine open PRs also cover older dependencies. The complete
190-ref snapshot, including SHAs, dates and merge bases, is in
`docs/evidence/reliquary-v1-final/branch-inventory.json`. Another 137 old refs
are exact ancestors of MAIN. Old unrelated branches were not imported merely
because they exist.

All 81 registered worktrees were inspected: 34 existed and 47 had absent/stale
paths. No worktree was pruned. Seven pre-existing dirty worktrees, their
personal reports, lockfiles, scratch files and local ignore edits were left
untouched. All six `work/reliquary-v1-*` worktrees were clean. The recent
detached `dde1` checkout is identical to MAIN. No SSH or production checkout
was used.

## Exhaustive PR and branch decision matrix

`L+R` means identical local and origin refs; `L` or `R` means only that scope.
Initial coverage is relative to selected V1. Test groups below map each row
to actual component checks. `included / superseded` means the valid behavior
is retained while the older implementation is replaced; it is not a blind
merge of the source history.

| PR / branch | Source SHA; base or fork | Initial coverage and actual purpose/files | Final decision and destination | Tests |
|---|---|---|---|---|
| #220 `design/reliquary-v1-reconciliation` L+R | `71672e14ac2bef54e7be2177e9271c7cd2295c8e`; MAIN | Selected base: generic ABI, release contract, epoch market, immutable checkpoints, durable queues/staging/cooldown | **included** in full; final branch starts here | V1, D |
| #198 `checkpoint-epoch-scheduling-prototype` L+R | `6979f7e6e6210916f01dd54e3e56cc1d2915f223`; PR base main, merge-base with V1 `942cc80586c7e7debaaf9d78656e3cfb3cf76ad0` | Already equivalent: epoch generation plans and trainer aggregation | **included / superseded** by V1 market ports; four patch-equivalent commits and one adapted without losing production ranking | V1, P |
| #213 `codex/hf-checkpoint-retention` L+R | `8e6d4375bc35106e77d853bea86b4df65f57f80c`; main/C | Missing: manual finished-run archival and HF quota guard, publisher invariants, archive script | **included / superseded** in `adb2df8de3b9ce67d95dcabbe84626e9c1bfce85` + `50c530b29ccd58fbce8dca81d7a3db884456e31f`; final manual design replaces earlier automatic squash | H |
| #214 `design/macro-window-dapo-batch` R | `90cf6a1677d86e1a23ddefa61a49d4ac9ab4846e`; main, exact V1 ancestor | Included history: fill closure, admission/proof streaming, telemetry and experiments | **included / superseded** by exact ancestry and the later ticketed V1 market; no reintroduction of older payment policy | V1, D, F |
| #215 `codex/control-plane-cooldown-hardening-draft` L+R | `1357b9646852b05e75220e95d15a9d613a3f60e7`; main/C | Partial: bounded miner state already present; missing retry, verdict feed, probes and query fixes in miner/protocol/server | **included / superseded** by `d666ce115907b7d980784d49907b268c28769b74` plus V1 durable recovery; per-feature decisions below | C, D |
| #216 `chore/deprecation-warnings-cleanup`, PR ref | `323697665697086acf3cdcbddcfd191a18138117`; main/C | Missing: warnings as errors and LR reconstruction warning cleanup | **included** in `eb3e040022bc6ce0e7d4ba3d8ef9d81da1574f94` + `d205562a4b65f877a34da1f6f3d9bdcc28f4385f`; preserve newer flat LR | S, CPU |
| #217 `codex/adaptive-auction-fairness` L+R, already merged | `625d8006b1633ed025b04991bc7a5f5276a20929`; C | Already equivalent: adaptive early close and production throughput tiebreak | **included** through squash `84dcc571f41c755fc00a0d9bc1bd3fb5f6e4a7f6`, ancestor of MAIN/V1 | F, S |
| #218 `design/fill-closed-v6` R, already closed | `9fa42ea006084d27184f58a70773237e957d1087`; common V1 ancestor `1424e6d6cb986940e30d36c6effb90d80172c95b` | Partially included; only divergent feature is R42 stale-backlog skipping | **included** through common ancestor; **excluded** R42 commits `96df9592b97d3d33d2cae4a3f27ea2e0caa825b6` and `9fa42ea006084d27184f58a70773237e957d1087`: unsafe cursor advance, detailed below | D, F |
| #219 `fix/lr-flat-schedule` R, already merged | `5b0ced28bccb45d7078a13de7541b4ec0ab38bb0`; `84dcc571f41c755fc00a0d9bc1bd3fb5f6e4a7f6` | Already equivalent: flat LR schedule and restart reconstruction | **included** through MAIN squash `4544394feecc312ebbfb30f7e0815fe62d034b33` | S |
| #221 `dependabot/pip/transformers-5.10.1` R | `27f6f0e5f248383c1bdd0f40615fb6cc5f647562`; MAIN | Missing isolated dependency bump in `pyproject.toml`, no qualified tokenizer/runtime migration | **excluded / deferred**; retain `transformers==5.9.0`; leave PR open for separate token/consensus/GPU qualification | P, CPU |
| #222 `standalone-environment-integration-v2` L+R | `a76bca595f45d73c71dade086af55dc274f7526b`; PR base suite at `656c86227b7cd5a96676c3e88e31412ababf464c` | Missing secure artifact loader, registry adapter, tests and repository plan | **included** via `da0ab841f46325aed6004429654901291e49b134`, `aac785d810c97f21e480e7374811ffdb44a9c18d`, hardened in `d8d5ca1af2bf47d9eef4994c140603d45eaa4500` | E, X |
| #223 `fix/v5-seed-auth-false-positive` L+R | `4114ed1f60a015a54d9d3875c7d3189dca8a6ccc`; MAIN | Missing v5-only shadow treatment for all-token authenticity false positives | **included** in `c4fcc9969c647081c5f7e27c3ed2e1387e0282b8`; exact inverse-CDF tail regression retained | S |
| `reliquary-environment-suite-v1` L+R | `656c86227b7cd5a96676c3e88e31412ababf464c`; C | Missing embedded logic/verifiable/episode environments, rollout masks, admission replay, targets and qualification tools | **included / reconciled** as final net snapshot `ccc6fcbd900422f11278a6f34aabe387dfe5b85e`; all 27 source commits assessed, superseded intermediate fixes collapsed | E, P |
| `work/reliquary-v1-admission` L | `34ee05191cc6aad86013ca68272dd7157dfbf188`; B | Already equivalent: admission identity, guards and HTTP boundary | **included / superseded** by three existing V1 ports; range-diff differences are context/imports/comments | V1, C |
| `work/reliquary-v1-contracts` L | `df4a22a48b194c87a69be57fb32897df873ee448`; B | Already equivalent: V1 release contract | **included** as `64dd77e3b3b5680e5048ecdb02cd9d8c7efa919a`, subsequently hardened | V1 |
| `work/reliquary-v1-cooldown` L | `4669c03e9b0776ad86afe95a5b4055e8751e937a`; A | Already equivalent: durable cooldown and checkpoint recovery | **included / superseded** by three `git cherry -` equivalents and later strict V1 recovery | D, C |
| `work/reliquary-v1-durability` L | `8a056f74446a31c7d2c64d07485c93c739ce10db`; B | Already equivalent: queue/writer, assembler, rotation, pacing and seal seam | **included / superseded** as `43f46402f97a43a92f3e6a40e0e9cdf7f4036dc2`, then hardened | D, F |
| `work/reliquary-v1-market` L | `57af2185274d7f4307c9c0a4de33ea1f9f4f343e`; B | Already equivalent: epoch intent/ticket market and ranking | **included** through six patch-equivalent ports; retain production and epoch ranking decisions | V1, P |
| `work/reliquary-v1-ticketed-streaming` L | `33ef7afd2a492055fd3c11fae759df326e936cc2`; A | Already equivalent: ticketed proof staging | **included / superseded** through two equivalent ports and later persistence hardening | V1, D |
| `origin/main`, alias `origin/HEAD` | `4544394feecc312ebbfb30f7e0815fe62d034b33`; target | Recent main fixes already exact ancestors | **included** in full. Local `main` at C is also an ancestor and remains untouched | CPU |
| #212 `feat/auction-early-close` R, older dependency | `7cef0345ff2d616a7295fc9d1f6df51f20debfb4`; historical main | Already included early-close state machine | **included** by exact main ancestry | F |
| `codex/remote-cpu-sandbox-prototype` L+R, older branch examined | `780df1c7cc62bcda0aaf37171644d546c0f49b29`; historical main | Separate remote CPU executor, signer/mTLS and deployment-role prototype | **excluded**, last activity Aug 28, no open PR or dependency from recent work; no deployment role change in this reconciliation | Scope/graph audit |
| Remaining historical inherited refs | All 137 exact SHAs in `branch-inventory.json`; MAIN | Old changes already ancestors | **included** by ancestry; no separate activation | CPU |

## Conflict and replacement decisions

### PR #215 control plane

The existing V1 `/miner-state` response already supplies bounded bitmaps,
per-environment admission budgets and ETags. Keep it. Add frozen membership
without rebuilding a history-sized fallback; a regression refuses iteration
of the legacy list when membership is available.

Add bounded control retries/jitter, private numeric `Retry-After`, exact query
parity, and a sequenced verdict feed. A per-process `stream_id` detects restart
even when new counters have already caught up. The miner starts polling after
its first submitted batch and cancels/awaits the task before closing HTTP.
Legacy verdict and submission response shapes remain unchanged. Epoch retry
defaults remain unchanged.

Local `/livez` and `/readyz` replace the old rich-health cache proposal.
Readiness includes registration, process, degraded prompt source, complete
cooldown and required proof-scheduler state, with no request-path HF/R2/drand
lookup. Existing diagnostics remain available. Unqualified rich-health
latency/backlog thresholds and a second precompression cache are excluded.

The old checkpoint/bootstrap implementation is superseded by V1 immutable
repository/OID, monotonic activation and persisted recovery. Bootstrap's
synchronous download precedes miner concurrency. Old delta/background cooldown
deletion and partial recovery are incompatible with V1 exact durable snapshots
and journals; preserve V1. Its off-loop JSON I/O and backlog telemetry already
cover the valid parts of those changes.

### Retention and checkpoint safety

Only the final manual finished-run workflow from #213 is carried. No automatic
runtime history squash is introduced. Unknown quota usage fails closed.
Archival verifies actual R2 bytes, not only metadata. Every selected source HF
file must exist in the archive with the correct size and digest. Ambiguous
checkpoint numbers, unknown sizes, retaining tags, mutable source OIDs and
changed head identities are rejected before finalization. Existing immutable
publisher identities and journal semantics remain authoritative. The tool was
tested with mocks; no remote repository history was squashed during this task.

### Environment and payload reconciliation

Keep both the generic ABI (`environment.abi`) and concrete registry/runtime
adapters (`environment.registry`). The embedded suite remains available as
historical/dormant functionality. New environment source lives only in the
external repository. Existing active profiles are unchanged; the three new
development profiles are explicit opt-ins and do not become the default.

Resolve the independent uses of payload schema 3 without reinterpreting old
archives: historical epoch-only v3 and historical episode-only v3 are accepted
only when unambiguous. New extended payloads use v4; combined epoch+episode
payloads use v5. Legacy v1/v2 remain readable; ambiguous v3 is rejected. Preserve
private assistant spans, masks, per-environment targets, and validate all lanes
before mutating trainer state. Keep V1 epoch bounds alongside episode targets.
Early shared admission guards reject interaction-mode mismatches before reward
work. Auction dead-batcher sealing retains V1 fairness behavior.

The old R42 skip mode advances a consumed cursor using a default depth of 16
and HEAD-only object existence, before decoding window/checkpoint identity or
receiving training acknowledgement. That violates strict V1 journal/ack
recovery. Its two divergent commits contain only that feature and related
tests/docs; no independent valid fix is omitted.

## Historical compatibility and external release

`historical-contracts.json` compares every pre-existing profile field against
both MAIN and V1. Added optional fields are null on historical profiles. The
default remains `qwen35-2b-auction-v2`. All historical contract bytes and these
SHA-256 values match:

| Contract | SHA-256 |
|---|---|
| v2 | `d763f85b1366ac1b25d0de1251899d6a8aceee11f5235f18817176a039d15dd1` |
| v3 | `3d0b7013d980b565065e19627fbaeb592c96a1604fb7bd3af8e31d15d04c87e8` |
| v4 | `e3098b00582d395fc7176ff835c988588e29a530343942c081781f1bab65de91` |
| v5 | `19e98f5a3ddac1980efe66fd80db1ec0f8db87a5e60934efd5d0e8985435eadd` |
| v6 fill | `1696eef2a8ff52284842f2253d6f699b50bc657dc93b20fc61a257db7d449385` |
| Historical epoch fixture | `a3fa5ef505987dfaa8110ea8b4d8aeb620199b64d6f42ddd575ca763c74ce04c` |

`tests/fixtures/determinism_corpus.jsonl` remains byte-identical (Git blob
`9819e06c24a69ba26b527e824bd3189f670ea7ac`). Embedded environment source-bound
manifests/goldens are preserved; no blanket formatter rewrote their sources.
The intentional #223 v5 authenticity-policy correction changes enforcement
behavior, not signed generation-contract contents.

External source: [reliquadotai/reliquary-environments](https://github.com/reliquadotai/reliquary-environments).
PR #1 is verified merged at `73285b192f0359ed75635d6d4ff1694ab3a1b106`.
Release [v0.1.0a1](https://github.com/reliquadotai/reliquary-environments/releases/tag/v0.1.0a1)
was built from `cdb998b355ee41355823f2a15cdd21ab19c4718a`.
Wheel SHA-256 is `f4d5480e57e66265faa78c53e36fa8ab781afe0ae907d7dc5749d2b0f9344155`.
Exact Prime-RL v0.9.0, Verifiers, transitive Git pins and model revision, plus
reproduction commands, are in `standalone-environment-qualification.md`.

The published wheel was downloaded and hashed before import in a separate
Python 3.12 environment with the pinned Verifiers revision. The core loader
rechecks every bound file on each create, executes verified source snapshots,
ignores shadow/bytecode imports, and rejects unverified preloaded modules.
Three reference tasks matched exact task IDs/state hashes and scored 1; wrong
answers scored 0. Evidence: `docs/evidence/reliquary-v1-final/external-live.json`.
`reliquary_stateful_tools_v2` belongs to no profile. CPU scripted replay is not
GPU, learned-policy, throughput or production evidence.

## Validation

Final CPU result: **3,007 passed, 8 skipped in 286.84s**. All eight skips are
existing optional EnvScaler checks requiring its separately pinned corpus; no
additional test path was excluded. Runtime: dedicated CPython 3.12.13,
Torch 2.7.0, Transformers 5.9.0. The final run uses exactly the two exclusions
from `.github/workflows/validator-tests.yml`; a short local temporary path
avoids macOS AF_UNIX pathname limits without excluding socket tests:

```sh
python -m pytest -q --basetemp=/tmp/rqv1-cpu \
  --ignore=tests/gpu --ignore=tests/integration/test_grader_e2e.py -ra
git diff --check
git diff --name-only --diff-filter=ACMR origin/main...HEAD -- '*.py' \
  | xargs ruff check --isolated --select E4,E7,E9,F
```

Ruff correctness rules E4/E7/E9/F cover all 198 changed Python files; the final
result is recorded in the evidence. The project has no repository Ruff rule configuration. Warnings
remain errors, apart from narrowly identified upstream compatibility warnings.

The first full pass exposed five stale test-double fields after reconciliation
and two macOS socket-path failures. `928f99b6b1a2a57dc0639e042021db677963180c`
supplies the actual target/membership invariants to those test doubles; no
production guard was relaxed. The final suite reruns all CPU tests.

| Group | Component checks and targeted evidence |
|---|---|
| V1 | `test_release_contract`, `test_environment_abi`, `test_checkpoint_epoch*`, `test_epoch_proof_staging`, `test_epoch_shadow_planner`; included in full CPU |
| D | `test_cooldown*`, `test_training_payload_queue`, `test_training_payload_writer`, `test_training_recovery_replay`, `test_trainer_train_runner`, checkpoint pull/identity; included in full CPU |
| F | `test_fill_close_and_emit`, `test_fill_closed_profile`, `test_v6_emission`, `test_v6_seal_seam`, auction early close, throughput tiebreak, collection deadline, pick pacing; included in full CPU |
| S | All-token auth, forced-seed constants, GRPO batcher and LR restart: 168 passed in 20.42s; final full suite also covers these |
| H | Archive finalization, HF storage guard and publisher: 46 passed in 4.57s; independent read-only mock audit rejects missing shard, early/late retaining tag and ambiguous number before destructive calls |
| C | Control, verdicts, state cache, bounded miner state, submitter, engine and external artifact: 94 passed in 5.86s; final added probes/membership/cursor regressions: 12 passed in 6.98s |
| E | Environment/unit/profile/ABI/replay/qualification group: 202 passed, 8 optional skips in 14.19s; external loader targeted group: 14 passed in 4.50s; all included in CPU |
| P | Codec, trainer aggregation and episode integration seams: 53 passed in 3.79s; historical token/profile/epoch fixtures also included in CPU |
| X | Real downloaded release wheel: 3/3 golden replays and three wrong-answer refusals; exact Verifiers Git revision checked; no GPU |

## Remaining gates and PR retirement

Keep this candidate draft. Qualify the pinned external Prime-RL path on an
isolated supported two-GPU CUDA host (training plus inference; H100/H200
capacity), covering reward variance, multiple optimizer steps, checkpoint
publication/resume, frozen held-out evaluation and the complete
miner/admission/GRAIL/trainer path. The stopped Nebius/Verda hosts were not
contacted. Historical embedded H200 results do not qualify this wheel or this
branch. Broader 16-lane ticket-only epoch activation also retains the ownership,
implementation and crash-recovery gates in
`checkpoint-epoch-proof-streaming-gate.md`.

After the final branch is pushed, its single draft PR exists, and CPU checks
are green, comment with the replacement and close only #198, #213, #214, #215,
#216, #220, #222 and #223. Leave #221 open. No remote branch is deleted; no PR
is merged and no main push or production activation is performed.
