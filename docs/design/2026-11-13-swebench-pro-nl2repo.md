# SWE-bench Pro and NL2Repo-Bench as agentic eval envs

Status: implemented, in review, 2026-11-13 (nothing merged or deployed). Catalog plan:
`~/runbooks/eval-catalog/PLAN.md` B4 and B7.

| Repo | Branch | What |
| --- | --- | --- |
| reliquary-environments | `feat/reliquary-swebench-pro` | `reliquary-swebench-pro` 0.1.0 (642 tasks, image lock, gold check, gate) |
| reliquary-environments | `feat/reliquary-nl2repo` | `reliquary-nl2repo` 0.1.0 (104 tasks, image lock, gold check, gate) |
| Catalyst | `feat/swepro-nl2repo-evals` | both in `BRIDGED_EVAL_PACKAGES`, this document |
| reliquary-platform | `feat/swepro-nl2repo-gateway-groups` | image groups `swepro` and `nl2repo` (migration 0064), sets and pins |

Both are served like `reliquary-terminal` and `reliquary-deepswe`: a verifiers v0.3.1 taskset
behind the sandbox's generic bridge, signed and eval only, graded in a pristine box of the task's
image, no network in any box. A Catalyst set freezes references to the gated `eval` split
(`reliquary eval build-set --source <package> --split eval --bridge-file bridge.json`).

## 1. SWE-bench Pro (`reliquary-swebench-pro`)

- Current public set: **V2**, 642 tasks, the default config of `ScaleAI/SWE-bench_Pro` since
  2026-09-22 (revision `2d52cb3d`; `v1` = the original 731, `hard` = 51). Tasks in Harbor format at
  `scaleapi/SWE-bench_Pro-os@66f92766` (`v2/tasks`, MIT), images
  `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` (642 digests, 1,051 GB compressed, 544 GB of
  distinct layers).
- Grading follows Scale's locked protocol: the agent's whole work tree against the base
  (committed or not), replayed with `git apply` / `--3way` / `patch --fuzz=3` in a fresh box,
  upstream's `test.sh` unchanged, verdict = every fail-to-pass and pass-to-pass test PASSED. The
  verdict is recomputed outside the box from the run script's raw logs with the task's own
  vendored parser.
- Upstream's verifier runs with network (`network_mode = "public"`); ours does not. Gate: gold 1
  and no-op 0 offline within 700 s (the 810 s grading step), per task, on gVisor.

Gold check (2026-11-13, Targon CPU VMs, gVisor, boxes at 1 CPU / 4 GB): **576 of 642 served**.
Left out: 32 need network at grading time (vuls 15, flipt 9, openlibrary 2, navidrome 2, NodeBB,
element-web, teleport, tutanota: Go module fetches from proxy.golang.org, `npm install`, host
lookups), 24 grade over 700 s on one CPU (ansible 8, vuls 7, protonmail 4, teleport 3, flipt 2),
4 whose gold fails offline otherwise, 4 whose gold breaks a pass-to-pass test offline, 2 flaky
(element-web). No no-op graded 1; the 13 conformance cases graded as declared.

Gateway set: `reliquary-swebench-pro-eval-r0-n576-k100-s7` (100 of the 576, seed 7; the whole
split's images, ~1 TB compressed, do not fit one gateway). Identity
`reliquary-swebench-pro==0.1.0+g9612511bc54064f7.vfb2`, image lock
`48b45ac6ea376d475577d7df1231fdb22c6e44ba670c19dca8c7a9d1ad65c100`, options
`cf93c6cea43146ba102828a146e9b2cf0322970f2ec14da628a321cbac8fd722`; its 100 images weigh ~150 GB
compressed.

## 2. NL2Repo-Bench (`reliquary-nl2repo`)

- 104 tasks at `multimodal-art-projection/NL2RepoBench@781a1da1` (**no upstream license**: the
  task files are pinned by sha256 and fetched at wheel build, never committed to the public env
  repo; the wheel's distribution needs that question settled). One image per task
  (`ghcr.io/multimodal-art-projection/nl2repobench/<task>:1.0`, ~28 GB of distinct layers).
- The agent builds a whole Python project from `start.md` in an emptied `/workspace`; reward =
  min(passed / upstream's `test_case_count`, 1) from JUnit reports of upstream's test commands,
  run offline in a pristine box over a sanitized copy of the workspace (no test hooks, no
  packaging files, no hidden-test paths). The images leaked the reference implementation (21
  local installs, develop links, source copies): removed in both boxes before anything runs.
- Boxes: 4 CPUs, 8 GB, 10 GiB of disk (Prime's adaptation declares 20 GB, above what a gateway
  and Catalyst's episode budgets admit; same cap as DeepSWE).
- No gold patch: the reference source is recovered from each image's build layers. Gold check:
  **53 of 104 served** (reference 1, empty workspace 0, offline, within 700 s). Left out: 20 images
  with no recoverable reference, 19 references scoring under 1 offline (12 at 0.977-0.998:
  root-run quirks or count drift; 7 lower), 5 harness/reference problems, 2 environment ones, 4
  where an empty workspace scores above 0 (pytz: 0.987, the package itself is installed), 1
  excluded (pyautogui). A reference >= 0.95 rule would serve 64.

Set: `reliquary-nl2repo-eval-r0-n53` (whole gated split). Identity
`reliquary-nl2repo==0.1.0+g29930100768f36e3.vfb2` as frozen; the env branch's head (10 GiB disk
cap) is `+g4b4f047421550a4f.vfb2`, so the set is re-frozen after the env merge. Image lock
`328e2e5db7be1d062af505b0cc4ca02f6b817e8307f247fb060fd25f8bf6d6af`, same options digest.

## 3. Network

No box gets network here. Tasks that need it at grading time are left out by the gate and listed
above (and in the packages' gold-check results); serving them is for the controlled egress design
(`docs/design/2026-11-13-sandbox-network-compose.md`, branch `feat/sandbox-network-compose`).

## 4. Images

Mirrored by digest into our Hippius registry (bucket `reliquary-task-images`, repo path and
manifest digest kept, `~/runbooks/image-mirror/mirror.sh`). Gateways still pull from GHCR, as
every gateway group does today; pulling from the mirror is a gateway change of its own.

## 5. Remaining before an order can run

1. Review and push the branches; merge the env packages, then pin `ENV_COMMIT` and the wheel
   identities on the platform branch from the merged commit (the pins here are from the local
   branch heads).
2. Build the kit: the two wheels in `wheels/`, `groups/swepro/{images,index}.txt` and
   `groups/nl2repo/{images,index}.txt` from the frozen sets.
3. Publish both sets on the platform and run one test order per set on a rented gateway.
4. Miners install the env wheels (task text only, no images), as for `reliquary-terminal`.
