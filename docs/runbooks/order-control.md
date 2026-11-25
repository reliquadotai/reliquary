# Order control: build and run

The GPU-less control serving every order job, evaluations (`${RELIQUARY_ADMIN_TASK_PREFIX}eval-`) and
generation orders on any model (`${RELIQUARY_ADMIN_TASK_PREFIX}gen-`). Design:
`docs/design/2026-10-01-any-model-datasets-design.md`; rulings: the "Any-model datasets" section of
`docs/design/2026-10-01-evaluation-orders-rulings.md`.

## Image

`docker/Dockerfile.order-control` layers on the validator image, as the production corpus image does: the
reviewed wheels of `reliquary-logic`, `reliquary-dapo-math`, `reliquary-instruction-following` and
`reliquary-code` (from reliquary-environments, at the pins the catalog digests name) are installed from a
build context named `envs`. The build then runs `reliquary corpus order-control-check`, which fails unless:

- `bittensor_drand` imports (it comes with `bittensor`; without it every record of a sampled job is audited,
  which multiplies executor load);
- `huggingface_hub` and `transformers` import (model files and tokenizers are read on CPU);
- each order source's package is installed, its artifact verifies, and its digest is the catalog's;
- every env a training order may name (`reliquary.admin.training.trainable_environments`: an env of a
  Gumbel-max profile with live RL rows in `reliquary/env_catalog.py`) is built in or installed at the version
  and artifact its registry spec pins (`rl_environments` in the report). `reliquary rl supervise` runs its
  controllers from this image, and a controller whose order names an env without its package crashes at load.

RL training envs beyond the four wheels are installed from reliquary-environments by `pip install
git+...@<commit>#subdirectory=...`: `reliquary-hard-math`, `reliquary-science`, `reliquary-competitive-code` and
`reliquary-general` (0.1.0a1) at `ENVS`, and `reliquary-telecom-solo` 0.1.0a1 (`reliquary_telecom_solo_v1`, the
one the profiles name) at `TELECOM_ENVS`. Each artifact pins the digest of every file and the version, so a wrong
commit fails the build check rather than the controller. `reliquary_telecom_solo_v2` is 0.2.0a1 of the same
distribution: an image carries one of the two. To move a pin, find the commit whose `artifact.json` hashes
(canonical JSON, sha256) to the spec's `environment_manifest_sha256` in `reliquary/environment/registry.py`.

What these envs need at run time:

| Env | Needs |
|---|---|
| `reliquary_hard_math_v1`, `reliquary_telecom_solo_v1` | nothing: the corpus ships in the package |
| `reliquary_science_v1` | Hub: `PrimeIntellect/INTELLECT-3-RL` at a pinned revision, checked by digest |
| `reliquary_competitive_code_v1` | Hub: `ReliquaryForge/competitive-code-curated` (public) at a pinned revision; programs run in the gVisor grading service, as `reliquary_code_v1`'s |
| `reliquary_general_v1` | Hub: `ReliquaryForge/general-prompts-curated@02b3ef7a` (public), or a local copy at `RELIQUARY_GENERAL_DATA` |
| `envscaler_tools_v2` (built in) | the EnvScaler corpus mounted and named by `RELIQUARY_ENVSCALER_DATA` (`scripts/fetch_envscaler_corpus.py`) |

Mount a persistent Hugging Face cache so the corpora are fetched once.

Build in a fresh clone of this repository, checked out at the Core `<commit>` whose validator image is the base,
where `git status --porcelain --ignored --untracked-files=all` prints nothing. The build reads
`docker/Dockerfile.order-control` and copies `tasksets/` from its context, the clone's working tree, not from
`<commit>`: a changed, untracked or ignored file there (a `build/` or `*.egg-info` that an earlier install left in a
taskset) would go into the image, and `.build-revision` (below) would not show it. `--untracked-files=all` overrides
the `status.showUntrackedFiles` setting: set to `no` in any git configuration of the host, it makes `git status`
leave out untracked and ignored files alike. The four commands are joined by `&&`, so that none runs after one that
fails: run one by one, a failed clone (into a directory that already exists, say) would leave
`git checkout --detach` to run in another repository, that directory's or the one the shell is in, and detach its
HEAD.

```bash
git clone https://github.com/reliquadotai/reliquary.git <new directory> &&
  cd <new directory> &&
  git checkout --detach <commit> &&
  git status --porcelain --ignored --untracked-files=all
```

Then, in that clone, `docker buildx imagetools inspect` prints the Digest of the base, the validator image the
image workflow built from `<commit>`:

```bash
docker buildx imagetools inspect \
  "ghcr.io/reliquadotai/reliquary-validator:sha-$(git rev-parse <commit> | cut -c1-7)-logic"
```

The next block builds from that digest and pushes. Its first three commands are a guard, joined to the build by
`&&`: the build runs only in a repository that holds `<commit>`, whose HEAD is that commit, and where
`git status --porcelain --ignored --untracked-files=all` prints nothing. Without it, the block run after a failed
clone or checkout would build another tree, and `.build-revision` would still say `<commit>`. The first command is
what stops the block outside any repository, where both sides of the comparison would be empty, and so equal. When
the guard passes it prints that commit and nothing else, so a block that stops after it, non-zero, with no build
output, ran at another HEAD or where `git status` shows a file. Keep the wheels' directory outside the clone, where
`git status` would show it. `--load` puts the image in the local image store whatever builder `docker build` uses:
with a `docker-container` builder, or with `BUILDX_NO_DEFAULT_LOAD=1`, a build without it passes, loads nothing and
ends 0, and the push then publishes whatever image last had that tag. The push, which runs only after a build that
succeeded, prints the digest of the image to check and to switch to:

```bash
git rev-parse --verify '<commit>^{commit}' &&
  test "$(git rev-parse HEAD)" = "$(git rev-parse --verify '<commit>^{commit}')" &&
  test -z "$(git status --porcelain --ignored --untracked-files=all)" &&
  docker build --load -f docker/Dockerfile.order-control --progress=plain \
    --build-arg BASE_IMAGE=ghcr.io/reliquadotai/reliquary-validator@sha256:<that digest> \
    --build-context envs=<directory holding exactly the four reviewed wheels> \
    -t <registry>/<repository>:<tag> . &&
  docker push <registry>/<repository>:<tag>
```

The build needs network to github.com (the `git+` installs); `ENVS`, `TELECOM_ENVS` and `SBX` are build args. Keep
the build and the push joined by `&&`: run as two commands, a failed build leaves the tag on the image an earlier
build gave it, and the push publishes that image, prints its digest and ends 0. Joined, the push having run is the
proof that the guard passed and that the build passed its two checks, `reliquary corpus order-control-check` and
the episodes check (which prints `ORDER_CONTROL_EPISODES_OK`): each is a `RUN` step, and a step that fails fails
the build. With `--load`, the image it pushes is the one this build made. So pipe nothing out of the build before
that `&&`: in `docker build ... 2>&1 | tee build.log && docker push ...` the `&&` reads the status of `tee`, and a
failed build still pushes. To keep a copy that way, run `set -o pipefail` first (bash or zsh), which makes the
pipeline fail with the build. `--progress=plain` prints the output of each step and leaves it on the screen, where
the default display on a terminal shows a step's output only while the step runs; a step taken from the build
cache prints `CACHED` instead, having passed in the build that cached it.

Put no comment in a block of this runbook, after a command or on a line of its own: zsh leaves
`interactivecomments` off by default, so it passes `#` and the words after it to the command as arguments
(`docker push` and `imagetools inspect` each take exactly one name), and runs a comment line as a command named `#`.

The validator image workflow (`.github/workflows/docker-image.yml`) tags each build that includes Logic
`sha-<the commit's first 7 characters>-logic` (`${GITHUB_SHA::7}`): `git rev-parse <commit> | cut -c1-7` prints
exactly those 7, where `git rev-parse --short=7` prints more when 7 are ambiguous in your clone. A push to main
builds only the commit at its tip, with Logic, and only when that push changes a file that is neither under `docs/`
nor a `.md` file (the workflow's `paths-ignore`): the commits under it, such as a pull request's own commits under
its merge commit, get none. A manual run (`workflow_dispatch`) builds the tip of the ref it runs on, whatever the
branch and whatever the commit changes, with Logic unless its `include_logic` input (true by default) is turned off.
So a `-logic` tag does not mean the commit is on main. A build that a later run of the workflow cancelled may have
pushed none. When `imagetools inspect` finds no such tag, take another commit.

Pin `BASE_IMAGE` by digest, never by tag: `latest` moves with main, and a new run of the workflow on a commit
pushes its tag again. Build the image once, on a non-production host, and push it: the checks before the switch
(below) and the switch itself use the digest that push printed. A second build from the same `BASE_IMAGE` and
arguments is another image, whose packages may differ: the `pip install` steps resolve dependencies at build time.

In the built image, `/opt/reliquary/.build-revision` is the Core commit that base was built from: the root
`Dockerfile` writes its `RELIQUARY_BUILD_REVISION` there (the validator image workflow sets it to the commit;
`unknown` when unset), and the order-control layers do not change it. It does not cover this image's own tasksets,
copied from the build context's `tasksets/`: the fresh clone above, which the build block's guard checks, is what
makes them those of `<commit>`.

For signed episode orders (SWE, TMax, Terminal-Bench 2.1, DeepSWE, SWE-bench Pro) the image also carries
reliquary-sandbox with its verifiers bridge at `SANDBOX_COMMIT`, `reliquary-swe` and `reliquary-terminal` at the
reliquary-environments commit the agentic eval sets pin (`ENVS`, checked against each set's `env_package`),
`reliquary-terminal-bench`, `reliquary-deepswe` and `reliquary-swebench-pro` at their own commits (`TB_ENVS`,
`DEEPSWE_ENVS`, `SWEPRO_ENVS`, checked the same way) and `renderers`; reliquary-harness comes with the base.
AutomationBench orders are suite episodes this control replays: `reliquary-automationbench` at
`AUTOMATIONBENCH_ENVS`, checked against its registry artifact. Every env package is at the commit
`reliquary/miner/env_packages.py` pins for miners. The Verifiers tasksets single-turn sets are graded with go to
`/opt/evalsets-site`, at the end of `sys.path`. A second build check fails unless all of these load at their pins.

## Before switching the admin or the order control

A rebuilt image grades only what its installed packages and grader fingerprint allow. Do these before switching
the admin or the order control to an image built from this Dockerfile, every step on the one image built and
pushed once (Image above), named by the digest its push printed:

```bash
IMG=<registry>/<repository>@sha256:<the digest docker push printed>
```

**Before the switch, never start the rebuilt image with production settings**, neither through its entrypoint nor
as `admin serve`. Its `ENTRYPOINT` is the order control itself (`reliquary corpus order-control`): a second order
control beside the running one, on the production bucket and registry. As `admin serve` it is a second admin, and
with `RELIQUARY_ORDER_CAP_SIZING=on` a second cap resizer writing the registry's caps. No setting makes the bucket
read-only: the store uses whatever `R2_*` credentials it is given. So until the switch the rebuilt image runs only
as one-shot containers that replace its entrypoint with `python` or `cat`, start no service, are removed on exit
(`--rm`) and are given no setting but the bucket variables of `$R2_ENV` (step 2); the running admin is checked
with `docker exec`. The one exception is in step 3, on a non-production host.

1. **The build's checks.** The push ran: the Image block's `&&` runs it only after its guard (a clean clone at
   `<commit>`) and a build that passed both checks, `reliquary corpus order-control-check` (its report,
   `rl_environments` included, is in the build's output) and the episodes check, which prints
   `ORDER_CONTROL_EPISODES_OK` (a step taken from the build cache prints `CACHED` instead: it passed in the build
   that cached it). Then, in the pushed image:
   ```bash
   docker run --rm --network none --entrypoint cat "$IMG" /opt/reliquary/.build-revision
   docker run --rm --network none --entrypoint python "$IMG" -m pip check
   docker run --rm --network none --entrypoint python "$IMG" -m pip freeze
   ```
   `.build-revision` must be the Core commit of the base you meant. `pip check` must be clean after the
   `--no-deps` installs. Compare `pip freeze` and each env package's and taskset's `direct_url.json` with the
   running containers (`docker exec <container> python -m pip freeze`).
2. **Seal check** (field level, read-only), in the running admin API container **and** in the rebuilt image, both
   reading the script from the same Core `<commit>`, whatever the checkout holds. `$R2_ENV` and `$SEAL`, the copy of
   the script that `git show` writes, live only as long as the check: the subshell creates both right before
   (`mktemp`, mode 600) and its `EXIT` trap removes them, whether the check passes, fails or is interrupted. The
   `INT`, `TERM` and `HUP` traps are what make bash and zsh run that `EXIT` trap on an interrupt, and they end the
   interrupted check with status 130, 143 or 129, not 0. Both checks read the script from `$SEAL`, never through a
   pipe from `git show`: bash 3.2 judges a Ctrl-C by a pipeline's first command, and `git show` has already exited
   0, so it would run no `INT` trap and the check would go on. In bash, a Ctrl-C that the running command survives
   runs no `INT` trap either: the check stops if that command fails, and goes on if it exits 0. Run the subshell as
   a command of its own, never chained with the switch: chained (`( ... ) && ...`, `||`, `if`), both shells ignore
   its `set -e`.
   ```bash
   (
     set -e -o pipefail
     R2_ENV="$(mktemp)"; SEAL="$(mktemp)"; trap 'rm -f "$R2_ENV" "$SEAL"' EXIT
     trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP
     git show <commit>:scripts/seal_check_admin.py > "$SEAL"
     docker exec <admin container> env |
       grep -E '^R2_(ACCOUNT_ID|ACCESS_KEY_ID|SECRET_ACCESS_KEY|BUCKET_ID|ENDPOINT_URL|REGION)=' > "$R2_ENV"
     docker exec -i <admin container> python - < "$SEAL"
     docker run --rm -i --entrypoint python --env-file "$R2_ENV" "$IMG" - < "$SEAL"
   )
   ```
   `$R2_ENV` holds exactly what that `grep` keeps: of the six variables `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
   `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_ID`, `R2_ENDPOINT_URL` and `R2_REGION`, those the admin sets. Never an
   `R2_PUBLIC_*` variable, nor any other setting. `R2_BUCKET_ID`, `R2_ENDPOINT_URL` and `R2_REGION` must be in it
   whenever the admin sets them: without `R2_BUCKET_ID` the store reads the `reliquary` bucket. The `grep` line
   copies the admin's own keys, so it runs on the admin's host only. Anywhere else, write the same variables by
   hand in its place, and the two keys must be those of an R2 token that can only read that bucket, never the
   admin's.

   It prints one line per set of its own list, `SETS`, fixed at `<commit>`: the script takes no argument and reads
   nothing from the platform. So its lines can include sets the platform has withdrawn, and miss sets it offers.
   Only the single-turn sets the platform offers count. The sets it offers are the rows of its `eval_sets` with
   `availability = 'qualified'`. Its public catalog, `https://api.reliqua.ai/api/v1/evaluations/catalog`, lists
   exactly those and needs no key. This read-only query, run in a checkout of reliquary-platform, lists them too,
   but it needs the operator's Cloudflare access, as step 5's queries do: `--remote` reads the production database.
   ```bash
   pnpm exec wrangler d1 execute reliquary-platform --remote --command "
     SELECT set_id, env FROM eval_sets WHERE availability = 'qualified' ORDER BY env, set_id"
   ```
   A set is single-turn when its card's `interaction` is `single_turn` or missing, and every Verifiers card's is
   `single_turn`. Today the single-turn sets offered are exactly those whose `env` starts with `verifiers:`. The
   others are agentic (SWE, TMax, Terminal-Bench, DeepSWE, SWE-bench Pro and AutomationBench today), graded from
   their jobs, and neither this step nor step 3 checks them. A non-null `subnet_episode` in the catalog marks an
   agentic set, but a null does not mark a single-turn one: with the platform's `EVAL_SUBNET_EPISODES` off, every
   set has a null `subnet_episode`.

   How to read the line of a single-turn set the platform offers:
   - `"sealed": false`: the card has no protocol, so neither comparison applies. Only step 3 tells whether the set
     grades.
   - `grader_ok: false` on a sealed card: the grader fingerprint is checked first. `/grade` answers 422, and the
     order ends in `grading_timeout` after generation: charged, no report. On an admin carrying #547 the order is
     refused at job creation instead.
   - `taskset_ok: false` on a sealed card: the rows come back ungraded and count as failures.

   For every single-turn set the platform offers whose card is sealed, the rebuilt image must print the same
   `grader_sha256_here` as the card and the same installed triple as the card. A single-turn set offered whose card
   fails either comparison goes through step 4 before the switch: a new set id, or its withdrawal. One the list
   lacks has no line: step 3, whose `load_sets` refuses a card sealed under another fingerprint, is its only check.
   The lines of other sets do not count.
3. **One-row grade** (the sufficient check). This step is not shipped as a script: write one that calls the
   grader's own entry points one by one, in this order:
   1. `reliquary.eval.grading.load_sets`;
   2. `require_gradable`, then `require_sandboxes`;
   3. `_Graders(...).grade(...)` on one row of the set's `grading.jsonl`, printing what it returns:
      `(score, detail, format_failure)`.

   In the rebuilt image, run it like the seal check, from its file (never through a pipe), with the ids of the
   single-turn sets the platform offers (step 2) as its arguments and `$R2_ENV` made as in step 2:
   ```bash
   (
     set -e -o pipefail
     R2_ENV="$(mktemp)"; trap 'rm -f "$R2_ENV"' EXIT
     trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP
     docker exec <admin container> env |
       grep -E '^R2_(ACCOUNT_ID|ACCESS_KEY_ID|SECRET_ACCESS_KEY|BUCKET_ID|ENDPOINT_URL|REGION)=' > "$R2_ENV"
     docker run --rm -i --memory 4g --entrypoint python --env-file "$R2_ENV" "$IMG" - <set ids> < <your script>
   )
   ```
   An empty detail or `format_failure` means the set grades, LiveCodeBench's excepted (below). `grader_error`,
   `grader_drift` or `source_drift` means it does not, as does a refusal by an entry point (`load_sets` raises
   `set protocol grader drift` for a card sealed under another fingerprint), with two exceptions. `require_gradable`
   refuses an agentic card with `set <id> holds <interaction> tasks: grade it from its job (eval grade)`: that set
   is not single-turn (step 2), and the refusal never means that it does not grade. And these `grader_error`s say
   nothing about the image:
   - **Memory.** `--memory` keeps the check from taking the memory of the services beside it; `4g` is a
     suggestion, not a measured need. A scoring child killed for lack of memory reads
     `grader_error: BrokenProcessPool: ...` (after one retry in a fresh child), and a script killed for it ends
     the container with exit status 137. Rerun with more memory before concluding anything.
   - **What this container lacks.** The GPQA taskset (`gpqa`), and HLE-Diamond's (`hle_diamond_text_det`) if the
     platform offers it, read gated Hub datasets with the `HF_TOKEN` of an account each dataset admits. GPQA's
     (research-environments `f2e0d709`, `gpqa/taskset.py`) loads `Idavidrein/gpqa` and the image holds no copy, so
     here the row of its set (`verifiers-gpqa-ab6f143e1-r0-n198`) reads `grader_error: DatasetNotFoundError:
     Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub. ...`. A LiveCodeBench or SciCode program runs in a
     grading service, reached through a socket mount and its variable ("Code-execution grading" and "SciCode"
     below). This container has neither, so these `grader_error`s never mean that the set does not grade: never
     withdraw a set for one. Grade their row on a non-production host instead, which holds none of the admin's
     settings: start the grading service there from `$IMG` as in those sections (it holds no secret and has no
     network), and give the one-shot container its socket and variable, a read-only `HF_TOKEN` of an admitted
     account (not the admin's) and `$R2_ENV` with the read-only R2 token. Never start the rebuilt image as the
     admin or the order control to grade them.

   LiveCodeBench's row needs more than an empty detail. Its taskset scores an answer that holds no fenced code
   block 0.0 before it calls the grading service, so with any other text the row reads as graded with no service
   at all. Grade it with a program, in a fenced code block, that passes that row's tests: the row grades only if
   its score is 1.0.
4. **A drifted set cannot be republished under its id.** Publication is create-only (`eval/storage.py`). Use a
   new set id and a new `eval_sets` row, and withdraw the old one.
5. **Do not switch the admin while a single-turn evaluation is in flight.** An order already taken is not
   refused: its rows come back ungraded.

   **Nor stop the admin while an export may be in flight**, for a switch, a rollback or any restart. Just before it
   stops, no order of the platform's `any_dataset_orders` (Batch orders and the other any-model orders) may be
   `exporting`, unless its `status_reason` is `export_waiting_for_settlement` (the admin answered
   `job_not_drained`: nothing is uploaded), nor `draining` with `drained_seen_at` set (its job is final, and the
   next cron steps move it to `exporting` and start its export). This read-only query, run in a checkout of
   reliquary-platform, lists the orders that forbid the stop, and must list none (`IS NOT`, not `<>`, keeps an
   `exporting` order whose `status_reason` is null):
   ```bash
   pnpm exec wrangler d1 execute reliquary-platform --remote --command "
     SELECT id, status, status_reason, drained_seen_at FROM any_dataset_orders
     WHERE (status = 'exporting' AND status_reason IS NOT 'export_waiting_for_settlement')
        OR (status = 'draining' AND drained_seen_at IS NOT NULL)"
   ```
   An export runs in the admin, which uploads each shard to the platform before `report.json` and `manifest.json`,
   and the platform keeps the first bytes of every delivery object: the same bytes again are accepted, others
   refused with 409 `delivery_conflict`. The next admin starts an interrupted export again. A Batch order's export
   (delivery v3) in `openai_jsonl` writes its stamp in every line (`created`), so an admin that takes a new stamp
   uploads other bytes and gets 409 `delivery_conflict`. It answers the platform 500, and the platform retries
   until the order has been `exporting` for `ORDER_EXPORT_TIMEOUT_SECONDS` (6 h by default), then fails it
   (`export_timeout`). #579 keeps the first stamp beside the job's records, so that the next attempt uploads the
   same bytes, but only between two admins that both carry it: a swap from an admin without it (which kept no
   stamp) or back to one (which never reads the kept stamp) still ends in `export_timeout`. Nor does it cover
   shard bytes that change for another reason: a void between the two attempts, or another output layout or
   pyarrow version (Parquet shards) in the new image.

   The admin also exports the platform's subnet task runs (delivery `subnet-<run id>`, `worker/subnet-tasks.ts` in
   reliquary-platform) to the same create-only store, which takes a run's objects only while it is `exporting`, and
   the next admin starts an interrupted one again too. That export is v2: its Parquet shards carry no stamp, and a
   restarted one reuses the stamp of a `report.json` already uploaded, but shard bytes that change between the two
   attempts (a void, another output layout or pyarrow version in the new image) get 409 `delivery_conflict` all the
   same. So this read-only query must list none too, whatever `SUBNET_TASKS_MODE` is: the cron advances every run
   already dispatched, whatever the mode. It lists the runs `exporting`, and those `draining` whose job the admin
   reported drained (`drained` in `progress_json`), which the next cron step can move to `exporting` and the one
   after it starts exporting:
   ```bash
   pnpm exec wrangler d1 execute reliquary-platform --remote --command "
     SELECT id, job_id, status, status_reason FROM subnet_task_runs
     WHERE status = 'exporting' OR (status = 'draining' AND json_extract(progress_json, '$.drained') = 1)"
   ```
   A subnet run has no export deadline: unlike an order, it never leaves `exporting` on a timeout. When every
   attempt of its export is refused with 409 `delivery_conflict`, the admin answers the platform 500, the run's
   `status_reason` keeps coming back to `The subnet is temporarily unavailable. Retrying the same task contract.`,
   and the query lists it for good. No attempt of this admin can finish such a run, so it does not forbid the stop:
   write its id down, let the query list nothing else, and have the run repaired by hand, since no platform route
   ends it.

   Run the queries a few seconds before the stop, not more: the platform's cron runs every minute, and one tick can
   start an export the queries let through (an `exporting` order waiting for settlement starts its own as soon as
   its job is final). Then run them again right after the stop, before the new container starts (Switch). An order
   or a run they list now and did not list before may have started its export in between, cut by the stop (an
   order whose `status_reason` is `export_running` did). Write it down and follow it up: if the next admin's
   attempt conflicts, an order retries with nothing flagged (`export_retry`) until `export_timeout` fails it and
   sets `needs_attention`, and a run retries for good.
6. **Switch to `$IMG` itself**, the digest every step above checked: never to a tag, nor to another build of the
   same commit. Switch only once you have seen every output below. A check that ends 0 has only run: the seal
   check prints its comparisons and ends 0 whatever they find, and `grade` returns a failing row's detail instead
   of raising, so a one-row grade script that prints it ends 0 too, as does the subshell around either.
   - Step 1: the push ran, so the guard and both build checks passed, `.build-revision` is the Core commit you
     meant, and `pip check` is clean.
   - Step 2, in the rebuilt image, for every single-turn set the platform offers (step 2 says how to list and
     recognise them; the lines of other sets do not count): a card (no `"card": null`) and, if it is sealed,
     `card_grader_sha256` equal to `grader_sha256_here`, `"grader_ok": true` and `"taskset_ok": true`. Such a set
     whose card fails goes through step 4 first, a new set id or its withdrawal; one the script does not list is
     left to step 3.
   - Step 3: one row of every single-turn set the platform offers grades (an empty detail or `format_failure`):
     GPQA's, and HLE-Diamond's if the platform offers it, with a read-only `HF_TOKEN`, LiveCodeBench's and
     SciCode's with their grading services, these on the non-production host, and LiveCodeBench's with a fenced
     program, at a score of 1.0.
   - Step 5, a few seconds before the admin stops: no single-turn evaluation in flight, and the queries list
     nothing but subnet runs written down as stuck. Right after the stop, before the new container starts, the
     same queries again, any new row written down (Switch).

   Before the switch, write down the image the old container runs, by its registry digest, and keep that image on
   the host and in the registry:
   ```bash
   docker inspect --format '{{.Config.Image}} {{.Image}}' <old container>
   docker image inspect --format '{{json .RepoDigests}}' <the image ID it printed>
   ```
   An image built on the host and never pushed or pulled has no registry digest: `RepoDigests` prints `[]`, or,
   with the containerd image store, a digest for each name of the image, which no registry holds
   (`docker buildx imagetools inspect <name>@<digest>` finds none). Then keep its image ID as the record (the
   second word the first command printed), with the image tagged (`docker tag <image ID> <name>:<tag>`) or pushed
   first. Either way, the rollback is the old container itself, stopped and renamed at the switch (Switch, below),
   which holds that image and its settings, unless that container was started with `--rm` (Switch, step 1).

## Switch

Once you have seen every output step 6 lists, switch on the host of the old container, the admin or the order
control:

1. While it still runs, write down the old container's ID, restart policy and `AutoRemove`:
   ```bash
   docker inspect --format '{{.Id}} {{json .HostConfig.RestartPolicy}} {{.HostConfig.AutoRemove}}' <old container>
   ```
   If `AutoRemove` is `true` (a container started with `--rm`), stop here and ask before going on: `docker stop`
   deletes such a container, and the rollback with it, which would then be a new container from the image ID step
   6 wrote down, with the deployment's options.
2. Set its restart policy to `no` while it still runs, then stop it (the admin a few seconds after step 5's
   queries listed nothing that forbids the stop):
   ```bash
   docker update --restart=no <old container> &&
     docker stop <old container>
   ```
   A restart of the Docker daemon (a reboot, an upgrade) starts a stopped container again if its policy is
   `always`, or `on-failure` and it ended with a non-zero status, as a stop can end it: the old container would
   then run beside the new one. Set before the stop, `no` leaves no moment for that. Keep the two commands joined
   by `&&`: the stop then runs only once the update has succeeded. Run as two commands, a failed update would leave
   the old policy on the stopped container, and step 3's check does not show the policy. For the admin, run step
   5's queries again now, before the new container starts, and write down any new row.
3. A stopped container keeps its name: rename it, which keeps it and its settings for a rollback, and check that
   the renamed container is the one step 1 wrote down, and has exited:
   ```bash
   docker rename <old container> <old container>-before-switch
   docker inspect --format '{{.Id}} {{.State.Status}}' <old container>-before-switch
   ```
   It must print the ID step 1 wrote down and `exited`. Only then start the new container from `$IMG`, under the
   old name, with the restart policy step 1 wrote down, not `no`: `--restart=<Name>`, with
   `on-failure:<MaximumRetryCount>` in place of `<Name>` when that count is not 0, and `no` only when the name is
   `no` or empty. Every other setting comes from the deployment's own options and env file, for the admin its
   entrypoint and command too (the image's `ENTRYPOINT` is the order control), never from a `docker inspect` of
   the old container written to a file: its `Env` holds the R2 write keys and, in the admin's,
   `RELIQUARY_ADMIN_SECRET`. The two never run at once (two admins, or two order controls, on the production
   bucket and registry).

A rollback is the same switch, back: steps 1 and 2 on the new container (with step 5's queries before and after
the stop, for the admin), then its rename and check as in step 3, under another free name. Then give the old
container its name back and the restart policy step 1 wrote down, and start it:
```bash
docker rename <old container>-before-switch <old container>
docker update --restart=<Name> <old container>
docker start <old container>
```
with `on-failure:<MaximumRetryCount>` in place of `<Name>` when that count is not 0, and `no` when the name is
empty.

## Run

```bash
docker run -d --name order-control --network host \
  --env-file <subnet bucket credentials> -e RELIQUARY_ADMIN_TASK_PREFIX=order- \
  <order-control image, by the digest checked before the switch> --netuid 81
```

Signed episode orders also need the order control's own sandbox token key (an Ed25519 PEM, mode 0600,
`python -m reliquary_sandbox.attest.signing generate PATH`; never the corpus validator's key): mount it read-only
and set `RELIQUARY_ORDER_SANDBOX_KEY_FILE`, `RELIQUARY_ORDER_SANDBOX_KEY_ID`,
`RELIQUARY_ORDER_SANDBOX_HOTKEY` (the ss58 address miners sign session requests for), optionally
`RELIQUARY_ORDER_SANDBOX_RETIRED_KEYS`, and `RELIQUARY_SANDBOX_MACHINE_IDS`. The key settings have their own
names so the corpus validator's (`RELIQUARY_SANDBOX_VALIDATOR_*`) are never read here; the sessions are stored
under `reliquary/sandbox/order-sessions/`, which no other validator restores. Each gateway lists the key's
public half in its validator keys. Without these settings, or when the sandbox services fail to start,
episode orders are refused (the reason is in the control's status) and every other order is served.
An order job's sandbox sessions are `/corpus/jobs/{job_id}/sandbox/sessions`, inside the order jobs' location.

Rented gateways. The platform rents sandbox gateways while agentic orders run and registers each one in the
machine directory through the admin service, under `RELIQUARY_ADMIN_SANDBOX_MACHINE_PREFIX` (for example
`rlq-gw-prod-`): the admin's `/admin/v1/sandbox-machines` routes register, drain or revoke, and read only the
machines named with it. Set the same prefix in the order control's `RELIQUARY_SANDBOX_MACHINE_PREFIXES`
(`["rlq-gw-prod-"]`, alongside or instead of `RELIQUARY_SANDBOX_MACHINE_IDS`): a gateway registered after the
start is polled and placed on from the next directory read (30 s), and one set `revoked` is left out, with no
restart. The control records each accepted heartbeat in the directory (`last_heartbeat`), which is how the
platform sees a gateway ready.

An episode order is kept from miners until a gateway can take its sessions: a fresh, active machine serving
its sandbox env at its exact package and options digest (its first task's), holding some of its images, with
slots. Until then `/corpus/jobs` lists it `starting` (not in `jobs`), its writes answer 409 `job_paused`, and
the control status shows `jobs[id]["held"] = "no_gateway"` (`"awaiting_audit"` while it waits for its audit
executor instead). It opens on the first directory read and heartbeat poll that see such a gateway. Once open it
stays open: if the fleet later drops to zero, session opens are refused retryably (`quota_live`, 429 with
`Retry-After`) rather than the job vanishing from miners.

The control's status (`GET /admin/v1/eval-control/status`, every 30 s) gives each episode order a `sandbox`
object the platform sizes the gateways from (design `docs/design/2026-11-12-sandbox-capacity-autoscale.md`):
`episodes_total`/`_finished`/`_left`, `sessions_held`, `episode_seconds` (median of the last 50 finished),
`tokens_per_episode`, `audit_episodes_per_second` and `quota_refusals_fleet_full` (last 10 min). A null is
not measured yet.

It needs network to drand and to the Hugging Face hub. Route it with the locations
`reliquary corpus order-nginx --port 8791` prints for the same `RELIQUARY_ADMIN_TASK_PREFIX`; miners need
no setting. The admin service and the order control must run with the same prefix.

## Code-execution grading (LiveCodeBench)

A Verifiers set whose reward runs code (`needs_runtime: true` on its card) is graded on the admin. For
LiveCodeBench the taskset's own `verify.py` (pinned by sha256 in `reliquary/eval/sandboxed_runtime.py`) runs in
the admin's scoring child, and each test program goes to the gVisor grading service's stdio mode: no Docker
socket in the admin. The service runs as a sidecar from the same image (it carries `runsc` and the grader bundle).
It holds no secret; like every local runsc grader it needs `--privileged`, and the hostile-code boundary is gVisor.

```bash
sudo install -d -m 0770 /run/reliquary-grader
docker run -d --name eval-grader --restart unless-stopped --init --privileged --cgroupns host \
  --network none -v /run/reliquary-grader:/run/reliquary-grader --entrypoint python <admin image> \
  -m reliquary.environment.grader.server --use-runsc \
  --socket /run/reliquary-grader/grader.sock --pool-size 16 --metrics-port 0
```

The admin container then gets the socket (mode 0660, root) and a persistent Hugging Face cache, with these among
its own options: `-v /run/reliquary-grader:/run/reliquary-grader`,
`-e RELIQUARY_EVAL_GRADER_SOCKET=/run/reliquary-grader/grader.sock` and
`-v <disk with ~12 GB free>:/root/.cache/huggingface`.

The taskset loads `livecodebench/code_generation_lite` (v6, ~4.5 GB, plus its Arrow cache) the first time a
grading opens it. Without the sidecar, a row whose answer holds a program, in a fenced code block, grades as
`grader_error` (`unreachable`), never as a wrong answer. An answer with no fenced code block scores 0.0 before the
service is called, sidecar or not: step 3 of "Before switching" grades LiveCodeBench's row with a program for that
reason.

## SciCode

SciCode (`verifiers:scicode-steps`, `tasksets/scicode_steps`) asks each sub-step in a fresh single-turn prompt that
carries the model's own code for the earlier steps, as upstream does. So one SciCode evaluation is a chain of sets,
one per step position (1..15 on the test split), each built from the graded answers of the one before
(`scicode_steps.chain`). One platform order runs the whole chain (design
`docs/design/2026-11-14-scicode-chained-orders.md`): after each position is graded the platform calls
`POST /admin/v1/evaluations/{eval_id}/chain-next {"chain": "scicode-steps"}`, which builds the next position's set
from that grading, publishes it to the subnet bucket and answers its id and prompt lengths; the last position
graded, `combine` with `"chain": "scicode-steps"` adds the main-problem and sub-step rates to the report.

Only the first position is published by hand, once (it is the catalog set). `build-set` makes
`verifiers-scicode-steps-a024c0a1e-r0-n64` (64 problems, prompts_sha256 `b8dda24b...7410`), and `publish-set`
publishes it to the subnet bucket, which the jobs and the grading read:

```bash
echo '{"split": "test", "step": 1, "previous": {}}' > args-1.json
reliquary eval build-set --source verifiers:scicode-steps --out sets/scicode-p1 --taskset-args @args-1.json
reliquary eval publish-set sets/scicode-p1
```

Then the platform copy, which `verifyOrderSets` reads: `prompts.jsonl` and a public `set.json` with `order`,

```bash
python - <<'PY'
import asyncio, json, pathlib
from reliquary.eval.storage import PlatformEvalStore, platform_key
d = pathlib.Path("sets/scicode-p1"); card = json.loads((d / "set.json").read_text())
prompts = (d / "prompts.jsonl").read_bytes()
order = [json.loads(l)["problem_id"] for l in prompts.decode().splitlines() if l.strip()]
public = {k: card[k] for k in ("set_id", "env", "count", "prompts_sha256")} | {"order": order}
store = PlatformEvalStore.from_environment()
asyncio.run(store.create_bytes(platform_key(card["set_id"], "prompts.jsonl"), prompts))
asyncio.run(store.create_bytes(platform_key(card["set_id"], "set.json"), json.dumps(public).encode()))
PY
```

then the platform's catalog row (platform `docs/EVALUATIONS_API.md`, SciCode). The admin host needs network to the
Hugging Face hub (the dataset, ~2 MB, and each graded model's tokenizer).

Its tests need numpy/scipy/sympy/h5py and the 1 GB reference targets, so its programs go to a second grading
service whose sandbox root is the SciCode grading image (`docker/scicode-grader`, stored by digest in the
Hippius registry, bucket `reliquary-task-images`, `reliquary/scicode-grader@sha256:484ab5f18a608c5a7568f2afcc145458d6e02f08632ba9fb0d46d831dc498488`,
0.57 GB; the raw targets file is `datasets/scicode/test_data-48b0272a88b1.h5` in the same bucket). Pull it through a
local `registry:2.8.3` on that bucket (the settings of the image mirror runbook). On the eval host, build the
bundle (the image's filesystem as the rootfs, with this checkout's worker and bundle config), then start the
service:

```bash
sudo install -d /opt/scicode-bundle
sudo install -m 0644 docker/scicode-grader/bundle-config.json /opt/scicode-bundle/config.json
sudo BUNDLE_DIR=/opt/scicode-bundle WORKER_SRC=reliquary/environment/grader/worker.py \
  PY_IMAGE=<registry>/reliquary/scicode-grader@sha256:484ab5f1...8488 bash scripts/build_grader_bundle.sh
sudo install -d -m 0770 /run/reliquary-grader-scicode
docker run -d --name eval-grader-scicode --restart unless-stopped --init --privileged --cgroupns host \
  --network none -v /run/reliquary-grader-scicode:/run/reliquary-grader-scicode \
  -v /opt/scicode-bundle:/opt/scicode-bundle:ro -e GRADER_BUNDLE_PATH=/opt/scicode-bundle \
  --entrypoint python <admin image> -m reliquary.environment.grader.server --use-runsc \
  --socket /run/reliquary-grader-scicode/grader.sock --pool-size 16 --metrics-port 0 --stdio-max-seconds 600
```

The admin container then gets, among its own options,
`-v /run/reliquary-grader-scicode:/run/reliquary-grader-scicode` and
`-e RELIQUARY_EVAL_SCICODE_GRADER_SOCKET=/run/reliquary-grader-scicode/grader.sock`.

A step has 300 s of wall clock (upstream: 1800 s) and 8 GiB of address space (10.11's reference needs ~4), one
BLAS thread: size `--pool-size` to the host's memory. Without
`RELIQUARY_EVAL_SCICODE_GRADER_SOCKET` a SciCode row is a grader error, never a zero: the default service's
sandbox has no numpy.
