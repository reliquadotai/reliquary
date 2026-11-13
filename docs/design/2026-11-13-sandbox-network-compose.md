# Sandbox network, sidecars, fleet size and images (agentic benchmarks as evals)

Date: 2026-11-13. Status: design + implementation on local branches (nothing pushed or deployed):

| Repo | Branch | What |
|---|---|---|
| reliquary-sandbox | `feat/sandbox-network-compose` (from public/main 2ab7eb9) | egress (task field, settings, proxy, host setup), sidecars, docs/egress.md |
| reliquary-platform | `feat/sandbox-fleet-scale` (from origin/main) | wildcard gateway names (> 5 gateways), progressive image readiness in the gateway bootstrap |
| Catalyst | `feat/sandbox-network-compose` (from origin/main) | this document |

Goal: lift the sandbox blockers that keep agentic benchmarks (Terminal-Bench 2.1/4.1, SWE-bench Pro,
NL2Repo, DeepSWE, ...) out of the eval catalog (`~/runbooks/eval-catalog/PLAN.md`).

## 1. Threat model

Gateways should hold nothing valuable, so the worst outcome to plan for is a miner taking a gateway
down (availability), not data theft. A session token is enough to run arbitrary commands in a box:
tool calls reach the box before any inference proof is checked (the draw audit runs after the
episode). So each box is assumed hostile, with full use of its budgets.

New with egress: our IPs can be abused (scanning, spam, large downloads, hammering third parties).

### What is on a gateway today

| Item | Where | Protection | Reachable from a box? |
|---|---|---|---|
| Gateway TLS key (cert covering every gateway name) | `/etc/caddy/certs/gw.key` | 0640 root:caddy | no (host file) |
| Machine key (signs records, finals, grades, capacity) | `/etc/reliquary-sandbox/machine.pem` | 0600 root, made at boot | no |
| Local admin API token | `/etc/reliquary-sandbox/gateway.env` (0600) | Caddy exposes the whole API on :8443, token-gated | over the internet only, with the token |
| Validator public keys | `/root/validator-keys.json`, gateway.env | public | n/a |
| Docker socket | `/var/run/docker.sock` | root (root-equivalent) | no |
| Provisioner token, Targon/Cloudflare/subnet credentials | **not on the gateway** (Worker / ops host) | | |

Without a gVisor escape a box reaches none of these: offline boxes have no network, online boxes
reach only the proxy port (measured, section 3). A gVisor escape has to beat two layers: the
Sentry runs under a seccomp filter with no host `open`, and files come only through the gofer,
which is confined to the box rootfs. So a single Sentry bug is not enough. With host root,
everything above is exposed, and the machine key matters most: the attacker could sign its own
grades.

Key isolation options:

- Separate signer process (another user, Unix socket): cheap (~1 day), but it guards only
  against bugs in the gateway process. It does not help against host root, since root controls
  what gets sent for signing.
- Remote signer: no gain. A remote signer cannot tell a true grade from a false one; it signs
  whatever the host asks.
- What does help, and costs little:
  1. Per-boot keys revoked at destroy. Already the case: the key is made at boot and the
     directory entry is revoked, so a stolen key dies with the gateway.
  2. **Cross-gateway regrading**: the control re-grades a small random share of finished
     episodes (state in the transcript plus the task) on another gateway and flags mismatches.
     This catches forged grades whatever their cause. It is recommended as the follow-up for
     grade integrity. Not implemented here.

## 2. Per-box limits and DoS resistance

What reliquary-sandbox enforces today, from the code and the bootstrap settings:

| Resource | Enforcement | Prod value (rented gateway) |
|---|---|---|
| CPU | `nano_cpus` per box; CPU reservations with deliberate overcommit; separate grading pool; per-episode `cpu_s` budget | 2 CPUs/box; 24 CPUs reservable x4; grading 12 CPUs |
| Memory | `mem_limit = memswap_limit` (no swap); OOM blamed on the box only when its own cgroup hit the limit; reservations | budget <= 8 GiB/box; 156 GiB total; grading 28 GiB |
| pids | `pids_limit` per box; host `kernel.pid_max` raised | <= 1024 |
| Disk | xfs project quota on the box's writable layer (`storage_opt size`); refused without overlay2-on-xfs-pquota and runsc `--overlay2=none`; `/etc/{hosts,hostname,resolv.conf}` read-only and controller-owned | <= 10 GiB/box |
| Output / state | exec output cap; transcript cap; state cap; grader output cap | 64 KiB/call; 8 MiB; 32 MiB; 32 MiB |
| Logs | box log driver `none`; Caddy log rolled (100 MiB x 5) | |
| Calls / time | `max_calls`, per-call timeout (kills the process group), wall | 512; <= 600 s; <= 4 h |
| Sessions | capacity slots; validator-signed tokens; present window | 39; 120 s |
| Network | `none` (offline), so no bandwidth to abuse | |

Gaps (each would let one box degrade the whole gateway):

1. **Inodes**: Docker's `storage_opt size` sets only the block limit of the project quota. A box
   can create files without bound (metadata is not charged to the block quota) until the shared
   xfs loop file is full. Fix: after create, set `ihard` on the box's project id
   (`xfs_quota -x -c 'limit -p ihard=<N> <projid>'`, N ~ 1M). Measure first.
2. **Overcommitted disk**: the bootstrap makes a 1100 GiB sparse loop file. Prod uses
   `cpu-medium-amd`, whose root disk is ~753 GB (seen on the test VM). Boxes (39 x 10 GiB
   agent + grading) plus images can exceed the real disk, and then every box gets ENOSPC. Fix:
   size the loop file to the real free disk minus a margin, and keep
   `slots x disk_bytes + images` below it.
3. **Disk I/O bandwidth**: no `io.max`/blkio weight, so one box doing heavy I/O starves the
   others. Fix: `blkio_weight` or `device_write_bps` per box (gVisor gofer I/O is charged to the
   box cgroup). Measure first.
4. **Gateway API request rate**: no per-token rate limit on the HTTP API, only per-episode call
   budgets. A Caddy `rate_limit` per client IP or per token is cheap.
5. **Egress**: now bounded by the proxy (section 3); a box has no other path out.

## 3. Network at run time (controlled egress), implemented

Declared per task set and off by default: `SandboxTask.network = TaskNetwork(profile,
agent=False, grading=True)`, or for a verifiers env the option `"network": {"profile": ...,
"agent": ..., "grading": ...}`. The gateway must offer the profile for the env
(`EPISODE_EGRESS_ENVS`); otherwise the open is refused 422 (signed, `task_refused`). The network
is in the task fingerprint and in the options digest, while offline tasks keep their old digests
(the bootstrap pins the option digests). The capacity report lists `egress`. Phases matter: a SWE
task online in its agent phase could fetch the upstream fix, so SWE sets should be online in
grading only. Terminal-Bench allows the agent internet, so it gets `agent=True`.

Mechanism (host-side only; in-box iptables does not work under gVisor, as measured on 10-05):

- Internal Docker bridge `rq-egress` (172.31.250.0/24, `enable_icc=false`, IPv4 only). Host
  firewall: from the bridge, only TCP to the proxy port is accepted; DOCKER-USER drops anything
  forwarded. The controller refuses to start when the network does not meet this.
- Proxy `reliquary_sandbox_service.egress_proxy` (standard library), a systemd unit with
  `DynamicUser` and a cgroup IP filter that denies private ranges:
  - accepts only `CONNECT`, `GET` and `HEAD`;
  - checks the host against an allowlist (names only) and ports 80/443;
  - resolves the name itself and dials only a global unicast address it checked (no
    metadata, loopback, private or CGNAT);
  - pins apt by rewriting plain-HTTP mirrors to dated `snapshot.ubuntu.com` /
    `snapshot.debian.org`, and resolves relative redirects of rewritten requests against the
    snapshot;
  - pins PyPI through the box env: `UV_EXCLUDE_NEWER`, `PIP_UPLOADED_PRIOR_TO` (pip >= 26);
  - caps per box: 32 connections, 600 connects/min, 8 GiB/h, 50 MB/s; for the whole proxy:
    2048 connections, 400 MB/s;
  - writes a JSONL line per connection (source, host, port, URL for plain HTTP, verdict,
    bytes). The gateway logs each online box's address, so lines map to episodes.
- Setup: `scripts/egress-host-setup.sh <venv> [stamp]` is idempotent and prints the settings.

Abuse of our IPs: there is no raw socket path and no DNS. Ports 80/443 to listed package hosts
are the only reach, which rules out scanning and SMTP spam. Volume is capped per box and in
total, and every connection is logged.

Validated live (Targon cpu-medium VM, runsc release-20260928.0, Docker 29.9.3), with boxes made by
`TaskBoxRuntime.create`:

- **Blocked:** direct internet; metadata 169.254.169.254; host sshd and the gateway port through
  the bridge; docker0; box to box; Docker's resolver (TCP refused, UDP no answer); an offline box
  reaching the proxy. Through the proxy, `example.com` and IP literals get 403.
- **Works:** `pip install`; `apt-get update && install jq` through `snapshot.debian.org`
  (after the redirect fix); the `curl | sh` uv installer (astral.sh, releases.astral.sh, GitHub
  release assets); a uv date pin (`UV_EXCLUDE_NEWER=2020-01-01` resolves `certifi==2019.11.28`);
  npm registry; `git clone` over HTTPS.
- **Rate cap:** a 5 MB/s cap gave 21.4 MB in 3.5 s (5 MB burst, then 5 MB/s).

Unit tests: `tests/test_egress_proxy.py` (policy, rebinding, rewrite and redirect, caps, sources,
total rate) and `tests/test_egress_episodes.py` (refusal, per-phase specs, fingerprint, settings,
host check, verifiers option).

Not done:

- HTTPS caching. It would need TLS interception and a CA in arbitrary images, so it was left
  out; snapshots and pins give reproducibility instead.
- Recording URLs inside CONNECT tunnels (only host and bytes are logged).
- A per-set allowlist beyond the one `packages` profile. Add profiles as sets need them.

## 4. Multi-container tasks: sidecars implemented, pod variant validated

The 11 TB 4.1 compose tasks run a main container plus services (databases, web servers).

**Sidecars (implemented).** `SandboxTask.services = (TaskService(name, image@digest, command,
env, user, memory_bytes, cpus), ...)`, at most 8.

- For each box, the agent's and each grading box, the runtime creates an internal network
  `rq-ep-<hash>` (no route out; the host drops `rq-ep+` input).
- It starts one runsc box per service, with task capabilities, `no-new-privileges`, 256 pids, a
  2 GiB quota and its own memory and CPUs (capped at the box's), aliased by name.
- It then starts the task box, with the sidecars' addresses in its read-only `/etc/hosts`.
  Docker's resolver does not answer a box whose `resolv.conf` we mount; sidecars still find each
  other by alias.
- Sidecar memory comes out of the task's `limits.memory_bytes`, which must cover the sidecars
  plus 256 MiB.
- An online box with sidecars joins the egress network too. Images must be approved and present.
  Sidecars and the network are removed with the box.

Validated live: the box reaches `redis` by name; another episode's box cannot reach it by IP; the
host is unreachable from the sidecar bridge; egress works alongside; nothing is left after
removal. Box plus sidecar are created in 0.7 s. Unit tests: `tests/test_task_services.py`.

Limitation: grading boxes get fresh sidecars. A test that reads state the agent left in a
service (a database it filled) needs one of these:

- the env's `extract` dumps that state (e.g. `pg_dump` from the sidecar), which needs a runtime
  `exec` on a sidecar (small follow-up);
- the pod variant below;
- grading in place (not our model).

**Pod variant (validated, not implemented).** Docker-in-gVisor: an extra Docker runtime
`runsc-dind` (`--net-raw`) and a box with `--cap-add ALL`, which gVisor confines. Inside the box,
`dockerd --iptables=false` with compose networks set to
`com.docker.network.advertise_addr_nmsgs: "0"` runs the task's compose file unchanged, with names
resolving. This was measured: an inner container ran, and two compose services talked by name.
Inner images come from read-only bind-mounted tars (`docker load`, ~30 s/GB; `docker cp` into a
runsc box is not seen by the Sentry) or from inner pulls through the egress proxy.

Use it for tasks that need Docker inside the box (build or run containers) or stateful services
that must be graded in place. Costs: image load per episode, and `CAP_ALL` inside gVisor, so
gVisor is the only boundary, as today.

What remains for TB 4.1:

- a Harbor/compose to `TaskService` translation in the env package (main service = the one
  with the agent; others become services; `depends_on`/healthchecks become a wait in `prepare`);
- extraction from sidecars where tests need it;
- the 3 tasks needing an H100 stay out.

## 5. More than 5 gateways, implemented (platform)

`GATEWAY_HOST_DOMAIN=<domain>` replaces the hostname list. The slots become
`sbx-1..sbx-N.<domain>`, with `N = GATEWAY_MAX` (default 20, at most 200), and A records are
repointed in place as before, so the zone holds at most N of them. This needs one wildcard
certificate `*.<domain>`, issued off-box by DNS-01 like the current one (lego and the Cloudflare
token on the ops host; Let's Encrypt allows wildcards with DNS-01). Use a domain only gateways use
(`sbx.reliqua.ai`). Every rented VM holds the key, so it must never be valid for other names. That
is the same exposure as today's 5-name certificate.

Catalyst control and admin need no change: machines are trusted by prefix and directory key,
with no hostname list or count cap. Platform tests: config cases added to
`scripts/test-sandbox-gateways.mjs` (16 of 17 scenarios pass locally; the built-Worker scenario
needs `pnpm build`, not run here); `tsc` is clean.

Rollout:

1. Issue `*.sbx.reliqua.ai` (lego DNS-01 from `~/runbooks/gateway-fleet/lego-sbx-eval`).
2. Put it in `kit-prod/gw.{crt,key}`.
3. Set the Worker vars `GATEWAY_HOST_DOMAIN=sbx.reliqua.ai` and `GATEWAY_MAX`, unset
   `GATEWAY_HOSTNAMES`.
4. Raise `FLEET_DAILY_SPEND_MAX_USD` to match.

Live gateways keep their old names until destroyed (the old certificate stays valid).

## 6. Images: Hippius registry and progressive readiness

Hippius: a parallel effort (SWE-bench Verified full, worktrees `catalyst-swe-verified-full`,
`platform-swe-verified-full`) is meant to own the Hippius pull. As of this writing those branches
have no Hippius change and no design doc, so nothing here duplicates it. Suggested shape, from
`~/runbooks/image-mirror` and `~/runbooks/hippius-test/RESULTS.md`:

- On each gateway, run a local read-only `registry:2.8.3` on `127.0.0.1:5001`, backed by the
  bucket `reliquary-task-images` (layout `docker/registry/v2`, repo paths kept, read-only S3
  key).
- Put it first in `registry-mirrors`. That covers docker.io images (SWE-bench, TB), and digests
  and names are kept.
- Images from other registries (ghcr.io, public.ecr.aws) need the containerd snapshotter's
  `hosts.toml` mirrors, or a pull from the mirror followed by an approved-alias map in the
  runtime.

Progressive readiness (implemented, platform bootstrap):

- The bootstrap pulls the first `RQ_GW_FIRST_IMAGES` (default 8) images in the set's order,
  starts the gateway (`BOOTSTRAP_DONE`), then pulls the rest in the background in the
  `rq-image-pull` unit, with retries.
- There is one pull script (`/usr/local/sbin/rq-pull-images`), so the image source is changed
  in one place.
- The gateway already lists the approved images present every 45 s, and its capacity report
  carries them. The control places a session only on a machine holding its image. For eval
  orders (no open queue) a row whose image is still pulling is refused retryably
  (`sandbox_capacity`).

Caveat: for the RL issuer (open queue), a row no machine holds is `task_unplaceable`, final for
the seed. Before using progressive gateways for RL, the capacity report should list
`images_pending` and `Fleet.could_place` should count a pending image as "later" (small change,
both repos). Validated on the VM: the pull script and the background unit retry a bad digest and
pull the rest.

## 7. Second VM provider (survey, 2026-11-13; prices not measured)

| Provider | API | ~32 vCPU / 128 GB | Disk | Notes |
|---|---|---|---|---|
| Vultr VX1 `vx1-g-32c-128g-1920s` | REST + bearer | ~$1.22/h (~$0.038/vCPU-h) | 1.92 TB NVMe incl. | 1-2 min boot; new accounts have low instance limits (ticket) |
| Latitude.sh `m4.metal.medium` (16c/32t/128 GB) | REST + bearer, `GET /plans` has stock | ~$0.62/h | 2 x 1.9 TB | bare metal (KVM); same supplier as Targon's VMs |
| AWS EC2 spot `m7a.8xlarge` | SigV4 (`aws4fetch`) | ~$0.61-0.69/h spot | EBS | deepest capacity; quota requests; reclaim notice |
| Hetzner CCX53 | REST + bearer | ~$0.84/h | 600 GB | capacity restricted for new customers since 2026-06 |
| DigitalOcean / OVH / Scaleway | | $1.0-1.8/h | | price or auth friction |

Recommendation:

- Vultr VX1 as the first fallback: a bearer REST API that fits the Worker, the right size with
  disk included, US and EU regions. Open the account and request a limit raise (e.g. 5 x 32
  vCPU) now.
- Latitude.sh second, half the price per thread and with a stock endpoint.
- AWS spot as the last-resort backstop.

Platform work: a `VmProvider` interface over `TargonProvider` (create/state/destroy/list by name
prefix), `GATEWAY_PROVIDERS` in order of preference, and a provider recorded per row. The
bootstrap is provider-neutral except the SSH user and ports.

## 8. Remaining steps (in order)

1. Review and push the three branches (sandbox, platform, this doc), then open PRs.
2. Sandbox release, then gateway bootstrap: pin the new sandbox commit (`SBX=`), run
   `egress-host-setup.sh` only on gateways serving a set with a network, and add the egress
   settings to `gateway.env` for those envs.
3. Wildcard certificate and Worker vars (section 5).
4. Close the DoS gaps: inode quota, loop-file sizing, I/O weight, API rate limit (section 2).
5. TB 2.1 first (network in grading and agent, no compose), then SWE-bench Pro (grading only),
   then TB 4.1 (sidecars plus extraction).
6. `images_pending` before progressive RL gateways; Hippius mirror per section 6, owned by the
   SWE-Verified effort.
7. Cross-gateway regrading (grade integrity).
8. Second provider adapter (Vultr).
