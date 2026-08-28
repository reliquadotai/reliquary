# ctrl-01 deployment and qualification report

- Deployment date: 2026-08-28
- Host: Hetzner AX42-1 `ctrl-01`
- Public addresses: intentionally retained only in the ignored local inventory
- OS after upgrade: Ubuntu 24.04.4 LTS, kernel `6.8.0-138-generic`
- CPU/RAM: Ryzen 7 PRO 8700GE, 8 physical / 16 logical CPUs, 61 GiB usable RAM
- Storage: two 512 GB NVMe devices, clean software RAID1, 437 GB root filesystem
- Authority: shadow only; no wallet, hotkey, provider credential, submission
  authority, weight authority, proof worker, trainer, or hostile-code runtime

## Deployed topology

~~~text
current authoritative validator (read-only public state source)
        |
        | bounded HTTP GET; shadow only
        v
ctrl-01 / reliquary-snapshot-sync
        | validates bounded JSON
        | content-addresses identity + gzip as one generation
        | atomically replaces the active symlink
        v
loopback nginx :8081
        |-- /state[?env=]       immutable snapshot
        |-- /health             immutable snapshot
        |-- /checkpoint         immutable snapshot
        |-- /runtime-contract   immutable snapshot
        |-- /livez, /readyz     local operational probes
        `-- writes/verdicts     503 control_not_authoritative

public firewall
        `-- SSH 22 only; HTTP/HTTPS closed
~~~

The direct HTTP source is acceptable only for non-authoritative shadow
comparison because the source data is already public. It must be replaced by a
local authenticated state sink when the validator control process moves to
this host. Production must not trust protocol state fetched over this shadow
HTTP hop.

## Installed controls

- Pinned SSH host key in the ignored local `inventory/known_hosts` file.
- Dedicated `reliquary-admin` account, public-key-only authentication, root SSH
  disabled, password/interactive authentication disabled, and SSH forwarding
  disabled.
- UFW default-deny inbound policy; only port 22 is open. Fail2ban uses the UFW
  action.
- Snapshot process runs as the zero-login `reliquary-edge` user with no
  capabilities and a restrictive systemd sandbox. `systemd-analyze security`
  reports exposure `3.4 OK`; its group-readable umask is intentional so only
  the nginx and Prometheus service groups can consume snapshots/metrics.
- nginx and Prometheus node exporter bind only to loopback. Successful state
  access logging is suppressed; errors and blocked write attempts remain
  observable.
- Docker is installed for a future pinned control image but has no containers,
  published port, or non-root socket user. Live restore, local bounded logs,
  no-new-privileges, and inter-container isolation are configured.
- Unencrypted swap was disabled before any key material was installed.
- Auditd, chrony, unattended upgrades, kernel/network sysctls, SMART tooling,
  and RAID monitoring are present.
- A daily local configuration-backup timer keeps 14 days. The archive was
  restored into a temporary tree and byte-compared with the deployed files.
  This is a recovery convenience, not an off-host backup.
- `/etc/reliquary/deployment.json` records the exact infrastructure Git
  revision and the non-authoritative, wallet-free role on every playbook run.
- Snapshot and fetch-age metrics are exported through the loopback node
  exporter textfile collector. No credential-bearing fields are emitted.

## Qualification evidence

| Gate | Result |
|---|---|
| Focused snapshot publisher tests | 3 passed |
| Ansible syntax and complete provisioning play | passed |
| Fresh root SSH connection | rejected |
| Admin public-key SSH and passwordless automation | passed |
| Effective SSH policy | root/password/interactive/forwarding all disabled |
| Host services after reboot | `running`, zero failed units |
| RAID1 | clean, 2 active, 2 working, 0 failed |
| NVMe SMART | passed on both devices |
| NTP | normal; measured offset about 0.3 ms |
| Public listeners | SSH only; application/metrics endpoints loopback-only |
| Wallet/provider/cloud credentials | absent |
| Identity/gzip integrity | decompressed bytes identical; SHA-256 equals generation name |
| Planned publisher stop | `/state` and `/readyz` immediately returned 503 |
| Publisher SIGKILL | one already in-flight 200, then 503 until automatic fresh-process recovery |
| Backup restore | required files extracted and byte-identical |
| 750 state requests/s | 15,000/15,000, 0 failures, p99 0.498 ms, 1.80 GB served |
| 1,500 state requests/s | 7,500/7,500, 0 failures, p99 0.361 ms, 898.61 MB served |

The load generator and nginx ran on the same machine, so these numbers isolate
server capacity from Internet latency. They prove the AX42 and static-serving
path exceed the 750 requests/s server-side gate; they are not a claim about
miner-to-Helsinki round-trip latency.

## Reproduction

From the repository root, install a temporary `ansible-core` environment, then:

~~~bash
cd reliquary-infra
ansible-playbook --private-key /path/to/admin-key playbooks/ctrl-01.yml
~~~

Copy and populate the files described in `inventory/README.md` first. The local
inventory defaults to `reliquary-admin`. A brand-new replacement host needs
one initial pass as root with `-e ansible_user=root -e ctrl_lock_root=false`, an
independent admin-login check, then the normal play to install the root lock.
Do not use those bootstrap overrides on the deployed host.

Local validation on the host:

~~~bash
sudo /usr/local/sbin/reliquary-validate-ctrl
~~~

The load test is intentionally not installed on the host:

~~~bash
ssh reliquary-admin@ctrl-01 \
  'python3 - --requests 15000 --rate 750 --concurrency 64 --allowed-status 200 --allowed-status 503' \
  < scripts/load_test_edge.py
~~~

`503` is an allowed `/state` protocol response between active windows. The
reported qualifying 750 and 1,500 requests/s samples happened entirely during
an active window and returned only 200.

## Rollback and failure behavior

- Stopping `reliquary-snapshot-sync` removes active symlinks through
  `ExecStopPost`; nginx immediately fails state/readiness closed.
- Starting it builds a new generation before readiness returns.
- A process crash is automatically restarted after two seconds. No stale
  generation is treated as ready during that interval.
- Disabling or removing `/etc/nginx/conf.d/reliquary-edge.conf` and reloading
  nginx removes the local shadow edge. No production route currently points at
  this host, so rollback has no miner impact.
- Re-running the playbook restores declared configuration. It never installs a
  wallet and refuses to run if `/root/.bittensor/wallets` is present.

## Gates that deliberately remain closed

`ctrl-01` is ready as a hardened shadow control foundation, not as the live
validator. Opening ports 80/443 or moving DNS now would break submissions and
is explicitly prohibited. Production authority requires, in order:

1. provision `cpu-exec-01`, generate the CPU-executor CA offline, install only
   the scoped mTLS identities, and pass remote gVisor parity/attack/load tests;
2. extract and qualify the versioned remote `ProofExecutor`, because the
   validator still needs a local CUDA proof process today;
3. shadow a CPU-only validator on `ctrl-01` and prove state, ranking, verdict,
   archive, checkpoint, and restart parity;
4. configure public DNS and TLS, then canary read traffic before any write;
5. provision a separate `signer-01` and move only the hotkey behind its narrow
   semantic API; the coldkey remains offline;
6. configure an authenticated private monitoring path and encrypted off-host
   backup target;
7. cut over one boundary at a time with the old validator retained as the
   rollback target.

No CPU-executor certificate was generated with a dummy address, no hotkey was
copied, no GPU was modified, and no production DNS or route was changed during
this deployment.
