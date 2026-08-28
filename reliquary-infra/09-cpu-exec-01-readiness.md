# cpu-exec-01 deployment readiness

## Outcome

The hostile CPU split is implemented as a deployable prototype without making
the new `ctrl-01` execute miner code. The future `cpu-exec-01` can be installed
from one immutable image artifact and one Ansible playbook after its private
address is known.

The boundary remains intentionally small:

~~~text
trusted coordinator / expected answers / scoring
  -> bounded content-bound request over HTTPS + mTLS
  -> cpu-exec-01 fixed admission capacity
  -> one gVisor/KVM container per candidate batch
  -> bounded primitive result
  -> trusted comparison
~~~

No queue cluster, database, Kubernetes control plane, provider SDK, wallet
library, GPU library, R2/HF client, or signer logic was added to the executor.

## Implemented controls

| Boundary | Implemented behavior |
|---|---|
| Source and runtime | Full Git revision, OCI image ID, grader runtime digest, dated gVisor release, committed SHA-256 plus independent SHA-512, and hash-locked Python artifacts; no APT dependency in the image build |
| Request | Strict schema, 4 MiB maximum, 1 MiB code maximum, deterministic content-bound job ID, maximum 256 ordered cases, maximum 30-second case timeout and 120-second overall batch deadline |
| Result | 2 MiB maximum, strict job/attempt/runtime/case binding, JSON-safe primitive values only; expected answers never leave control |
| Admission | Dedicated thread count equals configured in-flight limit; no hidden 40-thread cap; overload fails immediately with measurable `503` |
| Sandbox | gVisor KVM on bare metal, no network, read-only root, no capabilities, UID/GID 65534, no-new-privileges, bounded tmpfs/rlimits/output, new sandbox after every hostile batch |
| Outer service | Exact image ID, no pull, read-only container, private PID/IPC, host CPU/RAM/PID/no-file bounds, host networking bound to one explicit address |
| Host | KVM required, swap disabled, wallets refused, Docker bridge/NAT/iptables mutation disabled, host hardening, audit/fail2ban, loopback-only node exporter |
| Network | mTLS required, offline CA, 90-day leaf default, only ctrl source allowed inbound, new host egress to RFC1918/ULA/link-local rejected, sandbox has no NIC |
| Secrets | CPU host receives only its TLS server leaf. The CA key, validator client key, wallet, provider keys, storage keys, SSH keys for other hosts, datasets, and expected answers are absent |
| Evidence | Artifact manifest, package inventory, image inspection/history, checksums, host validator, smoke, malicious corpus, lifecycle, overload, and capacity tests |

The outer container is privileged because it starts gVisor sandboxes. This is
not presented as a control-host boundary: a gVisor escape is assumed to own the
disposable execution host. Host networking does not expand that assumed loss,
and it prevents Docker NAT rules from bypassing the firewall. The important
property is that the lost host contains no credential or trusted-network path.

## Why gVisor first and Firecracker second

The current Reliquary grader already has a tested OCI bundle, worker lifecycle,
failure taxonomy, and runsc cleanup behavior. gVisor/KVM reuses all of it and
makes the machine split a small change. The dated runtime in the image is
`20260817`.

Firecracker `v1.16.1` is the supported evaluation candidate as of this build.
Do not add a parallel guest-agent/rootfs/snapshot lifecycle before the gVisor
split is measured. Firecracker becomes justified if the hostile corpus exposes
a real gap, or if cold replacement and saturation measurements justify the
additional guest lifecycle. Any later snapshot files are trusted integration
artifacts: never resume state modified by a hostile guest, and never recycle a
microVM across miners.

## What is safe to do on ctrl-01 now

`ctrl-01` may:

- build the trusted source into a linux/amd64 executor image;
- inspect the image and run only its trusted version/digest commands;
- store the compressed image archive and evidence files;
- store the public Firecracker release archive/checksum for later evaluation;
- run Ansible syntax and manifest checks.

It must not:

- start the executor service;
- run the malicious miner corpus;
- mount `/dev/kvm` into this image;
- generate or retain the production offline CA private key;
- receive a Bittensor wallet merely because it is the control candidate.

## Exact deployment after the machine is rented

1. Attach `ctrl-01` and `cpu-exec-01` to an execution-only private link or a
   narrowly routed WireGuard link. Do not join the GPU workers or signer to it.
2. Generate production PKI on the operator workstation with
   `scripts/generate_cpu_executor_pki.sh`. Copy only `cpu-executor/` to the
   Ansible input and only `grader-client/` to control. Keep `ca/ca.key` offline.
3. Copy `inventory/cpu-exec.example.yml` outside Git. Fill its network values,
   artifact path/digests, PKI path, and small-host capacity values.
4. Run `playbooks/cpu-exec-01.yml` once as root. Confirm the new admin login,
   change inventory to that user, and rerun; idempotence must pass.
5. From control, prove no-client-certificate TLS failure, then run smoke,
   malicious-corpus, restart/reboot, lifecycle-soak, 1x, and 2x load tests.
6. Configure the current validator with the remote URL in `shadow` mode only.
   Keep local gVisor authoritative for at least 10,000 representative batches.
7. Require zero deterministic mismatch, no negative-label conversion on
   infrastructure failures, no sandbox/host/trusted-network access, healthy
   replacement, and admission-tail headroom before remote authority.
8. Change only the executor authority mode. Removing validator privilege and
   local runsc is the explicit post-cutover verification, not part of shadow.

## Measured gates, not guesses

For pool sizes 16, 32, 48, and 64 record request rate, API busy responses,
queue/admission wall time, sandbox replacement time, p50/p95/p99 execution,
timeout rate, runsc processes, worker restarts/reaps, CPU saturation, memory,
and Code-window deadline headroom. Test representative cases and worst-case
full-CPU/full-timeout cases separately.

An AX42 prototype should start at 16 workers, 16 in-flight, 7 CPU cores, 56 GiB
container memory, and one-use workers. Those values prove correctness and
isolation; they do not claim to absorb the theoretical 64-job production burst.
An AX162-class host should be bought only if the prototype shows CPU saturation
or admission queueing at required load.

## Remaining external gates

The code/package is ready, but production is not cut over. These facts cannot
be tested before the second machine exists:

- actual gVisor KVM startup and hostile corpus containment;
- provider/vSwitch or WireGuard routing and provider-firewall policy;
- certificate SAN for the final execution address;
- live shadow parity and end-to-end admission latency;
- host reboot/recovery and 10,000-job process/cgroup leak soak;
- the machine size required by measured production load.

`ctrl-01` remains non-authoritative and wallet-free. `signer-01`, remote GPU
proof verification, and the eventual pure CPU validator move remain separate
changes after this CPU boundary qualifies.
