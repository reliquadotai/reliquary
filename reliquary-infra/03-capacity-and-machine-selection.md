# Capacity model and machine-selection plan

## 1. What the live measurements imply

Define:

- W: collection target, 100 seconds;
- P: proof wall time for one sealed window;
- T: training plus post-proof local overhead;
- U: checkpoint publication cost amortized across its interval;
- K: number of independent proof worker slots.

Current production behaves approximately as:

~~~text
current GPU half = P + T
~~~

The live p50 values are:

~~~text
P50 = 105.1 s
T50 = 96.8 s
current sealed-window GPU half p50 = 208.0 s
~~~

The small difference between P plus T and the measured full half is expected
from log timestamp resolution and work not represented in the archive field.

With a separate proof plane and trainer, the steady-state lower bound becomes:

~~~text
target cycle >= max(W, proof stage with K workers, trainer stage)
~~~

The stages still have per-window dependencies, but different windows may occupy
different stages at the same time.

## 2. Expected throughput by configuration

### 2.1 One proof GPU plus one trainer GPU

With the current H100 proof timing:

~~~text
max(100, 105.1, 96.8) = about 105 s at p50
~~~

The live proof p95 is 156.2 seconds, so one proof GPU would still create a tail
backlog. It is a valid first split benchmark, not the final capacity answer.

The first safe architecture keeps checkpoint publication as a global trainable
window barrier for its observed 234 seconds once every 16 successful steps. Its
simple amortized cost is:

~~~text
234 / 16 = 14.6 s per trained step
~~~

That gives a conservative average planning value near:

~~~text
max(100, 105, 97) + 15 = about 120 s
~~~

If publication is materially accelerated, the value approaches 105 seconds.
Therefore a reasonable minimal-v1 shadow target for one proof H100 plus one
trainer H100 is about 30 trained windows/hour, with 34/hour as a later ceiling.

Relative to the measured 208-second p50 GPU half, that is approximately 1.9x to
2.0x before the publication barrier, about 1.7x including its observed
amortized cost, and around 1.7x at the measured proof tail.

### 2.2 Two proof workers plus one trainer

The p50 sum of individual proof-device time is 95.0 seconds and p95 is 141.4
seconds. Perfect division is impossible because rank order, prompt claims,
failure debt, and job-length tails limit parallelism, but the theoretical
device-time lower bounds are:

~~~text
p50: 95.0 / 2 = 47.5 s
p95: 141.4 / 2 = 70.7 s
~~~

The capacity gate for two qualified workers should be a real proof-wall p95
below 90 seconds, not an assumption of linear speedup. If achieved, collection
or the roughly 97-second trainer stage becomes the normal-window bottleneck.
With the observed publication barrier amortized, the minimal-v1 target is about
31 trained windows/hour. If later work safely hides or greatly accelerates that
barrier, the ceiling approaches 36/hour.

Two proof slots can be:

- two GPUs on one trusted node;
- two separately leased trusted/attested workers;
- one local and one remote worker during transition.

The scheduler already balances independent device IDs. The benchmark chooses
the cheapest arrangement that meets the exact proof-capacity and trust gate.

### 2.3 Faster trainer

Once proof p95 is below 100 seconds, the trainer is the next limit. A faster GPU
matters only if:

- the exact FLA/Triton kernels are supported;
- the 512-rollout step fits without changing microbatch semantics;
- twenty or more consecutive steps remain numerically equivalent;
- snapshot and restore are also supported;
- cost per accepted training step is better, not merely raw GPU peak FLOPS.

Changing the training algorithm, batch, collection duration, or checkpoint
cadence is outside this infrastructure project.

## 3. State-serving capacity

At inspection:

~~~text
state raw bytes       = 86,807
state gzip level 1    = 36,499
size reduction        = 58.0%
successful state rate = about 307 requests/s
~~~

The access-log sample carried about 25 MB/s, or about 200 Mbit/s, of successful
state bodies before other endpoints and protocol overhead. Gzip alone reduces
that estimate to roughly 10.5 MB/s. Serving the cached bytes outside Python
removes hundreds of upstream requests per second even though the edge still
sends the network bytes.

Initial capacity gate:

- 750 state requests/s for 30 minutes, roughly twice the observed total request
  rate;
- p99 edge latency below 100 ms from the load-generator region;
- state propagation from control below 250 ms p99;
- zero partial/cross-revision bodies;
- no increase in dynamic submission p95;
- no upstream 502/503 burst during an artificial 300-second checkpoint upload.

Use a CDN only if miner geography or egress cost justifies it. A local static
snapshot server is enough to remove the Python bottleneck.

## 4. Machine roles and candidate classes

This document deliberately selects capability classes, not a purchase order.
The research's quoted Hetzner prices are not stable enough to use: Hetzner's
current product pages and price presentation changed after the research pass.
Request an exact quote and availability immediately before purchase.

### 4.1 Control

Starting requirement:

- 16 fast physical cores / 32 threads;
- 64 GB RAM minimum, 128 GB preferred for headroom and process pools;
- ECC memory;
- two datacenter NVMe devices in RAID1;
- 1 Gbit/s initially;
- no GPU;
- rootless systemd services or narrowly scoped containers;
- no privileged container or Docker socket exposed to the public service.

An AX102-class system is a reasonable benchmark candidate. Hetzner currently
lists the AX102 as Ryzen 9 7950X3D, 128 GB expandable to 192 GB ECC, two 1.92 TB
datacenter NVMe devices, and 1 Gbit/s. The final choice should compare sustained
admission p95, single-thread latency, ECC, disk write latency, and actual price.

Do not buy a 48-core EPYC control host solely because the current all-in-one
validator shows high process counts. The grader and GPU model leave this host
in the target architecture.

### 4.2 CPU execution

Benchmark two classes:

- 16 physical fast cores / 128 GB ECC / two NVMe;
- 32-to-48 physical cores / 128-to-256 GB ECC / two or four NVMe.

Hard requirements:

- dedicated bare metal if Firecracker is used;
- KVM and IOMMU support confirmed before order;
- enough RAM for the target clean sandbox pool without swap;
- NVMe endurance appropriate for disposable scratch;
- provider API and wallet credentials absent.

The AX162/EPYC 9454P class is a scale-up candidate, not a starting conclusion.
It is justified only if the 2x-production load test cannot keep Code queue p95
under the gate on the smaller host. Current grading p95 is around 0.5 seconds,
so worker lifecycle and burst scheduling need to be profiled before purchasing
48 cores.

The active profile currently permits four concurrent Code admission groups of
16 rollouts, or a theoretical 64-sandbox burst. An EPYC 9454P supplies 48
physical cores and 96 SMT threads, not 96 independent cores. Count physical
cores for the security/capacity model; start with SMT disabled or keep sibling
threads in the same sandbox security domain. The detailed purchase gate is in
[CPU control and hostile-execution split](07-cpu-control-execution-split.md).

### 4.3 Signer

Starting requirement:

- 2 to 4 vCPU;
- 4 to 8 GB RAM;
- encrypted disk and encrypted hotkey file;
- no public ingress;
- a separate provider account or provider preferred;
- outbound chain access and narrowly required HF/R2 verification access;
- no CI runner, Docker socket, repository deployment key, or general shell
  automation.

Signer latency is not a throughput concern. Isolation and recoverability decide
this machine.

### 4.4 Proof GPU workers

Benchmark candidates:

| Candidate | Why test | Trust limitation |
|---|---|---|
| H100 80 GB | Exact current performance baseline and CVM-capable SKU | Cost |
| H200 141 GB | More memory/bandwidth and CVM capability | Runtime/provider availability |
| B200 | Potential speed and CVM capability | Kernel/runtime compatibility and cost |
| A100 40/80 GB | Potential lower-cost proof slot | Not Lium CVM-capable under current guidance |
| L40/L40S 48 GB | Potential low-cost proof slot | Not CVM-capable; must be dedicated/trusted for authority |

Proof verification should need much less memory than the combined live
train-plus-verify process, but no machine is approved from parameter count
alone. Measure real peak memory, FLA kernel support, both environments, long
completions, and forensic work.

For Lium specifically, current documentation limits its CVM path to Hopper
H100/H200 and Blackwell B200/GB200 families. Ordinary A100, L40S, RTX, and
workstation pods remain non-CVM and are shadow-only unless Reliquary accepts the
provider as an integrity authority.

### 4.5 Trainer GPU

Start from an H100 80 GB because it is the known live baseline. Test:

- H100 PCIe and SXM separately;
- H200;
- B200 after the exact production kernels qualify.

The existing RTX PRO 6000 Blackwell staging machine is useful for transport,
failure, state, and general training tests. Historical Reliquary evaluation
found that its runtime could not execute the production FLA Triton kernel and
used a PyTorch fallback. It must not be treated as a production speed or
bit-parity reference until that limitation is retested and removed.

The trainer remains warm. Auto-termination is based on idle/failure policy, not
one window. Replacement readiness is measured from provision request through:

1. image ready;
2. training state restored;
3. model and optimizer resident;
4. qualification complete;
5. fenced lease acquired.

### 4.6 Network and storage

The current OCI image is about 20 GB, and the model snapshot is multiple GB. At
1 Gbit/s, transferring 28 GB has a best-case wire time around 224 seconds before
protocol, disk, decompression, and model-load costs. This is why warm pools and
provider-side immutable caches matter more than impressive microVM boot
numbers.

Start German control and CPU hosts at 1 Gbit/s. Measure:

- state egress after gzip;
- R2 archive and job artifact throughput;
- checkpoint snapshot write;
- trainer-to-R2 state upload;
- publisher R2-to-HF transfer;
- proof worker model prewarm;
- recovery transfer.

Buy 10 Gbit/s only when a named stage is network-bound. Separate public state
egress from checkpoint transfer metrics before making that decision.

## 5. Benchmark matrix

### 5.1 Baseline first

Run the exact live image and checkpoint on a separate two-GPU machine:

1. current one-GPU shared mode;
2. local proof on GPU 0 and trainer on GPU 1;
3. two local proof GPUs plus one trainer if a three-GPU node is available.

This isolates the benefit of stage separation from network, provider, and
serialization overhead. It is the ceiling that a remote design must approach.

### 5.2 Proof qualification

For every GPU type/provider/runtime:

- exact OCI and model revision;
- at least 1,000 production-shaped proof jobs;
- Math, Code, forensics, long-token, reject, and infrastructure-error cases;
- p50/p95/p99 and maximum job time;
- per-window proof wall under the real scheduler;
- peak and retained GPU memory;
- job/result transfer bytes and latency;
- local-authoritative versus candidate result parity;
- worker restart and retry;
- old/new checkpoint drain and prewarm;
- cost per 1,000 proofs and per completed window.

Production gate:

- zero decision or evidence divergence in shadow corpus;
- exact checkpoint/profile binding;
- no infrastructure failure converted to a miner-negative result;
- aggregate proof p95 below 90 seconds with at least 20% capacity headroom;
- worker loss does not change winner ordering;
- attestation verified for a CVM worker before authority is enabled.

### 5.3 Trainer qualification

For every trainer candidate:

- start from an immutable full training state;
- replay identical ordered batches;
- run at least 20 consecutive accepted/skipped steps;
- record forward, backward, optimizer, cleanup, and total time separately;
- compare safety-gate metrics and candidate weights;
- measure peak GPU and host RAM;
- produce and restore a complete state bundle;
- kill the worker before compute, during compute, after compute, and during
  snapshot;
- replay from the last durable anchor;
- attempt a stale-lease commit from a second worker;
- measure checkpoint staging and publisher transfer independently;
- run the paired Math/Code promotion benchmark before any checkpoint is
  production-eligible.

Production gate:

- one and only one active lineage;
- no stale lease can commit;
- restored state passes replay and training-health checks;
- training p95 at or below the H100 baseline unless cost per step is materially
  better;
- public checkpoint content and profile match the approved state;
- no hotkey, HF token, provider key, or bucket credential on the worker.

### 5.4 CPU sandbox qualification

Run the same corpus through:

- current gVisor systrap;
- gVisor KVM on bare metal;
- Firecracker cold boot;
- Firecracker clean preboot pool.

Measure:

- queue wait and execution p50/p95/p99;
- sandboxes created/destroyed per second;
- CPU, RAM, disk, inode, process, and file-descriptor saturation;
- worker recycle/failure rate;
- result parity with the current grader;
- malicious corpus containment;
- host and sibling-sandbox survival.

Load gate:

- 2x the observed production Code arrival burst;
- grader execution p95 no worse than current;
- queue wait p95 below 1 second;
- zero cross-job visibility;
- no route or secret available from a compromised guest;
- supervisor loss is detected and Code jobs fail closed.

### 5.5 Provisioning and provider qualification

For each provider adapter:

- list/price/capacity response correctness;
- provision p50/p95;
- image and checkpoint cache hit/miss;
- bootstrap and certificate TTL;
- CVM quote availability and verification where claimed;
- interruption/failure behavior;
- termination confirmation and billing stop;
- leaked-instance reconciliation after orchestrator restart;
- cost reported versus invoice;
- quota and API rate-limit behavior.

Do not select a provider from hourly price alone. Use:

~~~text
cost per accepted trained window =
  trainer lease cost
  + proof warm-pool cost
  + recovery/provision waste
  + artifact storage and transfer
  + redundant or shadow verification
~~~

## 6. Recommended acquisition sequence

1. Do not modify or benchmark destructively on the production H100.
2. Rent or allocate a separate two-H100-class node for the local split ceiling.
3. Use the existing Blackwell staging host for protocol, transport, fencing,
   failure, and edge tests where exact FLA performance is not required.
4. Benchmark one smaller and one larger bare-metal CPU candidate before ordering
   the long-lived execution host.
5. Test an ordinary Lium worker only in shadow mode to validate the adapter and
   lifecycle.
6. Test a Lium CVM H100/H200 path, including renter-verifiable TDX and NVIDIA
   attestation, before considering it authoritative.
7. Compare at least one second GPU provider or a dedicated adapter.
8. Choose permanent control, CPU, signer, proof baseline, and trainer hardware
   only after the acceptance report contains real cost per accepted window.

## 7. Bottleneck migration

| Architecture state | Expected dominant bottleneck |
|---|---|
| Current | Proof plus training serialized on one H100 |
| Edge only | Same GPU path; public reliability improves |
| Separate proof/trainer, one proof slot | Proof p95 |
| Two qualified proof slots | Trainer step and 100-second collection |
| Separated checkpoint publisher, minimal v1 | Publication remains an explicit barrier but no longer stalls the public API process |
| Safely accelerated or later hidden publication | Trainer compute or collection |
| Faster trainer | Fixed collection duration unless protocol changes |

This is the desired progression: every added machine has a measured stage to
remove, and the design stops scaling when the protocol's fixed collection time
becomes the limit.

## 8. Current official references for machine assumptions

- Hetzner AX102 and AX162 family specifications:
  https://www.hetzner.com/dedicated-rootserver/ax102/
- Lium pod lifecycle and its approximate pre-cached-template deployment
  behavior:
  https://docs.lium.io/pod-users/create-pod
- Lium CVM hardware constraints:
  https://docs.lium.io/providers/nodes/cvm
- NVIDIA attestation prerequisites and claims:
  https://docs.nvidia.com/attestation/index.html

Prices and marketplace capacity are intentionally absent. Record them through
the provider adapter at benchmark time and obtain a fresh infrastructure quote
at the purchase decision.
