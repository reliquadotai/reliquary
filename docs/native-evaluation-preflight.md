# Offline native evaluation preflight

`reliquary affine native-evaluation preflight --bundle BUNDLE --config CONFIG`
checks the existing immutable signed bundle without native execution or network
access. The owner-private configuration outside Git has schema
`affine-native-preflight/v1` and exactly these fields:

- `maxSeconds`, `rateMicroUsdPerHour`, `maxCostMicroUsd`.
- `providerCeilingMicroUsdPerGpuHour`, `gpuCount`, `companyCostCeilingMicroUsd`.
- `image`, pinned by SHA256, `requirementsFile` and `wheelDirectory`.

The image may be a repository reference ending in `@sha256:...` or an exact
local Docker image ID `sha256:...`. A local image ID identifies its configuration;
it is not a repository manifest digest. Offline preflight does not inspect Docker.

All money and duration values are bounded integers. The maximum customer reserve
and provider compute estimate round up once to integer micro-USD. They cover the
requested allocation duration. The company estimate covers GPU time only;
additional operating costs require a separate approved allowance.

The supplied lock must contain exact versions and SHA256 hashes. Available wheel
metadata and bytes must match that lock and the original signed job's native
package versions. Missing wheels are returned as blockers without downloads.
This inventory does not resolve dependencies, build an image, establish wheel
platform compatibility or qualify native numerical behavior.

The requested duration must fit the original job lifetime, frozen allocation
limit and finalization reserve. Customer and company ceilings are independent.
Fixtures always remain blocked for native execution. The result always reports
`nativeExecutionReady: false`; a usable inventory still needs a resolved runtime
image, H200 numerical qualification and account/capacity approval.

Exit code 2 means valid but blocked preparation; 1 means refused inputs; 0 means
offline checks passed. These statuses never launch, publish or fund a job.

To stage a public checkpoint before preflight, native `prepare` accepts
`--hf-repository OWNER/MODEL --hf-revision FULL_40_HEX_REVISION
--max-checkpoint-bytes BYTE_LIMIT`. It authenticates the original signed job
before fetching and copies only checkpoint bytes matching its signed inventory.
The private source receipt binds the immutable revision to the prepared profile.
HF transport options are refused by offline preflight. An internally authorized
nonpayable benchmark remains distinct from an externally authorized customer job.
