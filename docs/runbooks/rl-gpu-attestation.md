# RL GPU admission — validator branch

Branch: `codex/rl-gpu-attestation`, based on `feat/rl-async` at
`f89bc58f669a115e2950b553636d1bf1a8a4dfae`. The adaptive miner work remains in
the separate `codex/rl-async-miner` checkout. No driver, GPU mode, wallet, public
validator, reward rule or main branch is changed by this feature.

## What is implemented

The CPU validator verifies NVIDIA NRAS signatures, freshness and GPU security
claims, records a recent admission for the submitting hotkey, and can require
that admission before granting new RL prompt leases. The ordinary rollout proof,
grading, measured concurrency, penalties and payment rules still apply.

Three operating modes, selected explicitly by the validator operator:

| `RELIQUARY_RL_GPU_ADMISSION` | Behaviour |
| --- | --- |
| `off` (default) | No GPU routes, no new admission condition, no JWT dependency loaded. |
| `observe` | Verify and collect hardware evidence; existing miners can still lease without it. |
| `required` | Refuse new leases without a current accepted hardware admission. |

An unsupported GPU or missing confidential-computing setup cannot silently fall
back to a self-declared identity in `required` mode. Start with `observe` while the
miner integration and actual hardware deployment are qualified. Changing mode
requires a validator restart. Keep the existing registration gate enabled.

## Trust and limits

The accepted input is the **NVIDIA-signed inner detached EAT**, exactly
`[["JWT", overall_jwt], {"GPU-0": device_jwt}]`. The Python SDK's outer HS256
wrapper, a miner's `verified: true` JSON, multi-GPU bundles and legacy claim
schemas are rejected. The current policy covers the documented v3/v4 base
claims, requires a matching overall/device version, and rejects opaque-data
mismatches. Compatibility with a real v4 device response still needs hardware
qualification; schema changes fail closed.

The validator checks both ES384 signatures, the fixed production NRAS issuer,
integer time claims, the nonce in both tokens, the exact SHA-256 digest of the
detached GPU JWT, secure boot, disabled debug, successful measurements and all
required driver/VBIOS integrity checks. Certificate status must be valid, OCSP
status good and certificate dates current. Present OCSP validity/nonce flags
must be true. Revocation warnings and error details are refused.

Keys come only from `https://nras.attestation.nvidia.com/.well-known/jwks.json`,
over verified HTTPS without redirects or token-selected URLs. The validator
caches them for five minutes, coalesces concurrent fetches and throttles refresh
to once per 30 seconds. An outage returns 503; expired keys never enable a new
admission. This is bounded caching, not an instantaneous revocation guarantee.

**This is hardware admission, not attested execution of the RL workload.**
The hotkey authenticates the request. NVIDIA authenticates the hardware report.
The fresh nonce ties that report to this hotkey/validator/run challenge, but an
operator could relay the challenge to another genuine GPU. A bare report does
not attest the miner's executable, local attachment, model weights, or which GPU
produced later tokens. Closing that gap needs an attested workload/channel and
session-key design, plus its own qualification. Every API status explicitly has
`execution_attested: false`.

## Data available to the validator and dashboard

Attested fields: a scope-specific UEID pseudonym, low-level hardware model,
driver and VBIOS versions, proof hash, verification and expiry times. The hardware
model is the vendor's measured identity, not a marketing SKU inferred by us.

Declared fields: GPU name, VRAM, compute capability, utilisation, temperature and
power. Their request is signed by the hotkey, but these are not NVIDIA-attested
telemetry. They are separately labelled and **do not raise quota or emissions**.
No GPU UUID from `nvidia-smi` is accepted as a cryptographic hardware identity.

Accepted admissions last at most five minutes from the report's issue time and
never beyond either token's expiry. One active hardware identity per hotkey and
one active hotkey per hardware identity are enforced transactionally **within
this validator/run**. This is not subnet-wide deduplication. Refreshing the same
GPU is allowed; active identity replacement is refused.

Admission is checked immediately before asking the existing lease collector to
allocate prompts. Each granted lease receives an immutable snapshot of that
check in `gpu_lease_hardware`. It states when the report expires and never claims
that generation occurred inside an enclave. Use `registry.lease_hardware(id)`
for later attribution. Refresh does not rewrite historical lease provenance.
Existing leases can still be submitted after admission expiry: renewal outages
must not strand already generated answers or replace native proof validation.

`GET /rl/gpu/fleet` exposes active verified device counts and hardware-model
aggregates without hotkeys, UEIDs or per-miner declarations. `GET /rl/quota`
adds only a compact admission status. Full status needs the owner's signature.
Responses use `Cache-Control: no-store`; there is no per-visitor NVIDIA polling.

## API and miner integration contract

Every GPU POST requires `X-Reliquary-Hotkey` matching `miner_hotkey`, registration
when the normal registration gate is enabled, a current integer `at`, and a
hotkey signature. The complete JSON request except `signature` is covered by
`reliquary.protocol.gpu_attestation.gpu_request_binding`, including its route,
validator audience, run, proof and declarations. The validator accepts a maximum
of four concurrent GPU requests and bounded request bodies; verification runs
off the event loop. Secrets, signatures, reports and raw UEIDs are not logged.

1. `POST /rl/gpu/challenge` with `{miner_hotkey, at, signature}` returns a random
   challenge ID and 32-byte hex nonce, valid for 120 seconds. At most six fresh
   challenges per hotkey per minute are issued, with the counter persisted.
2. Collect and attest the GPU using **that exact nonce**. For the NVIDIA CLI the
   relevant operation is `nvattest attest --device gpu --verifier remote --nonce
   <nonce>`. The miner/provider must configure NVIDIA credentials and compatible
   confidential-computing infrastructure separately. Extract the NVIDIA-signed
   `detached_eat` bundle, not an SDK wrapper or unverified decoded claims.
3. `POST /rl/gpu/attest` with `{miner_hotkey, at, challenge_id, bundle, declared?,
   signature}`. Challenges are atomically consumed before expensive verification,
   even for an invalid signed proof. A failed request or lost reply can be
   resolved with signed status and, if needed, a new challenge.
4. `POST /rl/gpu/status` with `{miner_hotkey, at, signature}` reads the caller's
   admission. Refresh before expiry, using a new challenge. The adaptive miner
   does **not yet perform this handshake automatically**; `required` is for the
   integrated pilot, not an unannounced switch for existing miners.

Responses distinguish malformed input (422), invalid evidence/signatures (403),
used/expired challenges and identity conflicts (409), challenge-rate limits
(429), and unavailable registration/JWKS/verification (503). Failures do not
themselves change RL scores, quota, reputation or payment.

## Installation and rollout

On the Linux validator environment, install the optional dependency:

```sh
python -m pip install -e '.[gpu-attestation]'
export RELIQUARY_RL_MODE=async
export RELIQUARY_RL_GPU_ADMISSION=observe
```

Start the validator using the existing operator procedure. State is stored in
`rl-gpu-admission.sqlite3` under the normal private validator state directory,
separately from the existing RL lease database. It is mode 0600 and refuses a
symlink as its state file. Used challenges, active ownership and lease snapshots
survive a restart. Keep that directory private and preserve the database when
restarting: deliberately deleting it resets this validator's replay history.

Before `required`, qualify: a real compatible GPU report with the production
NRAS chain, the miner's renewal loop, lost-response recovery, concurrent leases,
expiry and submission continuity. The supplied H100 reported **CC capable,
CC OFF, Not Ready** on 19 October 2026. Its confidential-computing configuration
was not changed. A positive hardware qualification is therefore still pending.
Do not activate CC or reboot a shared/rented server as a side effect of installing
this validator branch; arrange the provider configuration independently.

## Tests

```sh
python -m pytest tests/gpu_admission -q
python -m pytest tests/unit/test_rl_lease_routes.py tests/unit/test_rl_leases.py \
  tests/unit/test_rl_lease_quota.py tests/unit/test_rl_async_service.py \
  tests/unit/test_signatures.py -q
```

The GPU-admission suite uses actual P-384 signatures from an **ephemeral test
authority**, synthetic hotkeys in RAM, SQLite restarts/concurrency and the real
FastAPI lease/submission routes. It covers positive intake, forged/edited or
swapped tokens, nonce/issuer/run/audience errors, stale evidence, revoked
certificates, unsafe GPU state, duplicate JSON, algorithm/trust-root confusion,
JWKS outage/size/rate limits, replay after restart, identity collisions, declared
telemetry bounds, observe/required/off modes, private storage and submission
continuity after admission expiry. It never loads a real wallet or claims a
synthetic authority is NVIDIA. These tests qualify validator logic, not a
production GPU hardware attestation or attested model execution.

## Primary references

- [NVIDIA GPU claims](https://docs.nvidia.com/attestation/advanced-documentation/latest/claims-guide/gpu_claims.html)
- [Hopper example and detached-token digest](https://docs.nvidia.com/attestation/quick-start-guide/latest/attestation-examples/hopper_single_gpu.html)
- [NVIDIA CLI and nonce verification](https://docs.nvidia.com/attestation/nv-attestation-sdk-cpp/2.0.0/sdk-cli/command-reference.html)
- [NVIDIA reference NRAS key handling](https://github.com/NVIDIA/nvtrust/blob/main/guest_tools/attestation_sdk/src/nv_attestation_sdk/utils/nras_utils.py)
