"""GRAIL Protocol Constants.

Immutable values that all network participants must agree on.
No os.getenv() overrides. Changes require coordinated deployment.
"""

# ────────────────  GRAIL PROOF VERSION  ────────────────

GRAIL_PROOF_VERSION = "v7"

# ────────────────  CRYPTOGRAPHIC CONSTANTS  ────────────────

# Mersenne prime for modular sketch arithmetic.
PRIME_Q = 2_147_483_647

# Number of random challenge positions per completion.
CHALLENGE_K = 32

# PRF domain labels for different randomness derivations.
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}

# Transformer layer index for hidden state extraction (-1 = last layer).
LAYER_INDEX = -1

# Batch size for proof computation (log-softmax / GRAIL commitments).
# Fixed: changing causes numerical divergence between miner and validator.
PROOF_BATCH_SIZE = 16

# Top-K activation selection for sketch computation.
PROOF_TOPK = 16

# Logarithmic bucketing: buckets per sign (16 total = 8 positive + 8 negative).
PROOF_NUM_BUCKETS = 8

# Bounded coefficient range for sketch robustness: r in [-127, 127].
PROOF_COEFF_RANGE = 127

# Sketch tolerance at position 0. Keep this synchronized with the current
# production calibration and docs/mining.md. The value is deliberately higher
# than the strict same-GPU cheater-curve threshold to leave room for honest
# cross-kernel / cross-GPU numerical drift during the Qwen3.5 rollout.
PROOF_SKETCH_TOLERANCE_BASE = 5000

# Sketch tolerance sqrt growth factor per position.
# tolerance(P) = base + growth * sqrt(P).
PROOF_SKETCH_TOLERANCE_GROWTH = 5.0

# Attention implementation forced across all model loading paths.
# Override with GRAIL_ATTN_IMPL for test envs without flash-attn compiled
# (e.g. "eager" or "sdpa"). Production runs must stay on flash_attention_2
# because sketch commitments are bit-sensitive to attention kernel variance.
import os as _os
ATTN_IMPLEMENTATION = _os.environ.get("GRAIL_ATTN_IMPL", "flash_attention_2")

# ────────────────  TIMING (CONSENSUS)  ────────────────

# Blocks per window — 5 blocks × 12s ≈ 60s.
# All roles use this to determine window boundaries. With a typical tempo of
# 360 blocks, the EMA covers 72 windows of scoring history per on-chain
# weight submission, providing ~72× smoothing of miner scores over the epoch.
WINDOW_LENGTH = 5

# Bittensor block time target average (seconds).
BLOCK_TIME_SECONDS = 12

# Typical variance in block production time (seconds).
BLOCK_TIME_VARIANCE = 3

# Network latency allowance for file uploads (seconds).
NETWORK_UPLOAD_LATENCY = 30

# Grace period = block variance + upload latency.
UPLOAD_GRACE_PERIOD = BLOCK_TIME_VARIANCE + NETWORK_UPLOAD_LATENCY

# Buffer for future drand beacon (seconds).
DRAND_FUTURE_BUFFER = 30

# Buffer subtracted from per-window deadline to leave room for final submissions.
UPLOAD_BUFFER = NETWORK_UPLOAD_LATENCY

# ────────────────  ROLLOUT GENERATION  ────────────────

# Network-wide hard schema cap on completion length. BFT math rollouts use a
# lower measured local cap (thinking + FORCE + answer) that the validator
# recognizes separately for forced rows; keep this global cap high enough for
# non-BFT / opt-in code rollouts without revisiting verifier memory ceilings.
MAX_NEW_TOKENS_PROTOCOL_CAP = 32768

# Budget-Forced Termination (BFT): if a rollout has not emitted </think> by
# BFT_THINKING_BUDGET tokens, the miner appends BFT_FORCE_TEMPLATE and samples
# the answer in BFT_ANSWER_BUDGET more tokens, so it terminates with a real
# (gradeable) answer instead of an unparseable thinking truncation. H200 sweeps
# on real OpenMathInstruct prompts found 2048/512 to match 2048/256 reward
# (5/6) with better EOS rate, while 4096/256 was slower and lower reward.
BFT_ENABLED = True
BFT_THINKING_BUDGET = 2048
BFT_ANSWER_BUDGET = 512
BFT_FORCE_TEMPLATE = "</think>\n\nFinal Answer: \\boxed{"

# Two-sided length reward shaping (applied to ADVANTAGES, not the σ-gate).
# Under-thinking side: a non-forced rollout that finished early
# (completion_length < SHAPE_LEN_FRAC · BFT_THINKING_BUDGET) and is wrong gets its
# advantage set to −SHAPE_PENALTY. Overlong side: a cap-truncated rollout gets
# the same penalty. SHAPE_PENALTY = 0 disables shaping. TODO: sweep both values.
SHAPE_PENALTY = 0.5
SHAPE_LEN_FRAC = 0.5

# Cap/non-EOS truncation budget per submission. A single missing-EOS rollout
# can be an honest local max-token accident; repeated missing-EOS rollouts in
# one GRPO group are a sampling policy, not a rare exception, and have become
# the main manufactured-loser path.
MAX_TRUNCATED_PER_SUBMISSION = 1
BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION = 1

# Group-level reward-shape guard. The live attack manufactures binary reward
# vectors such as 11110000 while cutting every zero-reward rollout to the same
# local cap (120/150/4500 tokens). Exact repeated loser lengths are vanishingly
# unlikely under natural sampling, especially when they occupy the ordered
# suffix of the reward vector.
REWARD_SHAPE_ZERO_MODE_MIN_LENGTH = 64
REWARD_SHAPE_ZERO_MODE_MIN_SHARE = 0.75
REWARD_SHAPE_MIN_REPEATED_ZERO_ROLLOUTS = 2
REWARD_SHAPE_MIN_EXACT_ZERO_ROLLOUTS = 3

# Training quarantine policy. Quarantine is intentionally conservative: it
# skips train_step for windows whose selected batch has high-confidence poison
# signatures, while still archiving the window so emissions/forensics remain
# visible.
TRAINING_QUARANTINE_ENABLED = True
TRAINING_QUARANTINE_MAX_HOTKEY_SHARE = 0.75
TRAINING_QUARANTINE_MAX_REWARD_VECTOR_SHARE = 0.75
TRAINING_QUARANTINE_REWARD_VECTOR_MIN_GROUPS = 4
TRAINING_QUARANTINE_MAX_MEAN_COMPLETION_LENGTH = 24576
TRAINING_QUARANTINE_MAX_SINGLE_COMPLETION_LENGTH = 32768
TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_ROLLOUTS = 4
TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_GROUPS = 3
TRAINING_QUARANTINE_REJECT_SPIKE_MIN = 32
TRAINING_QUARANTINE_REWARD_SHAPE_MIN_GROUPS = 2
TRAINING_QUARANTINE_LONG_ZERO_TAIL_MIN_LENGTH = 16384

# Soft cap on per-hotkey entries persisted to ``archive["rejected"]`` per
# window. Beyond this, ``reject_counts`` still increments but no metadata is
# appended — protects the R2 payload size against a flood of garbage
# submissions from a single attacker.
REJECTED_LIST_CAP_PER_HOTKEY = 5

# ────────────────  GRPO BATCHING  ────────────────

# Default HTTP port the validator listens on for miner submissions.
VALIDATOR_HTTP_PORT = 8888

# Hard ceiling on grading attempts (reward computation) per window — the
# anti-DoS bound on admission and on the submit queue. Grading is what
# admission actually costs now that the GPU proof is deferred to seal, and a
# grading attempt is charged only when grading starts, so it is never refunded
# and discarded queue items never burn one. It is also the outer bound on GPU
# proof work at seal: _prove_ranked can prove at most this many candidates.
MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW = 96

# Absolute server-side bound across active and draining environment queues.
# The per-window grading ceiling remains the primary bound; this is the final
# backstop during a window swap or prolonged GPU stall.
MAX_PENDING_PROOF_QUEUE_DEPTH = 256

# A hotkey whose ranked candidates repeatedly fail the expensive proof should
# not consume the whole window's proof budget. Allow a small number of misses
# for honest stack drift, then skip that hotkey's remaining candidates until
# the next window. Enforced in ``_prove_ranked`` at seal.
MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW = 2

# Registered-hotkey admission cache. The validator refreshes the metagraph on
# this cadence and once on a cache miss. A last-known-good snapshot may survive
# a short chain outage, but admission fails closed after the grace period.
REGISTERED_HOTKEY_CACHE_TTL_SECONDS = 300.0
REGISTERED_HOTKEY_STALE_GRACE_SECONDS = 900.0
REGISTERED_HOTKEY_REFRESH_MIN_INTERVAL_SECONDS = 15.0
REGISTERED_HOTKEY_REFRESH_TIMEOUT_SECONDS = 20.0

# Safety timeout for the seal extension's drain phase. Once the trigger drand
# round expires, the validator waits for the queue AND in-flight GRAIL proofs to
# drain so every admitted trigger-round candidate is paid. Sized to cover serial
# verification (~1 proof/s) of a full window's admitted set; after this timeout
# it seals anyway so a slow or constantly refilled queue cannot freeze checkpoints.
MAX_SEAL_QUEUE_DRAIN_SECONDS = 60.0

# Liveness poll interval while an OPEN validator window waits for either a
# normal seal trigger or an exhausted proof-admission queue. This prevents a
# window from waiting WINDOW_TIMEOUT_SECONDS after all admitted proof work has
# drained but fewer than B valid submissions survived validation.
PROOF_ADMISSION_STALL_POLL_SECONDS = 0.5

# Sparse-window liveness breaker. After recent validator hardening, honest but
# stale/misconfigured miners can leave a window with some valid submissions but
# fewer than B distinct trainable prompts. Keep the normal 8-distinct seal as
# the happy path, but do not let sparse valid traffic hold checkpoint progress
# for the long safety-net timeout. Idle/age caps are sized to give slower
# fresh-model submissions (longer generations, bursty arrivals) time to reach
# 8 distinct before a sparse window is force-sealed.
SPARSE_VALID_IDLE_SEAL_SECONDS = 300.0
SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS = 4
SPARSE_VALID_MAX_WINDOW_SECONDS = 900.0

# Fixed collection window for the difficulty auction. The window no longer seals
# on the 8th distinct prompt; it stays open for this long and accepts everything.
#
# This is also the MINIMUM window duration, by construction. An early seal would
# be the speed race we are removing: whoever triggered it would cut off the
# slow-but-hard submissions still generating. Sized from live traffic — math
# generation is 176s at the median and 267s at p75, and windows already run 277s
# of collection today, so 300s captures ~89% of submissions at near-zero cadence
# cost ONCE proofs are deferred (spec §2.5).
WINDOW_COLLECTION_SECONDS = 300.0

# UID that receives unused slot emission budget (the burn address).
UID_BURN = 0

# ────────────────  VALIDATION RULES  ────────────────

# File size bounds for valid rollout window files.
MIN_ROLLOUT_FILE_SIZE_BYTES = 200
MAX_ROLLOUT_FILE_SIZE_BYTES = 350 * 1024 * 1024  # 350 MB

# ────────────────  CONTINUOUS VALIDATION  ────────────────

# How often the validator polls for new state (seconds).
POLL_INTERVAL_SECONDS = 10

# ────────────────  WEIGHT SUBMISSION  ────────────────

# Submit weights when blocks_until_next_epoch <= this value. Tuned so all
# validators of a netuid land in the same ~20-block window (≈4 min on
# 12s/block) and read near-identical R2 archive snapshots, then converge
# to identical weights via the deterministic EMA replay.
EPOCH_SUBMIT_LEAD_BLOCKS = 20

# ────────────────  STORAGE  ────────────────

CHECKPOINT_PREFIX = "reliquary/checkpoints/"

# ────────────────  HUGGING FACE CHECKPOINT PUBLISHING  ────────────────

# How often to publish the current in-memory model to Hugging Face.
# Training happens every window (stub in v2.1, real GRPO in follow-up),
# but HF uploads are slow for large models, so we publish only every
# N windows. Between publishes, miners stay on the last pushed revision.
CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10

# Default HF repo target for published checkpoints. Operator may
# override via --hf-repo-id CLI arg. Must be a writable repo id for
# the validator's HF token.
DEFAULT_HF_REPO_ID = "aivolutionedge/reliquary-sn"

# ────────────────  DEPRECATED (GRPO REFACTOR)  ────────────────
# Kept importable to avoid breaking transitive imports during the rollout.
# These knobs no longer participate in any runtime decision and will be
# removed in a follow-up cleanup once no consumer references them.

MINER_SAMPLING_ENABLED = True
MINER_SAMPLE_RATE = 0.25
MINER_SAMPLE_MIN = 2
MINER_SAMPLE_MAX = 35

ROLLOUT_SAMPLE_RATE = 0.10
ROLLOUT_SAMPLE_MIN = 16

VERIFICATION_BATCH_SIZE = 16
BATCH_FAILURE_THRESHOLD = 0.30

FAILURE_LOOKBACK_WINDOWS = 14
USED_INDICES_MAX_AGE_WINDOWS = 100

MAX_ROLLOUTS_PER_FILE = 6000

DATASET_NAME = "karpathy/climbmix-400b-shuffle"
DATASET_SPLIT = "train"

# ────────────────  GRPO MARKET (v2)  ────────────────

# Minimum reward-std for a group to pass the zone filter. For binary
# Bernoulli rewards this admits k ∈ [2, 6] for M=8 (σ at k=2/6 ≈ 0.433).
# For continuous rewards it filters groups whose rollouts clustered too
# tight to carry meaningful GRPO signal.
SIGMA_MIN = 0.43
BOOTSTRAP_SIGMA_MIN = 0.33

# Difficulty-auction tilt. The auction scores a group with
# ``rewards_std(r) * (1 - p) ** DIFFICULTY_DELTA`` where p is the group's mean
# reward. The first factor is sigma (what the zone filter already measures); the
# second is the failure rate, and it is the ONLY thing that separates a group the
# model mostly fails from its mirror image that it mostly passes.
#
# Sigma alone cannot: sqrt(p(1-p)) is symmetric, so k=2 and k=6 look identical to
# it. They are not. GRPO's advantage is (r - mu)/sigma, so at k=6 the dominant
# per-sample signal is -1.73 on the two mistakes (suppress noise on a prompt the
# model already solves) while at k=2 it is +1.73 on the two rare successes
# (amplify a discovery). Same sigma, opposite pedagogical value.
#
# delta = 1.0 puts the peak at k=2 of 8. Raising it tilts harder; delta = 0
# collapses the score back to plain sigma. The peak is deliberately NOT at k=1:
# a lone success in 8 may be a lucky guess on wrong reasoning, and reinforcing it
# teaches the wrong thing.
DIFFICULTY_DELTA = 1.0

# Cap on slots one OPERATOR (coldkey) may win per window under the difficulty
# auction. The 8-distinct seal cap is per PROMPT, not per miner — which is how a
# single coldkey took 13.1% of emission by flooding distinct prompts across many
# hotkeys. It also bounds the centralisation pressure the auction's speed
# tie-break creates: with a coarse 7-valued score, ties are the norm, so the
# fastest hardware would otherwise win every one of them.
MAX_SLOTS_PER_COLDKEY_PER_WINDOW = 2

# There is deliberately NO fixed per-window proof-count ceiling. `_prove_ranked`
# proves ranked candidates top-down until B_BATCH pass. A fixed cap (we tried 16)
# is worse than none: a fabricated group ranks at the TOP by construction (the
# score comes from a hand-writable reward vector, peaking at k=2) and fails GRAIL,
# so a fixed cap is exhausted by fakes BEFORE honest candidates ranked below them
# are reached — starving the batch. The griefer is bounded instead by the
# per-hotkey failure cap (each hotkey is skipped after
# MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW failures = one registration
# per 2 wasted proofs), and the pool itself by MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW.

# Random non-winners proven each window purely for forensics. Deferring the proof
# means the authenticity gates (token-auth, distribution, forced-seed) only ever
# see the WINNERS — enforcement stays complete (nobody unproven is paid) but fleet
# VISIBILITY is lost, and those gates are how we caught the pre-generating miner
# and 1088 seed_mismatch rejects in 855 windows. Sampled by drand so a miner
# cannot predict whether he will be looked at.
FORENSIC_SAMPLE_PER_WINDOW = 2

# Number of rollouts per submission (= size of each GRPO group).
M_ROLLOUTS = 8

# Training batch size per active environment. Final selection is drand-round /
# canonical ordered, not TCP FIFO; distinct prompt representatives feed GRPO.
B_BATCH = 8

# (env_name, prompts_per_batch). Sum across entries = total prompts
# processed per optimizer step. With 2 envs at B_BATCH each, we train
# on 16 prompts × M_ROLLOUTS = 128 sequences per step.
ENVIRONMENT_MIX: list[tuple[str, int]] = [
    ("openmathinstruct", B_BATCH),
    ("opencodeinstruct", B_BATCH),
]

# Runtime default for CLI/Docker operators. OpenCode remains available through
# ENVIRONMENT_MIX, but code execution is opt-in until the runsc canary and
# miner rollout are coordinated.
DEFAULT_ENVIRONMENTS: str = "openmathinstruct"

# Number of micro-batches accumulated before an optimizer step. Derived
# from the mix — one micro-batch per active env. Not separately tunable
# to keep semantics simple.
GRAD_ACCUM_STEPS: int = len(ENVIRONMENT_MIX)

# Per-env weight in the GRPO loss. The step normalizes token-level *within*
# each env (each batch is one env), then recombines as Σ_e w_e·L_e. Empty =
# equal weights (renormalized over the envs present in a window) — so a
# long-completion env (code) cannot dominate a short-completion env (math) via
# raw token mass. Override to bias the mix, e.g. {"openmathinstruct": 2.0,
# "opencodeinstruct": 1.0}; unlisted envs default to 1.0.
ENV_LOSS_WEIGHTS: dict[str, float] = {}

# Sampling fixed at protocol level. Miners who use different values produce
# samples from a different distribution -> biased GRPO gradient. Keep these
# values exactly mirrored by the miner generation path and the validator's
# chosen-token probability checks.
T_PROTO = 0.6

# Top-p and top-k for sampling (fixed alongside T_PROTO).
TOP_P_PROTO = 0.95
TOP_K_PROTO = 20

# Do not add miner-only logits processors here. A presence/repetition penalty
# changes the sampling policy; unless the validator also recomputes and verifies
# the same penalized old logprobs, PPO/DAPO trains against the wrong pi_old.

# A prompt that entered the training batch is ineligible for B_BATCH for
# the next N windows (= training steps). Forces curriculum rotation so
# the policy has time to shift between reuses.
# v2.3 + OpenMathInstruct (14M prompts): bumped from 200 to 1_000_000 so
# each prompt is effectively single-use across the lifetime of any
# realistic training run (1M windows ≈ 700 days at 5 blocks × 12s). The
# 14M-prompt env supplies enough fresh material without needing reuse.
BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000

# Cooldown is restored at startup from a run-keyed snapshot persisted to R2
# (see CooldownMap.export_state + service._restore_cooldown), so the FULL
# cooldown survives a restart without replaying the whole
# BATCH_PROMPT_COOLDOWN_WINDOWS history. COOLDOWN_REBUILD_LOOKBACK now only
# bounds the *gap replay* — the windows recorded between the last snapshot and
# the restart — so it just needs to comfortably exceed the snapshot cadence
# (CHECKPOINT_PUBLISH_INTERVAL_WINDOWS). It is also the scan size for the
# no-snapshot fallback. The old 300 default under-covered the 1M horizon and
# let prompts beyond 300 windows back re-enter the batch (replay exploit).
COOLDOWN_REBUILD_LOOKBACK = int(
    _os.environ.get("COOLDOWN_REBUILD_LOOKBACK", "2000")
)

# Identity the cooldown snapshot is keyed by. Stable across restarts of the
# SAME training run (successive checkpoints share it); change it when starting a
# fresh training so the cooldown resets to zero — a fresh model must be allowed
# to re-see every prompt.
TRAINING_RUN_ID = (
    _os.environ.get("RELIQUARY_TRAINING_RUN_ID", "default").strip() or "default"
)

# How often (in windows) to persist the cooldown snapshot, INDEPENDENT of the
# checkpoint-publish cadence. Publishing can stall (training starvation, HF
# publish failures) while windows keep advancing, which would let the snapshot
# fall arbitrarily far behind current_window — beyond what the gap replay
# (COOLDOWN_REBUILD_LOOKBACK) can recover, re-opening the replay exploit. A small
# fixed cadence keeps snapshot_window within this many windows of current.
COOLDOWN_SNAPSHOT_INTERVAL_WINDOWS = max(
    1, int(_os.environ.get("COOLDOWN_SNAPSHOT_INTERVAL_WINDOWS", "10"))
)

# Per-window prompt range (anti pre-curation). Each window, miner and
# validator derive the same contiguous slice of the prompt index space from
# the shared per-window randomness; once enforcement is armed only
# submissions whose prompt_idx falls in the slice are accepted. A static or
# shared bank of pre-curated prompts then lands in-range only
# ~PROMPT_RANGE_SIZE/len(env) of windows. See reliquary/shared/prompt_range.py.
PROMPT_RANGE_SIZE = int(_os.environ.get("PROMPT_RANGE_SIZE", "5000"))

# Window number from which the validator hard-enforces the prompt range.
# Below this window the slice is NOT enforced (current behavior, no rejects),
# so the upgraded miner client can ship ahead of the cutover. The default is
# a "never enforce" sentinel: set PROMPT_RANGE_ENFORCE_FROM_WINDOW=N* to the
# agreed cutover window AFTER the gated client is released and announced,
# otherwise un-upgraded miners are rejected ~every window.
PROMPT_RANGE_ENFORCE_FROM_WINDOW = int(
    _os.environ.get("PROMPT_RANGE_ENFORCE_FROM_WINDOW", str(2 ** 63 - 1))
)

# Per-rollout content dedup horizon. Independent of and strictly longer
# than the prompt cooldown: cooldown lets a prompt come back for fresh
# content, the hash set blacklists the specific (tokens) of every rollout
# already trained on. Startup rebuild is bounded for restart latency;
# operators can widen HASH_DEDUP_RETENTION_WINDOWS when they explicitly
# want a longer replay horizon and are willing to pay the R2 scan cost.
HASH_DEDUP_RETENTION_WINDOWS = int(
    _os.environ.get("HASH_DEDUP_RETENTION_WINDOWS", "300")
)

# Max submissions any single hotkey can send per window. Counter resets at
# every new window (on batcher swap). Excess submissions are HTTP-rejected
# as RATE_LIMITED before touching the validation pipeline. 8 matches B_BATCH
# — one slot per prompt a hotkey can credibly win in a window.
MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8

# Per-hotkey cap on BAD_ENVELOPE_SIGNATURE rejects per window. The
# envelope-signature gate (PR #35) deliberately does NOT bump
# ``_per_window_counts`` on bad-sig rejects so an anonymous spoofer
# cannot drain a victim's legitimate quota by spamming bogus packets
# under the victim's hotkey — that anti-DoS property is preserved.
#
# This cap closes a follow-on side-channel discovered in the wild: the
# zero-quota bad-envelope channel was being used by the LEGITIMATE
# hotkey owner to warm HTTP/1.1 keep-alive connections at zero quota
# cost (fire ~24 bogus POSTs to prime sockets, then ride the warm
# connections for the real signed POSTs and gain a ~20-30 ms RTT edge
# on the seal-trigger race against honest single-instance miners).
#
# Two defences combine in the fix:
#   1. ``Connection: close`` is set on every BAD_ENVELOPE_SIGNATURE
#      response, so the warm-up cannot happen (server tears the socket
#      down immediately; attacker pays handshake on the next packet).
#   2. This per-hotkey cap bounds bandwidth and verdict-ring noise even
#      against an attacker who doesn't care about priming — past the cap
#      the response is still BAD_ENVELOPE_SIGNATURE but the verdict is
#      no longer appended to the per-hotkey ring (which would otherwise
#      let a spoofer flood the victim's ``/verdicts/{hotkey}`` history
#      with junk that displaces legitimate entries).
#
# Crucially the per-hotkey rate-limit quota is never moved by these
# rejects, so PR #35's invariant holds end-to-end: an anonymous spoofer
# firing N bad packets against victim V's hotkey writes at most
# ``MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW`` verdict-ring entries and
# burns N handshakes, but V's full legitimate quota is untouched.
#
# Cap of 2 is low because honest miners have no reason to emit multiple
# bad envelopes per window — anything beyond a single accidental signing
# bug strongly suggests intent.
MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW = 2

# When True, /submit verifies ``envelope_signature`` before any per-hotkey
# rate-limit increment, and rejects unsigned / malformed / wrong-signer
# requests as BAD_ENVELOPE_SIGNATURE. This closes the trivial DoS where
# any caller can spam 8 unsigned packets claiming a victim's
# ``miner_hotkey`` and exhaust the per-window counter — locking the real
# miner out of the slot for the rest of the window. See
# ``reliquary.protocol.signatures.build_envelope_binding`` and the PR
# that introduced this flag for the full vector.
#
# Set to False ONLY for a rolling miner upgrade window: once all live
# miners are publishing envelope sigs, set to True (default). The
# False path is the pre-PR behaviour and remains DoS-exposed.
import os as _os
ENFORCE_ENVELOPE_SIGNATURE = _os.environ.get(
    "RELIQUARY_ENFORCE_ENVELOPE_SIGNATURE", "1"
).strip().lower() not in ("0", "false", "no", "off", "")

# Recompute the exact root emitted by existing wire-v1 miners. The check ships
# in shadow mode because privately maintained miners may have copied or
# reimplemented the historical serializer. Once live telemetry demonstrates
# byte parity, operators can enable rejection without changing miner payloads
# or signatures.
LEGACY_MERKLE_ROOT_ENFORCE = _os.environ.get(
    "RELIQUARY_LEGACY_MERKLE_ROOT_ENFORCE", "false"
).strip().lower() in ("1", "true", "yes", "on")

# How many drand-quicknet rounds backward of the validator's current round
# the batcher accepts on the ``drand_round`` field. Default = 0: strict
# equality. The miner must attach the drand round currently in progress
# at HTTP arrival.
#
# Why zero is safe now
# --------------------
# The two reasons the tolerance was widened in earlier iterations are
# both eliminated by the v2.3 fixes on this branch:
#
#   * Worker-side dequeue lag (PR #31 → tol=10). The seal/drand re-check
#     used to run on the worker, minutes after arrival under GRAIL queue
#     backpressure. Removed by the arrival-time refactor — drand is now
#     checked only once at HTTP arrival, against the middleware-stamped
#     ``t_arrival``.
#   * RTT boundary crossing (commit 8b7f483 → tol=1). A miner firing at
#     t=2.99 s of round R would land at the validator at t=3.00 s of
#     R+1. Well-behaved miners are expected to apply a small boundary-
#     safety margin client-side and sleep past the boundary if their
#     corrected clock is within a few hundred ms of one. With both
#     sides NTP-synced and the miner respecting the safety window, no
#     honest submission should land in the wrong drand round.
#
# Anything > 0 here opens an antedating window: an attacker could claim
# a slightly-earlier chronological tier than they actually earned. With
# zero, the only path to a slot is to actually be there in time —
# matches the original v2.3 design intent. Operators can re-widen via
# the ``DRAND_ROUND_BACKWARD_TOLERANCE`` env var if their cross-continent
# RTT profile justifies it (e.g. validators in EU serving miners in AU
# may need 1-2 to absorb 100-300 ms RTT spillover around a boundary).
# Tests pin specific values explicitly via
# ``GrpoWindowBatcher(drand_round_backward_tolerance=...)``.
#
# Forward direction stays zero (FUTURE_ROUND is unrecoverable: a miner
# that attaches round R+1 hasn't seen σ_{R+1} yet, so they're cheating).
DRAND_ROUND_BACKWARD_TOLERANCE = int(
    _os.environ.get("DRAND_ROUND_BACKWARD_TOLERANCE", "0")
)

# Bootstrap phase: first BOOTSTRAP_WINDOWS of a new subnet/checkpoint use
# relaxed thresholds to keep the batch filling while miner pop + env
# coverage are thin.
BOOTSTRAP_WINDOWS = 100

# First on-chain block at which this subnet deployed v2. Used to
# determine bootstrap eligibility. Set at the coordinated cutover.
SUBNET_START_BLOCK = 0

# ────────────────  v2.1 BATCH-DRIVEN WINDOWS  ────────────────

# Safety-net timeout: a window auto-seals after this many seconds even
# if fewer than B valid submissions have landed. The unused slots burn.
# Set generously — this is a backstop, not the cadence.
WINDOW_TIMEOUT_SECONDS = 7200

# Local JSON path for validator state (window_n counter + checkpoint_n).
# Resolved relative to the CWD if not absolute.
CHECKPOINT_STATE_PATH_DEFAULT = "reliquary/state/checkpoint.json"

# Local directory for staged checkpoint files before R2 upload.
CHECKPOINT_STAGING_DIR_DEFAULT = "reliquary/state/checkpoints"

# ────────────────  SCORING  ────────────────

# EMA smoothing factor for miner score. 2/(N+1) with N=72 (the EMA history depth).
# gives a ~25-window half-life — a miner that stops contributing loses
# half their score in ~25 windows.
EMA_ALPHA = 2.0 / (72 + 1)  # ≈ 0.0274

# ────────────────  GRPO TRAINING (v2.1)  ────────────────

# Learning rate for AdamW. RL fine-tuning on pretrained LLMs is sensitive;
# too high = collapse. Empirical drift measurement (scripts/measure_sketch_drift.py)
# showed 5e-7 produced a sketch delta of ~600 (≈10 % of the 6000 sketch
# tolerance) over 50 training steps — effectively indistinguishable from the
# base model, which also means stale-model cheaters pass GRAIL. Matched
# DAPO / R1-Zero-scale literature (1e-6 to 5e-6) by bumping to 5e-6.
LEARNING_RATE = 5e-6

# PPO clip range. Standard in GRPO/RLHF literature.
PPO_CLIP_EPSILON = 0.2

# KL penalty weight (DeepSeek's GRPO default). Keeps π_new close to the
# frozen reference; too low → drift / mode collapse; too high → no learning.
KL_BETA = 0.04

# Max gradient norm before step — standard RL stability guard.
GRAD_CLIP_NORM = 1.0

# train_step micro-batching: cap on padded tokens (n_seqs × longest_seq) per
# forward/backward. Short rollouts pack together; a rollout longer than this
# runs alone (= the legacy one-at-a-time path), so peak memory never exceeds a
# single sequence of this length. Sized at the protocol completion cap.
MICROBATCH_MAX_PADDED_TOKENS = 32768

# Linear LR warmup for the first N training steps (= N windows sealed).
LR_WARMUP_WINDOWS = 10

# Cosine schedule end target (in windows). Chosen large so LR never
# actually reaches zero at normal cadence — effectively a slow decay.
LR_COSINE_MAX_WINDOWS = 10_000

# Default base model (HF repo id). Served as the reference for KL and the
# cold-start checkpoint.
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"

# ────────────────  WANDB TELEMETRY (opt-in, validator-only)  ────────────────

# Wandb project name used by validator-side telemetry. Operators can
# override with the WANDB_PROJECT env var.
WANDB_PROJECT = "reliquary-validator"

# Bumping this constant (or setting RELIQUARY_WANDB_VERSION) starts a
# fresh wandb run. Same value across restarts → wandb resumes the
# existing run (resume="allow").
WANDB_TRAINING_VERSION = "v1"

# ────────────────  BEHAVIOURAL VALIDATORS  ────────────────
# Thresholds calibrated in the original grail repo against ~430k honest
# cross-GPU / cross-attn / cross-batch trials with 0 % false-positive
# rate. Do not re-tune without the same empirical setup.

# Minimum probability the model must have assigned to EOS at the position
# that produced it. Below this threshold, the rollout is presumed to be
# artificially truncated (a miner truncating mid-reasoning to lock in a
# favourable partial output). Upstream grail uses 0.02; we lowered to 0.01
# after Qwen-family + T_PROTO=0.9 prod logs showed honest EOS clustering just
# below 0.02. Mid-reasoning forgery still fails (p_stop typically < 0.001).
MIN_EOS_PROBABILITY = 0.01

# LogprobValidator: max allowed median importance-sampling deviation
# across K=CHALLENGE_K positions. dev_i = exp(|model_lp - miner_lp|) - 1.
# Reverted to GRAIL upstream's 0.10 (calibrated at 0% FP, ~50% headroom
# over the worst honest case). The previous 0.01 was tightened against
# same-stack miners observed at ~0.00013 median dev, but cross-stack
# honest drift (transformers 4.x miner ↔ 5.x validator) sits around
# 0.03-0.04 and was getting falsely rejected. 0.10 still flags clearly
# stale or forged checkpoints (cheater drift grows quickly past 0.10),
# while making the network functional for the majority of honest setups.
LOGPROB_IS_EPS = 0.10

# DistributionValidator: chosen-token probability thresholds. A "chosen
# token" is the token the miner sampled at step t; its probability under
# the validator's model (at the protocol temperature) is
# p_t = softmax(logits_{t-1} / T)[token_t].
SAMPLING_MIN_STEPS = 30         # completion must be at least this long
SAMPLING_LOW_P = 0.10           # prob <= this → "low" chosen token
SAMPLING_HIGH_P = 0.90           # prob >= this → "high" chosen token
SAMPLING_MEDIAN_LOW_MAX = 0.30  # median chosen prob must be above
SAMPLING_LOW_Q10_MAX = 0.025    # 10th-percentile must be above

# OpenMath final-answer tamper guard. The reward parser keys off the last
# \boxed{...} content; swapping a few tokens there flips the reward without
# moving median/q10. Tampered tokens land at p ≈ 1/vocab (~10⁻⁵ or lower)
# under the validator's forward; honest low-confidence answer tokens stay
# above ~10⁻³. Threshold sits in the gap.
BOXED_ANSWER_MIN_PROB = 0.001

# ────────────────  CODE EXECUTION GRADER  ────────────────

# Path to the Unix domain socket the grader server listens on.
# Default lives in /tmp so both validator and grader processes can reach it.
GRADER_SOCKET_PATH = "/tmp/reliquary-grader.sock"

# Number of warm gVisor workers in the grader pool. Sized to handle
# M_ROLLOUTS in parallel for a single submission with headroom for
# concurrent submissions. Increase for high-throughput validators.
GRADER_POOL_SIZE = M_ROLLOUTS

# Wall-clock timeout (seconds) for one structured OpenCode evaluation.
# Subprocess inside the sandbox is killed if it exceeds this. Tuned
# so that pathological miner code (infinite loops, slow algorithms)
# fails fast without blocking the queue.
GRADER_EVAL_TIMEOUT_SECONDS = 5

# How often (seconds) the server pings each worker via a no-op eval
# to detect zombies. Triggers respawn if a worker fails to respond.
GRADER_HEALTH_CHECK_INTERVAL_SECONDS = 30

# Token authenticity: a completion token whose chosen probability collapses
# below this while the model's argmax sits at >= TOKEN_AUTH_ARGMAX_CONF was not
# sampled — it was injected. Calibrated on 550k honest vLLM->HF tokens (floor
# 3.5e-7); measured injections <= 1e-13.
TOKEN_AUTH_THRESHOLD = 1e-8
TOKEN_AUTH_ARGMAX_CONF = 0.99
# Shadow mode: compute + log the check without rejecting. Flip to True once prod
# shadow logs confirm zero false positives.
TOKEN_AUTH_ENFORCE = True

# All-token argmax-gated authenticity gate: a chosen completion token below this
# threshold while the model's argmax is >= the confidence bound is rejected.
# Enforced live; the shadow telemetry counters are retained for monitoring.
ALL_TOKEN_AUTH_SHADOW_THRESHOLD = 1e-5
ALL_TOKEN_AUTH_SHADOW_ARGMAX_CONF = TOKEN_AUTH_ARGMAX_CONF
ALL_TOKEN_AUTH_ENFORCE = True

# OpenCode semantic-token authenticity shadow gate. Generic token auth catches
# near-impossible injections, and numeric auth catches many literal edits, but
# plausible code edits can live at probabilities far above 1e-10. In shadow
# mode, flag low-probability tokens that fall inside AST-sensitive code spans
# (operators, boolean/keyword values, return/index/literal regions). Keep this
# disabled until honest OpenCode calibration proves an acceptable false-positive
# rate.
CODE_SEMANTIC_AUTH_THRESHOLD = 0.001
CODE_SEMANTIC_AUTH_ARGMAX_CONF = TOKEN_AUTH_ARGMAX_CONF
CODE_SEMANTIC_AUTH_ENFORCE = False

# ──────────────── FORCED-SEED SAMPLING ────────────────
# Domain separation for the per-position public uniform u_{i,t}.
FORCED_SEED_DOMAIN = "reliquary-forced-seed-v1"
# A position counts toward the seed-consistency check only if its warped max
# probability is below this (i.e. the forced draw actually chooses the token).
FORCED_SEED_STOCHASTIC_MAXPROB = 0.99
# Reject a group whose stochastic-position match rate is below this floor.
# Keep the wire-v1 acceptance floor unchanged during the telemetry-only
# rollout. Live 2026-07-14 data supports a future 0.90 candidate, but changing
# acceptance policy belongs to the announced miner/protocol cutover after the
# exact-CDF shadow gate has been calibrated across implementations.
FORCED_SEED_CONSISTENCY_FLOOR = 0.80
# Below this many stochastic positions in a group, the gate abstains (accepts)
# rather than risk a false reject on thin signal.
FORCED_SEED_MIN_STOCH_POSITIONS = 30
# Per-rollout hardening: the group-average floor dilutes a partial swap (7
# honest rollouts at ~0.96 + 1 curated at ~0.60 still passes the group floor).
# Reject a
# group if ANY single rollout with enough stochastic positions falls below this
# per-rollout floor. Set lower than the group floor to absorb the higher
# single-rollout variance (empirical single-rollout: honest 0.94-1.0, non-forced
# 0.52-0.65); shadow-only until calibrated on the live floor.
FORCED_SEED_ROLLOUT_FLOOR = 0.75
# A single rollout is judged only if it carries at least this many stochastic
# positions; below it the per-rollout check abstains (never false-reject a
# short / peaked honest rollout).
FORCED_SEED_ROLLOUT_MIN_STOCH = 20
# Candidate tolerance for exact per-position CDF interval verification.
# A mismatch farther than this from the submitted token's validator-computed
# interval is a hard mismatch rather than numerical boundary ambiguity. The
# exact gate ships in telemetry mode until live incremental-vs-teacher-forced
# distance data establishes a consensus-safe epsilon.
FORCED_SEED_CDF_BOUNDARY_EPSILON = 0.002
FORCED_SEED_CDF_ENFORCE = _os.environ.get(
    "FORCED_SEED_CDF_ENFORCE", "false"
).strip().lower() in ("1", "true", "yes", "on")
# Master switch for forced-seed ENFORCEMENT. False = shadow (compute + log the
# consistency score, never reject); True = enforce (reject a group / rollout
# below the floors). Ships True so merging the branch arms the gate directly --
# merge ONLY once miners run the forced-seed client, else honest legacy miners
# are rejected. Env override (FORCED_SEED_ENFORCE=false) is a no-redeploy kill
# switch; any non-truthy value disables (never crashes, never auto-arms garbage).
FORCED_SEED_ENFORCE = _os.environ.get(
    "FORCED_SEED_ENFORCE", "true"
).strip().lower() in ("1", "true", "yes", "on")
# Wire-advertised on BatchSubmissionRequest.protocol_version by clients that
# sample from the forced stream (0 = legacy/pre-forced-seed). Lets the operator
# track adoption in the shadow window before arming enforcement.
FORCED_SEED_PROTOCOL_VERSION = 1
