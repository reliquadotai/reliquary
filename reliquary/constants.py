"""GRAIL Protocol Constants.

Immutable values that all network participants must agree on.
No os.getenv() overrides. Changes require coordinated deployment.
"""

# ────────────────  GRAIL PROOF VERSION  ────────────────

GRAIL_PROOF_VERSION = "v5"

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

# Sketch tolerance at position 0. Calibrated against a 10×30 cheater-curve
# sweep (scripts/cheater_curve_threshold.py) where a frozen base miner
# faces a freshly-trained validator: 1000 catches a 1-step-stale cheater
# with 75 % probability and 95 %+ from step 10 onwards, with 0 % false
# positives same-GPU. Subnet currently in test phase — miners are advised
# to use the same card as the validator (H200) until cross-GPU honest
# noise is measured. See docs/mining.md.
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

# Network-wide protocol cap on completion length.
MAX_NEW_TOKENS_PROTOCOL_CAP = 8192

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
TRAINING_QUARANTINE_MAX_MEAN_COMPLETION_LENGTH = 4096
TRAINING_QUARANTINE_MAX_SINGLE_COMPLETION_LENGTH = 7000
TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_ROLLOUTS = 4
TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_GROUPS = 3
TRAINING_QUARANTINE_REJECT_SPIKE_MIN = 32
TRAINING_QUARANTINE_REWARD_SHAPE_MIN_GROUPS = 2
TRAINING_QUARANTINE_LONG_ZERO_TAIL_MIN_LENGTH = 2048

# Soft cap on per-hotkey entries persisted to ``archive["rejected"]`` per
# window. Beyond this, ``reject_counts`` still increments but no metadata is
# appended — protects the R2 payload size against a flood of garbage
# submissions from a single attacker.
REJECTED_LIST_CAP_PER_HOTKEY = 5

# ────────────────  GRPO BATCHING  ────────────────

# Default HTTP port the validator listens on for miner submissions.
VALIDATOR_HTTP_PORT = 8888

# Hard cap on expensive proof-verification attempts admitted per window.
# This counts submissions reserved for GRAIL, not only submissions that pass.
# Failed proofs are deliberately not refunded: otherwise an attacker can make
# rejected spam free and keep the validator's GPU queue saturated.
MAX_PROOF_CANDIDATES_PER_WINDOW = 32

# After the B-th distinct prompt records a seal-trigger drand round, admit only
# a small tail of same-round stragglers. The hard window cap above remains the
# outer bound; this cap protects the boundary fair-split extension from turning
# one drand burst into an unbounded proof backlog.
MAX_POST_TRIGGER_PROOF_CANDIDATES = 8

# Safety timeout for the seal extension's submit-queue drain phase. Once the
# trigger drand round has expired, the validator waits briefly for already
# admitted trigger-round candidates to finish GRAIL; after this timeout it seals
# anyway so a slow or constantly refilled queue cannot freeze checkpoints.
MAX_SEAL_QUEUE_DRAIN_SECONDS = 20.0

# Liveness poll interval while an OPEN validator window waits for either a
# normal seal trigger or an exhausted proof-admission queue. This prevents a
# window from waiting WINDOW_TIMEOUT_SECONDS after all admitted proof work has
# drained but fewer than B valid submissions survived validation.
PROOF_ADMISSION_STALL_POLL_SECONDS = 0.5

# Active environment name (resolved by reliquary.environment.load_environment).
ENVIRONMENT_NAME = "openmathinstruct"

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

# Number of rollouts per submission (= size of each GRPO group).
M_ROLLOUTS = 8

# Training batch size — the first B valid in-zone submissions (FIFO by
# TCP arrival, distinct prompts, not in cooldown) feed the GRPO step.
B_BATCH = 8

# Sampling temperature fixed at protocol level. Miners who use a different
# T would produce samples from a different distribution → biased GRPO
# gradient. Value chosen in the GRPO-friendly range (non-zero).
T_PROTO = 0.9

# Top-p and top-k for sampling (fixed alongside T_PROTO).
TOP_P_PROTO = 1.0
TOP_K_PROTO = 0

# A prompt that entered the training batch is ineligible for B_BATCH for
# the next N windows (= training steps). Forces curriculum rotation so
# the policy has time to shift between reuses.
# v2.3 + OpenMathInstruct (14M prompts): bumped from 200 to 1_000_000 so
# each prompt is effectively single-use across the lifetime of any
# realistic training run (1M windows ≈ 700 days at 5 blocks × 12s). The
# 14M-prompt env supplies enough fresh material without needing reuse.
BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000

# Validator startup: cap the number of R2 archives scanned to rebuild
# CooldownMap. Independent of BATCH_PROMPT_COOLDOWN_WINDOWS — that
# constant can be astronomically large for one-shot semantics, but R2
# rebuild must stay bounded in elapsed wall time. Defaulting to 300 keeps
# prod restarts fast while still replaying recent windows; operators can
# widen it via COOLDOWN_REBUILD_LOOKBACK for one-off backfills or recovery
# runs. Older entries may be missing from the rebuilt in-memory cooldown
# map, which is safe: prompt cooldown is curriculum rotation, and the
# hash blacklist below still rejects re-submission of the same token
# sequence within its own retention horizon.
COOLDOWN_REBUILD_LOOKBACK = int(
    _os.environ.get("COOLDOWN_REBUILD_LOOKBACK", "300")
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

# Max GRAIL-validated submissions retained per prompt per window. Once this
# cap is reached for a prompt, further submissions for that prompt are
# rejected as PROMPT_FULL before the heavy verify. Bounds the validator's
# GPU cost when many miners attack the same prompt — combined with the
# per-hotkey cap above, worst-case GRAIL load per window is
# MAX_SUBMISSIONS_PER_PROMPT × min(|env|, MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW × n_hotkeys).
MAX_SUBMISSIONS_PER_PROMPT = 10

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

# Linear LR warmup for the first N training steps (= N windows sealed).
LR_WARMUP_WINDOWS = 10

# Cosine schedule end target (in windows). Chosen large so LR never
# actually reaches zero at normal cadence — effectively a slow decay.
LR_COSINE_MAX_WINDOWS = 10_000

# Default base model (HF repo id). Served as the reference for KL and the
# cold-start checkpoint.
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

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
# after Qwen3-4B + T_PROTO=0.9 prod logs showed honest EOS clustering just
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
