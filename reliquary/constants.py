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
import math as _math
import os as _os

from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE

ATTN_IMPLEMENTATION = _os.environ.get("GRAIL_ATTN_IMPL", "flash_attention_2")

PROTOCOL_PROFILE_ID = ACTIVE_PROTOCOL_PROFILE.profile_id
PROTOCOL_MODEL_ID = ACTIVE_PROTOCOL_PROFILE.model_id
PROTOCOL_MODEL_REVISION = ACTIVE_PROTOCOL_PROFILE.model_revision
PROTOCOL_VERSION = ACTIVE_PROTOCOL_PROFILE.protocol_version
# Draw ordering among equal-difficulty candidates. Part of the signed profile
# contract, so a miner reads the rule it competes under; None on v2.
PROTOCOL_THROUGHPUT_TIEBREAK = ACTIVE_PROTOCOL_PROFILE.throughput_tiebreak
PROTOCOL_GENERATION_CONTRACT = (
    ACTIVE_PROTOCOL_PROFILE.to_generation_contract()
)

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

# Grace period = block variance + upload latency. Profiles pin this explicitly
# so a miner and validator cannot silently disagree after a timing change.
UPLOAD_GRACE_PERIOD = ACTIVE_PROTOCOL_PROFILE.upload_grace_seconds
assert UPLOAD_GRACE_PERIOD == BLOCK_TIME_VARIANCE + NETWORK_UPLOAD_LATENCY

# A miner may commit a completed submission before the auction deadline and
# finish uploading the matching body during this bounded reveal interval.  The
# grace never extends generation: only a signed Merkle root and exact body size
# received before the collection cutoff can arm it.
SUBMISSION_UPLOAD_GRACE_SECONDS = float(UPLOAD_GRACE_PERIOD)

# ────────────────  ROLLOUT GENERATION  ────────────────

# Per-environment generation ceilings are part of the signed profile contract.
# The schema ceiling remains the maximum across environments; every
# environment-aware termination path must use ``max_new_tokens_for_environment``.
MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV = {
    environment: profile.max_new_tokens
    for environment, profile in ACTIVE_PROTOCOL_PROFILE.environments.items()
}
MAX_NEW_TOKENS_PROTOCOL_CAP = max(
    MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV.values()
)


def max_new_tokens_for_environment(environment: str) -> int:
    """Return the active cap; custom legacy/test environments use schema max."""

    return MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV.get(
        environment,
        MAX_NEW_TOKENS_PROTOCOL_CAP,
    )

# Budget-Forced Termination (BFT): if a rollout has not emitted </think> by
# BFT_THINKING_BUDGET tokens, the miner appends BFT_FORCE_TEMPLATE and samples
# the answer in BFT_ANSWER_BUDGET more tokens, so it terminates with a real
# (gradeable) answer instead of an unparseable thinking truncation. H200 sweeps
# on real OpenMathInstruct prompts found 2048/512 to match 2048/256 reward
# (5/6) with better EOS rate, while 4096/256 was slower and lower reward.
_MATH_PROFILE = ACTIVE_PROTOCOL_PROFILE.environments["openmathinstruct"]
_MATH_BFT_PROFILE = _MATH_PROFILE.bft
MATH_ANSWER_FORMAT = _MATH_PROFILE.answer_format
if MATH_ANSWER_FORMAT not in ("boxed", "boxed_or_trailing_number"):
    raise ValueError("openmathinstruct profile must declare an answer format")
BFT_ENABLED = _MATH_BFT_PROFILE is not None
# 2026-07 Qwen3.5-4B behavior study (held-out OMI, vLLM): the 4B thinks a median
# of 3766 / p90 11297 tokens before </think>; a 2048 budget would force-cut ~45%
# of its *productive* reasoning. A 32768-budget probe on the rollouts that don't
# finish at 16384 found only 23% just needed more room (62% of those correct)
# while 77% never conclude even at 32k (43% detectable loops). So the answer is a
# HIGH cap (not unbounded): ~15.6k captures ~95% of productive reasoning while
# bounding looper compute. Sized so thinking + answer + a force-span margin
# lands exactly on MAX_NEW_TOKENS_PROTOCOL_CAP (16384). WIRE CONSTANT — miner generates to it and the verifier
# checks against it, so both sides must ship the same value (coordinated deploy).
BFT_THINKING_BUDGET = (
    _MATH_BFT_PROFILE.thinking_budget if _MATH_BFT_PROFILE else 0
)
BFT_ANSWER_BUDGET = (
    _MATH_BFT_PROFILE.answer_budget if _MATH_BFT_PROFILE else 0
)
BFT_FORCE_TEMPLATE = "</think>\n\nFinal Answer: \\boxed{"
# Clean-cap vs forced-answer at the budget. True (legacy): an unterminated rollout
# is force-closed with BFT_FORCE_TEMPLATE + a sampled answer. False (clean-cap):
# it is left truncated and grades as bad_termination (an honest failure — the
# verifier already rejects an unterminated, non-forced, sub-protocol-cap rollout).
# On the 2B the forced answer was a coin-flip that dominated reward variance; the
# 4B is capable enough that within-group variance exists without it, so clean-cap
# is the less-polluted signal. Kept True for a STAGED rollout: bump the budget to
# 16k first (contract-consistent), then flip this to False once miners ship it.
# WIRE-AFFECTING — flipping requires a coordinated miner+validator deploy.
BFT_FORCE_ANSWER = bool(
    _MATH_BFT_PROFILE and _MATH_BFT_PROFILE.force_answer
)
# DAPO-style overlong filtering, now OFF by default on every profile. Masking
# made non-termination gradient-free while terminated-and-wrong stayed punished
# in proportion to length — the inverted incentive behind the ckpt84 rumination
# collapse (hard-prompt EOS 84%→50%). Superseded by the soft overlong
# punishment below; these flags remain as emergency kill switches.
#
# Masked rollouts STAY in the group mean/std (baseline unchanged, DAPO Eq. 9) and
# in the sigma gate / auction / emission — the economic layer keeps scoring them,
# so a miner cannot dodge a wrong answer by not terminating. Because the zeros
# stay in the mean, the surviving terminated-and-correct rollouts get LARGER
# advantages: the pressure is toward finishing correctly, not toward rambling.
# Validator-only TRAINING controls — no miner/wire change. The aggregate flag
# remains an explicit experiment switch for both environments. Whether a rollout
# is masked is read from validator metadata via training._is_masked_from_loss,
# never inferred from an advantage of 0.0.
MASK_BUDGET_ENDED_FROM_LOSS = (
    _os.environ.get("RELIQUARY_MASK_BUDGET_ENDED_FROM_LOSS", "0")
    not in ("0", "false", "False")
)
MASK_MATH_FORCED_FROM_LOSS = (
    _os.environ.get("RELIQUARY_MASK_MATH_FORCED_FROM_LOSS", "0")
    not in ("0", "false", "False")
)
MASK_CODE_TRUNCATED_FROM_LOSS = (
    _os.environ.get("RELIQUARY_MASK_CODE_TRUNCATED_FROM_LOSS", "0")
    not in ("0", "false", "False")
)
# The global cap must leave room for a forced completion (thinking + force-span +
# answer). Guard against a future budget bump that would make forced rollouts
# exceed the schema cap and get rejected.
assert (
    max_new_tokens_for_environment("openmathinstruct")
    > BFT_THINKING_BUDGET + BFT_ANSWER_BUDGET
), (
    "the Math protocol cap must exceed BFT_THINKING_BUDGET + "
    "BFT_ANSWER_BUDGET to fit the forced answer's force-template span"
)

# v4+ ships a raw-completion protocol: prompts are encoded with no chat template
# at all. This cannot be inferred from the tokenizer, which is the trap — a
# "-Base" repo ships the family tokenizer config, chat template included, so
# Qwen3-4B-Base DECLARES one it was never trained on. Left to auto-detection the
# v4 prompt would silently take the chat path, wrapping the problem in role
# markers the base model has no learned behaviour for and (because the template
# mentions enable_thinking) opening a <think> block on a model chosen precisely
# because it never opens one. One switch drives both encode_prompt and
# render_canonical_prompt so the encoded tokens and prompt_content_sha256 can
# never describe different prompts.
RAW_COMPLETION_PROMPTS = ACTIVE_PROTOCOL_PROFILE.prompt_encoding == "raw"
if ACTIVE_PROTOCOL_PROFILE.prompt_encoding not in ("raw", "chat_template"):
    raise ValueError("protocol profile declares an unknown prompt encoding")

# v4+ restricts the OMI manifest to the canonical `train-*` shards. The
# train_1M/2M/5M shards are curated subsets OF train, so including them
# duplicates 8,000,000 rows (36.4% of the index space) and draws the curated
# rows 2-4x too often. Changing this changes len(env), which is prompt-range
# consensus, so it is only safe at a profile cutover.
OMI_TRAIN_SHARDS_ONLY = PROTOCOL_VERSION >= 4

# Two-sided length reward shaping (applied to ADVANTAGES, not the σ-gate).
# Under-thinking side: a non-forced rollout that finished early
# (completion_length < SHAPE_LEN_FRAC · BFT_THINKING_BUDGET) and is wrong gets its
# advantage set to −SHAPE_PENALTY. Overlong side: a cap-truncated rollout gets
# the same penalty. These are validator-only objective controls, not generation
# or wire constants, so operators can calibrate them without changing miners.
# SHAPE_PENALTY = 0 disables shaping.
SHAPE_PENALTY = float(_os.environ.get("RELIQUARY_SHAPE_PENALTY", "0"))
if not _math.isfinite(SHAPE_PENALTY) or not 0.0 <= SHAPE_PENALTY <= 10.0:
    raise ValueError(
        "RELIQUARY_SHAPE_PENALTY must be finite and within [0, 10]"
    )
SHAPE_LEN_FRAC = float(_os.environ.get("RELIQUARY_SHAPE_LEN_FRAC", "0.5"))
if not _math.isfinite(SHAPE_LEN_FRAC) or not 0.0 < SHAPE_LEN_FRAC <= 1.0:
    raise ValueError(
        "RELIQUARY_SHAPE_LEN_FRAC must be finite and within (0, 1]"
    )

# Soft overlong punishment (DAPO Eq. 13), applied to TRAINING rewards only in
# training._training_rewards — the graded reward that drives the σ-gate, the
# auction and emission is untouched, so miner economics cannot move. A rollout
# that runs into the last OVERLONG_PENALTY_CACHE_TOKENS before its env cap pays
# a linear penalty reaching −factor at the cap, correct or not. 0 disables.
# 0.5 matches DAPO's penalty-to-correctness ratio: they use 1.0 on the 2-wide
# {−1,+1} reward scale, ours is [0,1].
OVERLONG_PENALTY_FACTOR = float(_os.environ.get(
    "RELIQUARY_OVERLONG_PENALTY_FACTOR",
    "0.5" if PROTOCOL_VERSION >= 3 else "0",
))
if (
    not _math.isfinite(OVERLONG_PENALTY_FACTOR)
    or not 0.0 <= OVERLONG_PENALTY_FACTOR <= 10.0
):
    raise ValueError(
        "RELIQUARY_OVERLONG_PENALTY_FACTOR must be finite and within [0, 10]"
    )
OVERLONG_PENALTY_CACHE_TOKENS = int(_os.environ.get(
    "RELIQUARY_OVERLONG_PENALTY_CACHE_TOKENS", "4096"
))
if OVERLONG_PENALTY_CACHE_TOKENS <= 0:
    raise ValueError(
        "RELIQUARY_OVERLONG_PENALTY_CACHE_TOKENS must be positive"
    )
# A BFT-forced rollout's grade comes from validator-injected tokens the policy
# never sampled — a coin flip that lands correct ~43% of the time (R2, v3 era).
# Crediting the policy for that flip is what kept rumination reinforced, so the
# forced rollout's base TRAINING reward is zeroed and it pays the full overlong
# penalty flat (its length is a protocol constant, not a choice). This is
# clean-cap semantics without the BFT_FORCE_ANSWER wire change; the miner is
# still paid for a lucky forced answer.
TRAIN_FORCED_REWARD_ZERO = (
    _os.environ.get(
        "RELIQUARY_TRAIN_FORCED_REWARD_ZERO",
        "1" if PROTOCOL_VERSION >= 3 else "0",
    )
    not in ("0", "false", "False")
)

# Cap/non-EOS truncation budget per submission. A single missing-EOS rollout
# can be an honest local max-token accident; repeated missing-EOS rollouts in
# one GRPO group are a sampling policy, not a rare exception, and have become
# the main manufactured-loser path.
MAX_TRUNCATED_PER_SUBMISSION = 1
BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION = 1
# Both environments keep the single-truncation ceiling. The former Code
# allowance of three was inconsistent across admission paths and was not safe
# under fractional rewards. V3 prices the one unknown outcome over the exact
# validator-known reward lattice.
# Code has no BFT to force termination, so the 4B loops to the cap on ~10% of
# code rollouts; at eight rollouts that puts two or more truncations in ~19% of
# HONEST groups, which a ceiling of one rejects wholesale. Three brings that to
# ~0.5%. The exactness objection to a larger allowance is answered by
# robust_truncation_utility pricing every JOINT assignment of the unknowns, so
# the manufactured-truncation guarantee does not weaken as this rises. Math
# stays at one: BFT terminates every rollout, so a truncated math rollout means
# a non-compliant client.
MAX_TRUNCATED_PER_SUBMISSION_BY_ENV: dict[str, int] = {
    "opencodeinstruct": 3,
}


def max_truncated_for_environment(
    environment: str,
    *,
    bootstrap: bool = False,
) -> int:
    if bootstrap:
        return BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION
    return MAX_TRUNCATED_PER_SUBMISSION_BY_ENV.get(
        environment,
        MAX_TRUNCATED_PER_SUBMISSION,
    )

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

# Hard ceiling on total grading attempts (reward computation) admitted per
# window — the anti-DoS / queue bound and the outer cap on GPU proof work.
# Counts every grading attempt that actually starts and is never refunded.
# Pending work reserves capacity separately, so a degenerate-reward flood cannot
# grow the bounded submit queue or saturate the grader pool while discarded
# queue items do not burn work that never ran.
# Admission remains at 64 started grading jobs per environment. It is consumed
# before deferred GRAIL proof, so coupling it to the smaller seal-time proof
# prefix would let two eight-submission hotkeys fill a v3 environment with
# forged commitments before any GPU authentication runs.
MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW = 64

# Seal-time GRAIL work is serial and an adversarial ranked prefix may fail one
# candidate after another. The attempt ceiling bounds cardinality; this second
# bound limits elapsed GPU time. It is checked between groups, so one in-flight
# group may finish after the budget expires.
MAX_PROOF_WALL_SECONDS = float(
    _os.environ.get("RELIQUARY_MAX_PROOF_WALL_SECONDS", "240")
)

# Retained submission payload bounds. JSON bytes are measured from the parsed
# Pydantic request before queue insertion. The per-environment ceiling is
# independent for Math and Code; with the canonical two-env mix the aggregate
# retained ceiling is 1 GiB. Python object overhead is higher than JSON size,
# hence the deliberately conservative limits relative to host RAM.
MAX_SUBMISSION_PAYLOAD_BYTES = int(
    _os.environ.get(
        "RELIQUARY_MAX_SUBMISSION_PAYLOAD_BYTES", str(64 * 1024 * 1024)
    )
)
MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY = int(
    _os.environ.get(
        "RELIQUARY_MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY",
        str(128 * 1024 * 1024),
    )
)
MAX_PENDING_SUBMISSION_BYTES_PER_ENV = int(
    _os.environ.get(
        "RELIQUARY_MAX_PENDING_SUBMISSION_BYTES_PER_ENV",
        str(512 * 1024 * 1024),
    )
)

if MAX_PROOF_WALL_SECONDS <= 0:
    raise ValueError("RELIQUARY_MAX_PROOF_WALL_SECONDS must be positive")
if MAX_SUBMISSION_PAYLOAD_BYTES <= 0:
    raise ValueError("RELIQUARY_MAX_SUBMISSION_PAYLOAD_BYTES must be positive")
if MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY < MAX_SUBMISSION_PAYLOAD_BYTES:
    raise ValueError(
        "per-hotkey pending byte cap must allow one maximum-size submission"
    )
if (
    MAX_PENDING_SUBMISSION_BYTES_PER_ENV
    < MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY
):
    raise ValueError("per-environment pending byte cap must cover hotkey cap")
# Absolute server-side bound across active and draining environment queues.
# Per-batcher reservation caps remain the primary bound; this is the final
# backstop during a window swap or prolonged GPU stall.
MAX_PENDING_PROOF_QUEUE_DEPTH = 64

# Reward admission runs before deferred GRAIL proof. Math grading benefits from
# bounded CPU parallelism; four Code group workers each fan eight rollouts over
# the 32-slot sandbox pool. These are validator-runtime capacities, not miner
# wire constants.
MATH_ADMISSION_WORKERS = int(
    _os.environ.get("RELIQUARY_MATH_ADMISSION_WORKERS", "8")
)
CODE_ADMISSION_WORKERS = int(
    _os.environ.get("RELIQUARY_CODE_ADMISSION_WORKERS", "4")
)
if not 1 <= MATH_ADMISSION_WORKERS <= 16:
    raise ValueError("RELIQUARY_MATH_ADMISSION_WORKERS must be within [1, 16]")
if not 1 <= CODE_ADMISSION_WORKERS <= 8:
    raise ValueError("RELIQUARY_CODE_ADMISSION_WORKERS must be within [1, 8]")

# A hotkey that repeatedly reaches the expensive proof path and then fails
# behavioural/integrity checks should not be allowed to consume the whole
# window's scarce proof budget. Allow a small number of misses for honest
# stack drift, then reject further proof admissions from that hotkey until the
# next window.
MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW = 2

# A coldkey may own many hotkeys, so a hotkey-only debt limit does not bound a
# single operator's seal-time GPU denial of service. Applied independently per
# environment after ownership is resolved from the window's metagraph snapshot.
MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW = 4

# Registered-hotkey admission cache. Registration changes at chain cadence, not
# submission cadence: refresh once per tempo-equivalent interval at a quiescent
# pre-window boundary. Request handling only reads the immutable snapshot.
REGISTERED_HOTKEY_CACHE_TTL_SECONDS = float(360 * BLOCK_TIME_SECONDS)
REGISTERED_HOTKEY_STALE_GRACE_SECONDS = float(
    2 * REGISTERED_HOTKEY_CACHE_TTL_SECONDS
)
REGISTERED_HOTKEY_REFRESH_TIMEOUT_SECONDS = 20.0

# After the B-th distinct prompt records a seal-trigger drand round, admit only
# a small tail of same-round stragglers. The hard window cap above remains the
# outer bound; this cap protects the boundary fair-split extension from turning
# one drand burst into an unbounded proof backlog.
MAX_POST_TRIGGER_PROOF_CANDIDATES = 8

# Safety timeout for the pre-seal drain phase. Auction environments stop new
# admission at the collection deadline, drain work that arrived before it, then
# freeze the pending population. Legacy mode uses the same bound after its seal
# trigger. A stuck queue cannot freeze checkpoints indefinitely.
MAX_SEAL_QUEUE_DRAIN_SECONDS = 60.0

# Auction reveals are already bounded by 64 receipts and 8/4 isolated workers.
# After upload grace closes, wait for the complete accepted population. If this
# emergency wall is ever reached, abort the whole window instead of training a
# timing-dependent partial snapshot.
AUCTION_ADMISSION_DRAIN_DEADLINE_SECONDS = 390.0

# Liveness poll interval while an OPEN validator window waits for either a
# normal seal trigger or an exhausted proof-admission queue. This prevents a
# window from waiting WINDOW_TIMEOUT_SECONDS after all admitted proof work has
# drained but fewer than B valid submissions survived validation.
PROOF_ADMISSION_STALL_POLL_SECONDS = 0.5

# Legacy-selector sparse-window liveness breaker. Production auction
# environments use the fixed collection deadline instead; these values remain
# for the emergency kill-switch path and any out-of-scope environment.
SPARSE_VALID_IDLE_SEAL_SECONDS = 300.0
SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS = 4
SPARSE_VALID_MAX_WINDOW_SECONDS = 900.0

# Fixed collection timing is versioned with the generation budget. We do not
# early-close: doing so would reward arrival rather than let the complete
# auction population compete under the advertised deadline.
WINDOW_COLLECTION_SECONDS = float(
    ACTIVE_PROTOCOL_PROFILE.collection_seconds
)

# Difficulty-auction v2: number of ranked-pass non-winners proven per window
# purely for the forensic auth gates (token-auth, distribution, forced-seed),
# which otherwise only ever run on the winners. Selection is keyed by a drand
# beacon fetched after the collection deadline, so miners cannot grind it.
FORENSIC_SAMPLE_PER_WINDOW = 2

# UID that receives unused slot emission budget (the burn address).
#
# Unset (the default) means "this validator's own uid", resolved from the
# metagraph by its own hotkey at weight-submission time. The historical hard 0
# was the SUBNET OWNER's uid, which is not a stable address: subnet ownership
# changes, and a uid can move on re-registration — a stale literal silently
# pays the wrong account every epoch. Set RELIQUARY_UID_BURN=<n> to pin an
# explicit target (0 restores the legacy owner-burn) without a code release.
_UID_BURN_RAW = _os.environ.get("RELIQUARY_UID_BURN", "").strip()
UID_BURN: int | None = int(_UID_BURN_RAW) if _UID_BURN_RAW else None
if UID_BURN is not None and UID_BURN < 0:
    raise ValueError("RELIQUARY_UID_BURN must be a non-negative uid")

# ────────────────  CONTINUOUS VALIDATION  ────────────────

# How often the validator polls for new state (seconds).
POLL_INTERVAL_SECONDS = 10

# ────────────────  WEIGHT SUBMISSION  ────────────────

# Submit weights when blocks_until_next_epoch <= this value. Tuned so all
# validators of a netuid land in the same ~20-block window (≈4 min on
# 12s/block) and read near-identical R2 archive snapshots, then converge
# to identical weights via the deterministic EMA replay.
EPOCH_SUBMIT_LEAD_BLOCKS = 20

# ────────────────  HUGGING FACE CHECKPOINT PUBLISHING  ────────────────

# How often to publish the current in-memory model to Hugging Face. Keep the
# serving behavior policy close enough to the training policy that PPO's ratio
# contract remains meaningful, while avoiding a multi-GB upload on every step.
# This also sets how many optimizer steps share one pi_old: verify_model is a
# frozen snapshot of the last published checkpoint, and miners generate from it,
# so publication IS the pi_old refresh.
#
# DAPO §4.1 samples a rollout batch of 512 prompts x 16 responses = 8,192
# sequences and consumes it in 16 gradient updates at mini-batch 512. v4's
# per-window batch is already exactly that mini-batch (2 envs x 16 prompts x 16
# rollouts = 512), so publishing every 16 windows reproduces their structure:
# 16 steps over 8,192 sequences from one pi_old. At 4 it is 4 steps over 2,048.
#
# The cost is drift: pi_theta moves 16 steps from pi_old instead of 4, so the
# clip does more work. That is what eps_high=0.28 is for; the backstop that
# STOPS a run drifting too far — PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD — is
# armed at 0.5 on v4. Watch train/ppo_ratio_outside_clip_ratio and tighten it
# from real telemetry once the run's normal late-interval level is known.
CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = int(_os.environ.get(
    "RELIQUARY_CHECKPOINT_PUBLISH_INTERVAL_WINDOWS",
    "16" if PROTOCOL_VERSION >= 4 else "4",
))
if CHECKPOINT_PUBLISH_INTERVAL_WINDOWS <= 0:
    raise ValueError(
        "RELIQUARY_CHECKPOINT_PUBLISH_INTERVAL_WINDOWS must be positive"
    )

# Default HF repo target for published checkpoints. Operator may
# override via --hf-repo-id CLI arg. Must be a writable repo id for
# the validator's HF token.
DEFAULT_HF_REPO_ID = "aivolutionedge/reliquary-sn"

# ────────────────  GRPO MARKET (v2)  ────────────────

# Minimum reward-std for a group to pass the zone filter. For binary
# Bernoulli rewards the 0.43 steady-state gate admits k ∈ [2, 6] for M=8
# (σ at k=2/6 ≈ 0.433); for continuous rewards it drops groups whose rollouts
# clustered too tight to carry meaningful GRPO signal.
#
# v4 lowers the gate to DAPO's dynamic-sampling intent: admit any group with a
# spread of correctness. For M=16 binary rewards the extreme non-degenerate
# groups k=1 and k=15 have σ = √(1/16·15/16) = 0.2421, so a 0.24 floor admits
# every k ∈ [1, 15] while the degenerate k=0 / k=16 (σ=0) stay rejected. The
# old 0.43 floor rejected k ∈ {1,2,3,13,14,15}, starving the trainer of exactly
# the rare-correct (hard) and near-all-correct (easy) prompts DAPO learns from.
#
# 0.24, not 0.0: a 0.0 floor would also admit near-degenerate CONTINUOUS (code)
# groups — fractional test-pass rewards clustered at e.g. 0.48-0.52 have a tiny
# but > 1e-8 σ — which carry no usable GRPO gradient. 0.24 keeps filtering
# those while passing every binary variance group. This is the pragmatic
# binary-faithful port; the fully faithful `metric=acc` (binarise correctness,
# drop all-same) is deferred.
#
# NOTE: the gate also governs admission/payment, so this widens what is PAID and
# softens the manufactured-variance surface — deliberate, see the v4 cutover
# plan; the auction value σ-weighting and forced-seed / distribution layers are
# the remaining defense. v3 keeps 0.43/0.33 so live economics are byte-identical.
SIGMA_MIN = 0.24 if PROTOCOL_VERSION >= 4 else 0.43
BOOTSTRAP_SIGMA_MIN = 0.22 if PROTOCOL_VERSION >= 4 else 0.33

# Number of rollouts per submission (= size of each GRPO group).
M_ROLLOUTS = ACTIVE_PROTOCOL_PROFILE.sampling.rollouts

# Maximum proven winners and uniform reward slots per active environment.
# Distinct prompt representatives feed GRPO.
#
# v4 doubles it alongside G. The two fix different errors: G shrinks the
# per-group advantage bias, B_BATCH shrinks the across-prompt gradient
# variance. Affordable because v4 rollouts are ~500 tokens against v3's 16,220
# — 4x the sequences at an eighth of the token budget. The break-even against
# today's load is a ~4,000-token median, which is roughly where DAPO's own
# length curve lands by end of run (their Fig. 7a), so this is the number to
# watch as training lengthens responses, NOT the slot count.
B_BATCH = 16 if PROTOCOL_VERSION >= 4 else 8

# (env_name, prompts_per_batch). Sum across entries = total prompts
# processed per optimizer step: 2 × B_BATCH prompts × M_ROLLOUTS sequences.
# v3: 16 × 8 = 128. v4: 32 × 16 = 512.
ENVIRONMENT_MIX: list[tuple[str, int]] = [
    ("openmathinstruct", B_BATCH),
    ("opencodeinstruct", B_BATCH),
]

# Auction-v3 deliberately narrows only the ranked GPU proof prefix: B_BATCH
# winners plus B_BATCH possible failed candidates per environment. Derived from
# B_BATCH rather than written as a literal, so a batch-size change cannot leave
# the ranked prefix unable to prove the slots it must fill. A v3 fleet
# qualifies this full ceiling plus the independent forensic budget.
MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW = (
    2 * B_BATCH if PROTOCOL_VERSION >= 3 else 64
)

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
T_PROTO = ACTIVE_PROTOCOL_PROFILE.sampling.temperature

# Top-p and top-k for sampling (fixed alongside T_PROTO).
TOP_P_PROTO = ACTIVE_PROTOCOL_PROFILE.sampling.top_p
TOP_K_PROTO = ACTIVE_PROTOCOL_PROFILE.sampling.top_k
DO_SAMPLE_PROTO = ACTIVE_PROTOCOL_PROFILE.sampling.do_sample

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

# Max submissions any single hotkey can send per window (per-env: one batcher
# per environment). Counter resets at every new window. Excess submissions are
# HTTP-rejected as RATE_LIMITED before touching the validation pipeline; a
# precommit reserves one unit and its reveal does not re-consume, so it is one
# unit per prompt either way.
#
# At exactly B_BATCH a hotkey can cover every prompt slot but has ZERO headroom:
# a single retry (dropped packet, transient reject) is the (B_BATCH+1)th and is
# rate-limited out. v4 doubles it to 2·B_BATCH for one full re-coverage of
# retry/improvement margin. Safe to widen because the EXPENSIVE seal-time GRAIL
# work is capped independently (MAX_RANKED_PROOF_ATTEMPTS = 2·B_BATCH, grading =
# MAX_PROOF_GRADING_ATTEMPTS), so this only widens cheap admission bandwidth,
# not GPU proof load. v3 stays at B_BATCH so live economics are byte-identical.
MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = (
    2 * B_BATCH if PROTOCOL_VERSION >= 4 else B_BATCH
)

# Signed upload precommits are tiny, but still bounded independently from the
# large-body proof queue.  A precommit consumes the same per-window hotkey quota
# as a direct submission, so abandoning an upload cannot create free receipt
# spam or reserve an unbounded number of deadline extensions.
MAX_PENDING_UPLOAD_PRECOMMITS_PER_ENV = MAX_PENDING_PROOF_QUEUE_DEPTH

# Process-isolated auction preparation. These hard per-candidate walls make the
# accepted receipt population drainable even when submitted math or code is
# pathological. They do not control GPU proof time, which has its own budget.
MATH_ADMISSION_WALL_SECONDS = 45.0
CODE_ADMISSION_WALL_SECONDS = 20.0
ADMISSION_PROCESS_MAX_TASKS = 128

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

# Armed production mechanism. The kill switch restores the legacy selection
# path without changing the public schema. Math and Code use the same auction;
# their independent resource accounting is enforced per environment.
DIFFICULTY_AUCTION_ENFORCE = _os.environ.get(
    "RELIQUARY_DIFFICULTY_AUCTION_ENFORCE", "1"
).strip().lower() not in ("0", "false", "no", "off", "")
DIFFICULTY_AUCTION_ENVIRONMENTS = (
    "openmathinstruct",
    "opencodeinstruct",
)

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

# Max graded pending submissions retained per prompt per window. Once reached,
# further submissions for that prompt are rejected as PROMPT_FULL before any
# deferred GPU proof. Auction logical dedup separately permits only one claim
# per (operator, prompt), so this cap bounds distinct-operator crowding.
MAX_SUBMISSIONS_PER_PROMPT = 10

# Legacy counterfactual retained for the emergency kill-switch path. Production
# auction environments emit an armed schema through the same historical field
# name for dashboard compatibility; this computation runs only when production
# auction enforcement is disabled for an environment.
DIFFICULTY_AUCTION_SHADOW_ENABLED = _os.environ.get(
    "RELIQUARY_DIFFICULTY_AUCTION_SHADOW_ENABLED", "1"
).strip().lower() not in ("0", "false", "no", "off", "")
DIFFICULTY_AUCTION_SHADOW_ENVIRONMENTS = ("openmathinstruct",)
DIFFICULTY_AUCTION_DELTA = 1.0

# Flat auction valuation: every in-zone group ranks (and records) the same
# value, so slot selection reduces to the sigma gate plus the throughput /
# arrival tie-breaks. Removes the k=2 value peak (the Sybil duel target), the
# manufactured-zero premium (+68% at k=6 under the peaked value function), and
# the difficulty-hunting incentive: the batch k-distribution becomes the
# natural corpus-times-model distribution, DAPO-style uniform + filter.
#
# Measured on production winners (.r2_analysis/_kcost.py, windows 27410-27494,
# math): median generation cost is flat across k — 127.6k/127.6k/127.1k/
# 127.0k/126.5k tokens for k=2..6 — while value-per-token under delta=1 pays
# k=2 three times k=6 (2.55 vs 0.86 per Mtok). Same cost, 3x the pay: the
# peak is pure incentive distortion, and the trained pool sits at mode k=2
# (39% math, 65-88% code). Flat pay-per-token is ~7.87/Mtok for every k.
#
# Per-slot payment is already flat (pool / B_BATCH), so this changes which
# groups win slots, never how much a slot pays. Default off.
DIFFICULTY_AUCTION_FLAT_VALUE = _os.environ.get(
    "RELIQUARY_DIFFICULTY_AUCTION_FLAT_VALUE", "0"
).strip().lower() in ("1", "true", "yes", "on")

# Conservative valuation of truncated rollouts (closes the manufactured-zero
# hole). The auction pays more for HARD prompts (value peaks at low k), so a
# miner can inflate a prompt's value by breaking one of its own correct
# rollouts: suppressing EOS runs it to max_tokens, where it grades 0 and the
# prompt looks harder. Measured on the live value function, one manufactured
# zero at k=6 is worth +68%. GPU tests showed this is NOT reliably detectable —
# an EOS-suppressed rollout looks like honest rambling, and any detector is
# farmable (a miner retries prompts until one evades).
#
# So price it instead of policing it: a truncated rollout has no gradeable
# answer, and the validator cannot know whether it would have been correct, so
# the group is valued under the interpretation LEAST favourable to the miner
# (min over "j of the truncated were actually correct", j = 0..t). The true
# outcome is always one of those interpretations, so a manipulated group can
# never score above its honest value — verified exhaustively over every
# (k, truncated-correct, truncated-wrong) combination: 0 profitable cases.
# Cost: an honestly-rambling group is also valued down, worst at high k (easy
# prompts, already near-worthless) and mildest at low k (−7% at k=2), which is
# where the auction competition actually happens. It also lets
# MAX_TRUNCATED_PER_SUBMISSION be relaxed: groups that are rejected outright
# today (earning zero) become admissible and earn a conservative value instead.
# Auction/emission only — training keeps the real reward vector.
# Declared in code, not by environment: this decides HOW MUCH a submission is
# worth, and a miner cannot read the validator's environment. An env-only
# scoring rule is invisible to the people it prices — and it silently reverts on
# a clean redeploy (see DRAND_ROUND_BACKWARD_TOLERANCE, which rejected honest
# miners for exactly that reason). Turning this on is a one-line PR: reviewable,
# versioned, announced.
CONSERVATIVE_TRUNCATION_VALUE = False
# V3 uses the exact reward lattice and includes sigma eligibility in the
# minimization. The older endpoint-only scorer remains available solely for
# historical replay tests and must never be activated as the v3 defense.
ROBUST_TRUNCATION_UTILITY_ENABLED = PROTOCOL_VERSION >= 3
# The live proof budget currently caps each environment at 96 grading attempts.
# Keep an independent ceiling so a future admission change cannot turn passive
# telemetry into unbounded seal-time CPU or archive growth.
DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES = int(
    _os.environ.get("RELIQUARY_DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES", "96")
)
DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR = int(
    _os.environ.get(
        "RELIQUARY_DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR", "2"
    )
)
if DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR <= 0:
    raise ValueError(
        "RELIQUARY_DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR "
        "must be positive"
    )

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
# Submitted drand no longer participates in auction rank, but accepting an old
# round still weakens the signed freshness contract and makes replay analysis
# ambiguous. Keep this at zero. Cross-continent body upload is handled by the
# signed precommit/reveal grace, whose drand observation is fixed when the small
# precommit reaches the validator; widening this tolerance is not the transport
# fix.
# Tests pin specific values explicitly via
# ``GrpoWindowBatcher(drand_round_backward_tolerance=...)``.
#
# Forward direction is also zero (FUTURE_ROUND is unrecoverable: a miner
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
_PROFILE_LEARNING_RATE = "1e-6" if PROTOCOL_VERSION >= 3 else "5e-6"
LEARNING_RATE = float(
    _os.environ.get("RELIQUARY_LEARNING_RATE", _PROFILE_LEARNING_RATE)
)
if not _math.isfinite(LEARNING_RATE) or not 0.0 < LEARNING_RATE <= 1e-3:
    raise ValueError(
        "RELIQUARY_LEARNING_RATE must be finite and within (0, 1e-3]"
    )

# PPO clip range. Standard in GRPO/RLHF literature.
# PPO clip band, decoupled per DAPO §3.1 (Clip-Higher). The UPPER clip is what
# bounds how fast a low-probability token can grow: at a symmetric 0.2, a token
# at p=0.01 can reach at most 0.012 while one at p=0.9 reaches 1.08, so
# exploitation tokens grow freely and exploration tokens cannot. Raising only
# eps_high reopens exploration without loosening the downside.
#
# These read as true ratio bounds only from v4 on, because v4 sampling makes
# warp() the identity so the trainer's ratio lives in the space the samples came
# from. Pre-v4 the two spaces differ by a token-dependent factor, so the band is
# left symmetric there rather than nominally "tuned" against a moving target.
# Optimizer-state precision. bitsandbytes PagedAdamW8bit saves VRAM, but its
# quantisation noise is a non-trivial fraction of a 1e-6 update, and DAPO §4.1
# trains with plain AdamW. v4 therefore defaults to full precision; pre-v4 keeps
# the 8-bit paged optimizer. Env-overridable in both directions, because whether
# fp32 optimizer state fits alongside train_model + verify_model is a property
# of the box, not of the profile.
OPTIMIZER_STATE_8BIT = (
    _os.environ.get(
        "RELIQUARY_OPTIMIZER_STATE_8BIT",
        "0" if PROTOCOL_VERSION >= 4 else "1",
    )
    not in ("0", "false", "False")
)

PPO_CLIP_EPSILON_LOW = 0.2
PPO_CLIP_EPSILON_HIGH = 0.28 if PROTOCOL_VERSION >= 4 else 0.2

# Dual clip (verl's clip_ratio_c, set to 10.0 in the DAPO reference script):
# with a negative advantage min(r·A, clip·A) follows r·A unbounded, so one
# exploding-ratio token can dominate a step; floor the surrogate at c·A.
# inf pre-v4 keeps live training byte-identical.
PPO_DUAL_CLIP_C = 10.0 if PROTOCOL_VERSION >= 4 else float("inf")

# Optional pre-step trust-region circuit breaker. A value of 1.0 is the
# compatibility default and cannot trigger because the observed fraction is at
# most 1. Recovery runs can set a lower calibrated ceiling so an unpublished
# policy that has drifted too far from the serving behavior policy is never
# stepped or published.
# v4 arms it at 0.5: a coarse backstop for the 16-window pi_old interval with
# KL off — only catastrophic drift trips it, and a trip triggers the adaptive
# early publication (fresh pi_old), not a livelock. Tighten from telemetry.
PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD = float(_os.environ.get(
    "RELIQUARY_PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD",
    "0.5" if PROTOCOL_VERSION >= 4 else "1.0",
))
if (
    not _math.isfinite(PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD)
    or not 0.0 < PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD <= 1.0
):
    raise ValueError(
        "RELIQUARY_PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD must be finite "
        "and within (0, 1]"
    )

# KL penalty weight. The correct value depends on the reference lifecycle:
# DeepSeekMath reports 0.04, while DAPO removes the reference KL term entirely.
# Neither choice transfers mechanically to this smaller mixed-domain online
# run. Keep the current value as the compatibility default, but make it
# operator-configurable so fixed-reference recovery can be calibrated without
# another code release.
# v4 follows DAPO §2.3 and removes the KL term outright: over a long-CoT run the
# policy diverges from the initial model anyway, so the restriction buys nothing
# while costing the exploration Clip-Higher exists to protect. β=0 leaves the
# circuit breakers as the collapse guard; the calibrated fallback is a PINNED
# KL_BASE_MODEL (the run's own ck0) plus an explicit β, never the rolling
# reference, which anchors nothing beyond ~4 windows.
_PROFILE_KL_BETA = (
    "0.0" if PROTOCOL_VERSION >= 4
    else "0.01" if PROTOCOL_VERSION >= 3
    else "0.04"
)
KL_BETA = float(_os.environ.get("RELIQUARY_KL_BETA", _PROFILE_KL_BETA))
if not 0.0 <= KL_BETA <= 1.0:
    raise ValueError("RELIQUARY_KL_BETA must be finite and within [0, 1]")
KL_BETA_EXPLICIT = "RELIQUARY_KL_BETA" in _os.environ

# Optional fixed KL reference checkpoint. This should normally be the run's ck0
# policy, not an unrelated raw pretrained model. Fixed mode is an explicit
# reproducibility contract: the value must be "repo@<full HF commit SHA>".
# Empty keeps the legacy rolling last-published verify_model reference.
KL_BASE_MODEL = _os.environ.get("RELIQUARY_KL_BASE_MODEL", "").strip()

# Max gradient norm before step — standard RL stability guard.
GRAD_CLIP_NORM = 1.0

# Reject a pathological update direction before optimizer.step(). Gradient
# clipping caps magnitude, but a 10k-scale pre-clip gradient can still move the
# model in a corrupted direction. The default is far above the observed healthy
# band (~1-8) while catching the 2026-07-16 production spike (10,432).
GRAD_NORM_SKIP_THRESHOLD = float(
    _os.environ.get("RELIQUARY_GRAD_NORM_SKIP_THRESHOLD", "100.0")
)
if (
    not _math.isfinite(GRAD_NORM_SKIP_THRESHOLD)
    or GRAD_NORM_SKIP_THRESHOLD <= 0.0
):
    raise ValueError(
        "RELIQUARY_GRAD_NORM_SKIP_THRESHOLD must be finite and positive"
    )

# PPO's old policy is the published checkpoint miners generated against. When
# enabled, recompute those log-probabilities with verify_model rather than
# trusting the miner-provided vector. A fixed KL anchor remains a separate model.
#
# Default ON: with the rolling KL reference, ref_model IS verify_model and the
# trainer aliases the forward (behavior_lp = ref_lp), so the exact pi_old costs
# zero extra forwards — while removing miner-fidelity noise (quantization,
# kernels, hardware) from the PPO ratio and activating the pi_old_claim_*
# telemetry that separates that noise from genuine checkpoint staleness.
RECOMPUTE_PI_OLD_FROM_VERIFY = _os.environ.get(
    "RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY", "true"
).strip().lower() in ("1", "true", "yes", "on")

# Reuse the seal-time verify-pass logprobs as pi_old instead of re-running the
# behavior forward in train_step. The verifier already computes, on the SAME
# frozen verify_model and the same tokens, the raw (T=1) log-softmax of every
# completion token during GRAIL proofs; recomputing it at train time is a
# second identical no-grad forward. Definitionally equal to the behavior
# forward for any profile (both are raw log-softmax of verify_model); the only
# delta is bf16 batch-shape noise (~1e-2 logprob), far inside the clip band.
# Rollouts lacking attached verify logprobs fall back to the behavior forward
# (never to miner-claimed values). Kill-switch, no redeploy: set to 0.
PI_OLD_FROM_VERIFY_LOGPROBS = _os.environ.get(
    "RELIQUARY_PI_OLD_FROM_VERIFY_LOGPROBS", "true"
).strip().lower() in ("1", "true", "yes", "on")

# Pipelined window collection (design: docs/superpowers/specs/
# 2026-08-19-pipelined-window-collection-design.md). When on, the next
# window's collection opens BEFORE the previous window's GPU half (proofs +
# train) runs, hiding the collection deadline entirely under GPU work:
# cycle -> max(GPU, collection) ~= P+T. Nothing is proven before its own
# deadline, the GPU stays strictly serial, and the verify-model swap at
# publish is deferred until the last window generated under the old
# checkpoint has been proven. Default OFF: flipped per-deploy in the compose
# overlay after a soak of the serial path on the same image.
PIPELINED_WINDOWS = _os.environ.get(
    "RELIQUARY_PIPELINED_WINDOWS", "0"
).strip().lower() in ("1", "true", "yes", "on")

# Detached-trainer plumbing (spec 2026-08-21-detached-trainer-r2).
# WRITE_TRAINING_PAYLOADS turns on the per-window R2 training payload
# (and tombstone) writer; it is independent from the trainer cutover so
# a shadow trainer can consume live payloads while in-process training
# still runs. DETACHED_TRAINER removes the in-process train/publish and
# swaps checkpoints from the trainer's R2 candidate manifest instead.
WRITE_TRAINING_PAYLOADS = _os.environ.get(
    "RELIQUARY_WRITE_TRAINING_PAYLOADS", "0"
).strip().lower() in ("1", "true", "yes", "on")
DETACHED_TRAINER = _os.environ.get(
    "RELIQUARY_DETACHED_TRAINER", "0"
).strip().lower() in ("1", "true", "yes", "on")

# Memoize transformers' flash-attention unpad metadata per attention mask.
# The stock helper runs once per decoder layer per pass on the SAME mask
# tensor and contains torch.nonzero — an implicit host<->device sync — so one
# train_step pays on the order of a thousand pipeline stalls for identical
# results (py-spy 2026-08-20: 12% self time; the live A/B is authoritative).
# Bit-identical memoization. Kill-switch read at import: flipping it
# requires a validator restart.
UNPAD_SYNC_CACHE = _os.environ.get(
    "RELIQUARY_UNPAD_SYNC_CACHE", "1"
).strip().lower() in ("1", "true", "yes", "on")

# Optional persistent canary ceiling. When non-zero, a validator whose current
# published checkpoint is already at or above this number keeps serving and
# accumulating but performs no further optimizer steps. Unlike an in-process
# step counter, this remains effective after a watchdog restart because the
# checkpoint number is bootstrapped from Hugging Face.
TRAIN_UNTIL_CHECKPOINT_N = int(
    _os.environ.get("RELIQUARY_TRAIN_UNTIL_CHECKPOINT_N", "0")
)
if TRAIN_UNTIL_CHECKPOINT_N < 0:
    raise ValueError("RELIQUARY_TRAIN_UNTIL_CHECKPOINT_N must be non-negative")

# train_step micro-batching: cap on padded tokens (n_seqs × longest_seq) per
# forward/backward. Short rollouts pack together; a rollout longer than this
# runs alone (= the legacy one-at-a-time path), so peak memory never exceeds a
# single sequence of this length. Sized at the protocol completion cap.
MICROBATCH_MAX_PADDED_TOKENS = 32768

# Linear LR warmup for the first N training steps (= N windows sealed).
LR_WARMUP_WINDOWS = 10

# Short LR re-ramp applied after a process restart WITHIN the same training
# run: the schedule position is reconstructed from the published checkpoint
# count (so the full warmup no longer replays on every restart), but Adam
# moments are not persisted — this many windows of ramp lets them re-estimate
# before full LR. A NEW run (fresh repo, or a checkpoint published under a
# different RELIQUARY_TRAINING_RUN_ID) still gets the full warmup.
LR_RESTART_REWARMUP_WINDOWS = int(
    _os.environ.get("RELIQUARY_LR_RESTART_REWARMUP_WINDOWS", "2")
)
if LR_RESTART_REWARMUP_WINDOWS < 0:
    raise ValueError("RELIQUARY_LR_RESTART_REWARMUP_WINDOWS must be >= 0")

# Cosine schedule end target (in windows). Chosen large so LR never
# actually reaches zero at normal cadence — effectively a slow decay.
LR_COSINE_MAX_WINDOWS = 10_000

# Default base model (HF repo id). Served as the reference for KL and the
# cold-start checkpoint.
DEFAULT_BASE_MODEL = PROTOCOL_MODEL_ID
DEFAULT_BASE_MODEL_REVISION = PROTOCOL_MODEL_REVISION

# ────────────────  WANDB TELEMETRY (opt-in, validator-only)  ────────────────

# Wandb project name used by validator-side telemetry. Operators can
# override with the WANDB_PROJECT env var.
WANDB_PROJECT = "reliquary-validator"

# Bumping this constant (or setting RELIQUARY_WANDB_VERSION) starts a
# fresh wandb run. Same value across restarts → wandb resumes the
# existing run (resume="allow").
WANDB_TRAINING_VERSION = _os.environ.get("RELIQUARY_WANDB_VERSION", "v1")

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
# v4 (full-support T=1.0) values below were calibrated on H100 2026-08-12:
# 40 honest Qwen3-4B-Base rollouts, real OMI prompts, 16384 cap, generation
# through the miner's forced-seed path and verification through this repo's
# validator functions, same box (a LOWER bound on cross-stack drift).
MIN_EOS_PROBABILITY = 0.001 if PROTOCOL_VERSION >= 4 else 0.01

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
# v4: honest full-support sampling legitimately visits the tail, so both
# floors drop to sit under the calibrated honest minima (median 0.121,
# q10 0.00064) while staying far above forged-token mass (~1/vocab ≈ 7e-6).
SAMPLING_MEDIAN_LOW_MAX = (
    0.05 if PROTOCOL_VERSION >= 4 else 0.30
)                               # median chosen prob must be above
SAMPLING_LOW_Q10_MAX = (
    0.0002 if PROTOCOL_VERSION >= 4 else 0.025
)                               # 10th-percentile must be above

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
GRADER_POOL_SIZE = 4 * M_ROLLOUTS

# Wall-clock timeout (seconds) for one structured OpenCode evaluation.
# Subprocess inside the sandbox is killed if it exceeds this. Tuned
# so that pathological miner code (infinite loops, slow algorithms)
# fails fast without blocking the queue.
GRADER_EVAL_TIMEOUT_SECONDS = 5


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
FORCED_SEED_DOMAIN = f"reliquary-forced-seed-v{PROTOCOL_VERSION}"
# A position counts toward the seed-consistency check only if its warped max
# probability is below this (i.e. the forced draw actually chooses the token).
FORCED_SEED_STOCHASTIC_MAXPROB = 0.99
# Reject a group whose stochastic-position match rate is below this floor.
# Keep the wire-v1 acceptance floor unchanged during the telemetry-only
# rollout. Live 2026-07-14 data supports a future 0.90 candidate, but changing
# acceptance policy belongs to the announced miner/protocol cutover after the
# exact-CDF shadow gate has been calibrated across implementations.
# Group-aggregate floor: _forced_seed_verdict sums (n_stoch, n_match) over ALL
# rollouts of a prompt, so this gates the GROUP mean — ~0.95 for honest miners
# at every sampling envelope (measured: v4 short groups 0.948-0.981, the 16k
# aggregate 0.9495, v3 unchanged) while a pregenerated group scores 0.59-0.67
# (H100 red-team 2026-08-12). 0.80 sits mid-gap, ~13pt margin on both sides.
# It never needed a v4 value: only the per-ROLLOUT floor below had to drop for
# v4's tail-heavy single-rollout drift. The 08-12 calibration briefly set this
# to 0.70, which left only 3pt over pregeneration — reverted.
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
# Per-rollout floor: catches a partial swap hidden in a single rollout that the
# group mean would dilute. v4 lowers it because full-support single-rollout
# honest match bottoms at 0.78 (16k, same-box) against v3's tighter envelope;
# 0.70 stays 8pt under that while still catching a 10%-swapped rollout (0.59-
# 0.66 measured, 08-12). Reconfirm against cross-stack drift during shadow.
FORCED_SEED_ROLLOUT_FLOOR = 0.70 if PROTOCOL_VERSION >= 4 else 0.75
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
# identify clients that implement the forced stream.
# v2 drops the hotkey from the forced seed (u_at) to kill multi-hotkey variance
# farming — a coordinated miner+validator change, so it bumps the version.
# This runtime implements only the v2 stream. With a checkpoint pinned and
# forced-seed enforcement enabled, the validator rejects every other advertised
# version before quota, reward grading, or proof admission. Deploy miners and
# validator as a coordinated hard cutover; FORCED_SEED_ENFORCE=false is the
# emergency compatibility switch.
FORCED_SEED_PROTOCOL_VERSION = PROTOCOL_VERSION

# Operator assertion that the CURRENT runtime is at least as fast on the proof
# path as the one the proof-capacity manifest benchmarked (e.g. right after a
# kernel upgrade such as installing causal-conv1d). Lets the stored
# qualification carry over as a conservative lower bound instead of
# crash-looping the weight-setter on the runtime-fingerprint check. Fail-closed
# by default. NEVER assert this for a change that could slow proofs down —
# re-run the capacity benchmark instead.
PROOF_CAPACITY_ACCEPT_FASTER_RUNTIME = (
    _os.environ.get("RELIQUARY_PROOF_CAPACITY_ACCEPT_FASTER_RUNTIME", "0")
    not in ("0", "false", "False")
)
