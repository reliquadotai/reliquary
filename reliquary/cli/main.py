"""Reliquary CLI — mine and validate commands."""

import asyncio
import atexit
import logging
import math
import os
import shutil
import socket as _socket
import subprocess
import sys
import threading
import time as _time
from pathlib import Path

import typer

from reliquary.constants import (
    PROOF_PROCESS_ISOLATION,
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_ENVIRONMENTS,
    DEFAULT_HF_REPO_ID,
    ENVIRONMENT_MIX,
    FORENSIC_SAMPLE_PER_WINDOW,
    MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV,
    MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW,
    MAX_PROOF_WALL_SECONDS,
    PROOF_SLOTS_PER_DEVICE,
    PROTOCOL_MODEL_ID,
    PROTOCOL_MODEL_REVISION,
    PROTOCOL_PROFILE_ID,
    PROTOCOL_VERSION,
    VALIDATOR_HTTP_PORT,
)
from reliquary.validator.errors import FatalProofPlaneError

_DEFAULT_ENVS = DEFAULT_ENVIRONMENTS

app = typer.Typer(name="reliquary", help="Reliquary — Verifiable Inference Subnet")

logger = logging.getLogger(__name__)

_grader_proc: "subprocess.Popen | None" = None


def _run_validator_event_loop(coroutine) -> None:
    """Run the validator and hard-exit after an unrecoverable proof fault.

    ``FatalProofPlaneError`` is raised only after ``ValidationService.run`` has
    executed its best-effort cleanup.  A faulted proof worker can still be
    blocked inside a synchronous CUDA call, though, so normal interpreter and
    extension teardown is not safe to rely on.  ``os._exit`` gives the shell
    supervisor an actual child exit and lets Docker apply its restart policy.
    """

    try:
        asyncio.run(coroutine)
    except FatalProofPlaneError:
        logger.critical(
            "Fatal proof-plane cleanup completed; forcing process exit for "
            "supervisor restart",
            exc_info=True,
        )
        # Logging handlers flush each record, but flush the standard streams
        # explicitly because os._exit deliberately skips interpreter cleanup.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os._exit(1)


def _configured_proof_device_identities(torch_module):
    raw = os.environ.get("RELIQUARY_PROOF_DEVICES", "")
    requested = tuple(
        device.strip() for device in raw.split(",") if device.strip()
    )
    if PROTOCOL_VERSION < 3:
        if requested:
            logger.warning(
                "Ignoring RELIQUARY_PROOF_DEVICES under protocol profile %s",
                PROTOCOL_PROFILE_ID,
            )
        return ()
    if not requested:
        raise RuntimeError(
            f"{PROTOCOL_PROFILE_ID} requires explicit proof replicas; "
            "set RELIQUARY_PROOF_DEVICES after capacity qualification"
        )

    from reliquary.validator.proof_capacity import (
        resolve_cuda_proof_devices,
    )

    return resolve_cuda_proof_devices(
        requested,
        cuda=torch_module.cuda,
    )


def _v3_activation_checkpoint_revision(
    checkpoint: str,
    resume_from: str,
) -> str | None:
    if PROTOCOL_VERSION < 3:
        return None
    if checkpoint != PROTOCOL_MODEL_ID:
        raise RuntimeError(
            f"{PROTOCOL_PROFILE_ID} must bootstrap from "
            f"{PROTOCOL_MODEL_ID}@{PROTOCOL_MODEL_REVISION}"
        )
    prefix = "sha:"
    revision = (
        resume_from[len(prefix):].strip().lower()
        if resume_from.startswith(prefix)
        else ""
    )
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError(
            f"{PROTOCOL_PROFILE_ID} requires "
            "RELIQUARY_RESUME_FROM=sha:<stamped-40-char-checkpoint>"
        )
    return revision


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _miner_requires_grader(env_names: list[str]) -> bool:
    # Miners never grade: opencode reward is validator-authoritative, so the
    # reference miner only generates rollouts. The gVisor grader runs on the
    # validator side. (Operators self-testing best-of-n run their own grader.)
    return False


def _grader_bundle_python() -> Path:
    bundle = os.environ.get(
        "GRADER_BUNDLE_PATH",
        "/opt/reliquary/reliquary/environment/grader/bundle",
    )
    return Path(bundle) / "rootfs" / "usr" / "local" / "bin" / "python3"


def _grader_is_running(socket_path: str, timeout: float = 0.5) -> bool:
    """Return True iff the grader is reachable on the Unix socket."""
    try:
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(socket_path)
        return True
    except (FileNotFoundError, ConnectionRefusedError, _socket.timeout, OSError):
        return False


def _ensure_grader_running(use_runsc: "bool | None" = None) -> None:
    """Start the grader server in the background if no one is listening.

    The grader is required for reward computation on code-execution envs
    (OpenCodeInstruct). Without it, OCI rewards silently return 0.0 and
    the validator rejects every OCI submission as a reward-claim mismatch.

    If `use_runsc` is None, auto-detect: use runsc when both the binary
    and the OCI bundle are present. Plain Python fallback is refused unless
    RELIQUARY_ALLOW_UNSANDBOXED_GRADER=1 is set for an isolated lab.
    """
    global _grader_proc
    from reliquary.constants import GRADER_SOCKET_PATH

    _logger = logging.getLogger("reliquary.cli")

    if _grader_is_running(GRADER_SOCKET_PATH):
        _logger.info("Grader already running at %s; reusing it", GRADER_SOCKET_PATH)
        return

    if use_runsc is None:
        use_runsc = bool(shutil.which("runsc")) and _grader_bundle_python().exists()
    if not use_runsc:
        if not _env_flag("RELIQUARY_ALLOW_UNSANDBOXED_GRADER", "0"):
            raise RuntimeError(
                "opencodeinstruct requires the gVisor/runsc grader sandbox. "
                "Install runsc and build the grader bundle, or set "
                "RELIQUARY_ALLOW_UNSANDBOXED_GRADER=1 only on isolated throwaway labs."
            )
        _logger.warning("Launching UNSANDBOXED grader because RELIQUARY_ALLOW_UNSANDBOXED_GRADER=1 is set.")

    cmd = [sys.executable, "-m", "reliquary.environment.grader.server"]
    if use_runsc:
        cmd.append("--use-runsc")

    sanitized_env = {
        "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": os.environ.get("GRADER_HOME", "/tmp/reliquary-grader-home"),
        "GRADER_SOCKET_PATH": GRADER_SOCKET_PATH,
        "GRADER_BUNDLE_PATH": os.environ.get(
            "GRADER_BUNDLE_PATH",
            "/opt/reliquary/reliquary/environment/grader/bundle",
        ),
    }

    _logger.info("Launching grader server (use_runsc=%s, scrubbed_env=1) ...", use_runsc)
    _grader_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=sanitized_env,
        start_new_session=True,
    )

    def _cleanup() -> None:
        if _grader_proc is not None and _grader_proc.poll() is None:
            try:
                _grader_proc.terminate()
                _grader_proc.wait(timeout=5)
            except Exception:
                try:
                    _grader_proc.kill()
                except Exception:
                    pass
    atexit.register(_cleanup)

    deadline = _time.time() + 15.0
    while _time.time() < deadline:
        if _grader_is_running(GRADER_SOCKET_PATH):
            _logger.info("Grader server ready at %s", GRADER_SOCKET_PATH)
            return
        _time.sleep(0.2)

    _logger.error(
        "Grader server failed to bind %s within 15s. OCI rewards will "
        "be 0 and all OCI submissions will be rejected. Diagnose by "
        "running `python -m reliquary.environment.grader.server%s` manually.",
        GRADER_SOCKET_PATH, " --use-runsc" if use_runsc else "",
    )


def setup_logging(level: str = "INFO"):
    # ``%(threadName)s`` distinguishes the main asyncio loop from the
    # dedicated ``weight-setter`` thread (see ``validate`` below) when
    # tailing logs.
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def mine(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    wallet_path: str = typer.Option(
        os.getenv("BT_WALLET_PATH", ""),
        help="Optional wallet base path",
    ),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    environments: str = typer.Option(
        os.getenv("RELIQUARY_ENVIRONMENTS", _DEFAULT_ENVS),
        help="Comma-separated environment names (env: RELIQUARY_ENVIRONMENTS)",
    ),
    validator_url: str = typer.Option(
        "",
        help=(
            "Override the validator URL (otherwise discovered from the metagraph). "
            "Useful for local testing — e.g. http://127.0.0.1:8888"
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run Reliquary miner."""
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    env_names = [n.strip() for n in environments.split(",") if n.strip()]
    logger.info(
        "Starting Reliquary miner (network=%s, netuid=%d, envs=%s)",
        network, netuid, env_names,
    )

    # Miners never grade (opencode reward is validator-authoritative), so this
    # stays False; the gVisor grader runs on the validator only.
    if _miner_requires_grader(env_names):
        _ensure_grader_running()
    elif "opencodeinstruct" in env_names:
        logger.info("OpenCode miner: reward is validator-authoritative; skipping local grader launch.")

    async def _run():
        import bittensor as bt
        import torch
        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.environment import load_environments
        from reliquary.infrastructure.chain import get_subtensor, get_metagraph, NETUID
        from reliquary.miner.engine import MiningEngine
        from reliquary.miner.submitter import discover_validator_url, get_window_state_v2
        from reliquary.shared.modeling import (
            MODEL_SNAPSHOT_ALLOW_PATTERNS,
            load_text_generation_model,
            load_tokenizer,
        )

        wallet_kwargs = {"name": wallet_name, "hotkey": hotkey}
        if wallet_path:
            wallet_kwargs["path"] = wallet_path
        wallet = bt.Wallet(**wallet_kwargs)
        subtensor = await get_subtensor()

        # --- Resolve initial checkpoint from validator if available ---
        initial_path = checkpoint  # fallback to --checkpoint arg
        try:
            if validator_url:
                url = validator_url
            else:
                metagraph = await get_metagraph(subtensor, NETUID)
                url = discover_validator_url(metagraph)

            import httpx
            from huggingface_hub import snapshot_download
            async with httpx.AsyncClient(timeout=30) as client:
                state = await get_window_state_v2(url, client=client)
            if state.checkpoint_repo_id and state.checkpoint_revision:
                logger.info(
                    "Validator at %s is on checkpoint %d (%s@%s). "
                    "Downloading to seed the miner model.",
                    url, state.checkpoint_n, state.checkpoint_repo_id,
                    state.checkpoint_revision[:12],
                )
                initial_path = snapshot_download(
                    repo_id=state.checkpoint_repo_id,
                    revision=state.checkpoint_revision,
                    allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
                )
                logger.info("Using initial checkpoint path: %s", initial_path)
            else:
                logger.info(
                    "Validator has no published checkpoint yet — using --checkpoint=%s",
                    checkpoint,
                )
        except Exception as e:
            logger.warning(
                "Could not fetch validator checkpoint (%s); falling back to "
                "--checkpoint=%s",
                e, checkpoint,
            )

        # --- Load models from resolved path ---
        logger.info("Loading models from %s...", initial_path)
        base_load_kwargs = (
            {"revision": DEFAULT_BASE_MODEL_REVISION}
            if initial_path == DEFAULT_BASE_MODEL
            else {}
        )
        tokenizer = load_tokenizer(initial_path, **base_load_kwargs)

        # Use 2 GPUs when available (vllm on 0, HF proof on 1). Fall back to
        # sharing GPU 0 for test boxes that only expose one device.
        proof_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"

        vllm_model = load_text_generation_model(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
            **base_load_kwargs,
        ).to("cuda:0").eval()

        hf_model = load_text_generation_model(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
            **base_load_kwargs,
        ).to(proof_device).eval()

        envs = load_environments(env_names)
        mix = [(n, w) for n, w in ENVIRONMENT_MIX if n in envs]
        engine = MiningEngine(
            vllm_model,
            hf_model,
            tokenizer,
            wallet,
            envs=envs,
            mix=mix,
            proof_gpu=0 if proof_device == "cuda:0" else 1,
            validator_url_override=validator_url or None,
        )

        # Seed engine's _loaded_checkpoint_path so the first
        # maybe_pull_checkpoint sees we're already synced (skips redundant reload).
        if initial_path != checkpoint:
            engine._loaded_checkpoint_path = initial_path

        logger.info("Miner ready. Entering main loop.")
        try:
            await engine.mine_window(subtensor, 0, use_drand=use_drand)
        except KeyboardInterrupt:
            logger.info("Miner interrupted by user")
        except Exception as e:
            logger.error("Mining loop crashed: %s", e, exc_info=True)
            raise

    asyncio.run(_run())


@app.command()
def validate(
    train: bool = typer.Option(
        True,
        "--train/--no-train",
        help=(
            "Run full trainer mode (default). "
            "Pass --no-train for weight-only mode: reads R2 archives, "
            "computes EMA, submits weights. No GPU, no HF, no HTTP server."
        ),
    ),
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    wallet_path: str = typer.Option(
        os.getenv("BT_WALLET_PATH", ""),
        help="Optional wallet base path",
    ),
    checkpoint: str = typer.Option(DEFAULT_BASE_MODEL, help="HF repo id or local path of the model to load (trainer mode only)"),
    environments: str = typer.Option(
        os.getenv("RELIQUARY_ENVIRONMENTS", _DEFAULT_ENVS),
        help="Comma-separated environment names (trainer mode only; env: RELIQUARY_ENVIRONMENTS)",
    ),
    http_host: str = typer.Option("0.0.0.0", help="HTTP bind address (trainer mode only)"),
    http_port: int = typer.Option(VALIDATOR_HTTP_PORT, help="HTTP listen port (trainer mode only)"),
    external_ip: str = typer.Option(
        "",
        help=(
            "Public IP this validator is reachable at. Published on-chain via "
            "serve_axon so miners can discover it through the metagraph. "
            "Leave empty to skip publishing (miners then need --validator-url). "
            "Trainer mode only."
        ),
    ),
    external_port: int = typer.Option(
        0,
        help="Public port to advertise on-chain; defaults to --http-port when 0. Trainer mode only.",
    ),
    hf_repo_id: str = typer.Option(
        DEFAULT_HF_REPO_ID,
        help="HuggingFace repo ID to publish checkpoints to (must be writable with HF_TOKEN). Trainer mode only.",
    ),
    resume_from: str = typer.Option(
        os.getenv("RELIQUARY_RESUME_FROM", ""),
        help=(
            "Resume trainer from a checkpoint instead of the base model. "
            "Accepts 'sha:<40-hex>' (HF commit on --hf-repo-id) or "
            "'path:<dir>' (local ckpt_<N> directory). Trainer mode only."
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run Reliquary validator (trainer mode by default; --no-train for weight-only)."""
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    env_names = [n.strip() for n in environments.split(",") if n.strip()]
    if train and "opencodeinstruct" in env_names:
        _ensure_grader_running()
    if train:
        logger.info(
            "Starting Reliquary validator [trainer] (network=%s, netuid=%d, envs=%s, http=%s:%d)",
            network, netuid, env_names, http_host, http_port,
        )
    else:
        logger.info(
            "Starting Reliquary validator [weight-only] (network=%s, netuid=%d)",
            network, netuid,
        )

    async def _run():
        import bittensor as bt

        from reliquary.infrastructure.chain import get_subtensor

        wallet_kwargs = {"name": wallet_name, "hotkey": hotkey}
        if wallet_path:
            wallet_kwargs["path"] = wallet_path
        wallet = bt.Wallet(**wallet_kwargs)
        subtensor = await get_subtensor()

        if train:
            import torch
            from reliquary.constants import ATTN_IMPLEMENTATION
            from reliquary.shared.modeling import load_text_generation_model, load_tokenizer
            from reliquary.validator.service import (
                ValidationService,
                load_validator_replica,
            )

            activation_checkpoint_revision = (
                _v3_activation_checkpoint_revision(checkpoint, resume_from)
            )
            logger.info("Loading model from %s...", checkpoint)
            base_load_kwargs = (
                {"revision": DEFAULT_BASE_MODEL_REVISION}
                if checkpoint == DEFAULT_BASE_MODEL
                else {}
            )
            tokenizer = load_tokenizer(checkpoint, **base_load_kwargs)

            # Resolve the proof plane's topology BEFORE loading this process's
            # replica: whether the plane is isolated decides which device that
            # replica belongs on, and "isolated" means a plane was actually
            # built, not merely that the flag is set.
            from reliquary.constants import DETACHED_TRAINER
            from reliquary.validator.proof_capacity import expand_proof_slots
            from reliquary.validator.proof_worker import (
                assert_isolation_supported,
                assert_proof_slots_supported,
            )

            assert_isolation_supported(
                isolation=PROOF_PROCESS_ISOLATION,
                detached_trainer=DETACHED_TRAINER,
            )
            assert_proof_slots_supported(
                slots_per_device=PROOF_SLOTS_PER_DEVICE,
                isolation=PROOF_PROCESS_ISOLATION,
            )
            proof_device_identities = _configured_proof_device_identities(
                torch
            )
            proof_devices = tuple(
                identity.device_id for identity in proof_device_identities
            )
            # Capacity is validated against the PHYSICAL devices below and must
            # stay that way — it is a claim about cards, not processes. Only
            # the plane is widened to one entry per proof slot.
            proof_slots = expand_proof_slots(
                proof_devices, PROOF_SLOTS_PER_DEVICE
            )
            isolated_plane = bool(PROOF_PROCESS_ISOLATION and proof_slots)

            model = load_validator_replica(
                checkpoint,
                isolated_plane=isolated_plane,
                **base_load_kwargs,
            )

            proof_capacity_qualification = None
            if PROTOCOL_VERSION >= 3:
                from reliquary.shared.runtime_fingerprint import (
                    collect_runtime_fingerprint,
                )
                from reliquary.validator.observability import (
                    immutable_build_revision,
                )
                from reliquary.validator.proof_capacity import (
                    load_proof_capacity_qualification,
                )

                manifest_path = os.environ.get(
                    "RELIQUARY_PROOF_CAPACITY_MANIFEST", ""
                ).strip()
                manifest_sha256 = os.environ.get(
                    "RELIQUARY_PROOF_CAPACITY_MANIFEST_SHA256", ""
                ).strip()
                if not manifest_path or not manifest_sha256:
                    raise RuntimeError(
                        f"{PROTOCOL_PROFILE_ID} requires a pinned "
                        "proof-capacity manifest"
                    )
                qualification = load_proof_capacity_qualification(
                    manifest_path,
                    expected_sha256=manifest_sha256,
                )
                hardware = tuple(
                    identity.hardware_class
                    for identity in proof_device_identities
                )
                device_uuids = tuple(
                    identity.device_uuid
                    for identity in proof_device_identities
                )
                runtime_fingerprint_hash = collect_runtime_fingerprint(
                    generation_model=model,
                    proof_model=model,
                )["profile_hash"]
                from reliquary.validator.proof_capacity import (
                    compute_proof_path_hash,
                )

                proof_capacity_qualification = qualification.validate(
                    profile_id=PROTOCOL_PROFILE_ID,
                    model_revision=PROTOCOL_MODEL_REVISION,
                    software_revision=immutable_build_revision(),
                    checkpoint_revision=(
                        activation_checkpoint_revision or ""
                    ),
                    runtime_fingerprint_hash=runtime_fingerprint_hash,
                    proof_path_hash=compute_proof_path_hash(),
                    configured_devices=proof_devices,
                    configured_hardware=hardware,
                    configured_device_uuids=device_uuids,
                    proof_wall_seconds=MAX_PROOF_WALL_SECONDS,
                    minimum_proofs_per_environment=(
                        MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW
                        + FORENSIC_SAMPLE_PER_WINDOW
                    ),
                    minimum_completion_tokens_per_environment={
                        environment: math.ceil(cap * 0.9)
                        for environment, cap in (
                            MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV.items()
                        )
                    },
                )
                carried_from = proof_capacity_qualification.get(
                    "qualification_carried_over_from"
                )
                if carried_from:
                    logger.info(
                        "Proof capacity qualification carried over from "
                        "image %s: this image's proof path is byte-identical",
                        carried_from,
                    )
                logger.info(
                    "Proof capacity qualified: %s",
                    proof_capacity_qualification,
                )
            proof_models = {}
            proof_worker_pool = None
            if isolated_plane:
                # The proof plane leaves this interpreter: every replica is
                # loaded by a worker, this process keeps its own pair on the
                # CPU, and the event loop can no longer convoy a proof thread
                # off the GIL. Several slots may share one card.
                from reliquary.validator.proof_worker import (
                    build_isolated_proof_plane,
                )

                logger.info(
                    "Starting isolated proof plane: %d slot(s) over %d GPU(s) "
                    "(%s); replicas load in the workers",
                    len(proof_slots),
                    len(proof_devices),
                    ", ".join(proof_slots),
                )
                proof_worker_pool, proof_models = build_isolated_proof_plane(
                    devices=proof_slots,
                    checkpoint=checkpoint,
                    load_kwargs=base_load_kwargs,
                    reference_model=model,
                )
                proof_worker_pool.start()
                logger.info(
                    "Isolated proof plane ready on %s",
                    ", ".join(proof_slots),
                )
            else:
                for device in proof_devices:
                    if device == "cuda:0":
                        continue
                    logger.info(
                        "Loading frozen proof replica on %s from %s",
                        device,
                        checkpoint,
                    )
                    proof_models[device] = load_text_generation_model(
                        checkpoint,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=ATTN_IMPLEMENTATION,
                        **base_load_kwargs,
                    ).to(device).eval()

            mix = [(n, w) for n, w in ENVIRONMENT_MIX if n in env_names]
            service = ValidationService(
                wallet,
                model,
                tokenizer,
                netuid=netuid,
                use_drand=use_drand,
                http_host=http_host,
                http_port=http_port,
                external_ip=external_ip or None,
                external_port=(external_port or http_port) if external_ip else None,
                hf_repo_id=hf_repo_id,
                resume_from=resume_from or None,
                env_mix=mix if mix else None,
                proof_devices=proof_slots or None,
                proof_models=proof_models or None,
                proof_capacity_qualification=(
                    proof_capacity_qualification
                ),
                proof_worker_pool=proof_worker_pool,
            )
            # Run the weight setter in a dedicated OS thread with its own
            # event loop. asyncio is single-threaded, so any sync blocking
            # call on the trainer's loop (e.g. /state acquiring a lock the
            # GRAIL verifier is holding) would stall set_weights too. The
            # weight setter's own subtensor (see WeightOnlyValidator.run)
            # plus its own loop here means neither side can block the other.
            from reliquary.validator.weight_only import WeightOnlyValidator

            def _run_weight_setter() -> None:
                try:
                    worker = WeightOnlyValidator(wallet=wallet, netuid=netuid)
                    asyncio.run(worker.run())
                except Exception:
                    logger.exception("weight-setter thread crashed")

            threading.Thread(
                target=_run_weight_setter,
                name="weight-setter",
                daemon=True,
            ).start()
            await service.run(subtensor)
        else:
            from reliquary.validator.weight_only import WeightOnlyValidator

            validator = WeightOnlyValidator(wallet=wallet, netuid=netuid)
            await validator.run()

    _run_validator_event_loop(_run())


@app.command("train-worker")
def train_worker(
    shadow: bool = typer.Option(
        False,
        "--shadow",
        help=(
            "Consume payloads and train but never publish — for the "
            "pre-cutover comparison against the in-process trainer."
        ),
    ),
) -> None:
    """Detached trainer: consume R2 training payloads, publish checkpoints.

    See docs/superpowers/specs/2026-08-21-detached-trainer-r2-design.md.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(threadName)s | %(name)s | "
               "%(levelname)s | %(message)s",
    )
    from reliquary.trainer.cli import run_train_worker

    run_train_worker(shadow=shadow)


if __name__ == "__main__":
    app()
