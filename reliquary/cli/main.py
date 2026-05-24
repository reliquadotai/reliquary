"""Reliquary CLI — mine and validate commands."""

import asyncio
import atexit
import logging
import os
import shutil
import socket as _socket
import subprocess
import threading
import time as _time

import typer

from reliquary.constants import DEFAULT_BASE_MODEL, DEFAULT_HF_REPO_ID, ENVIRONMENT_MIX, VALIDATOR_HTTP_PORT

_DEFAULT_ENVS = ",".join(name for name, _ in ENVIRONMENT_MIX)

app = typer.Typer(name="reliquary", help="Reliquary — Verifiable Inference Subnet")

logger = logging.getLogger(__name__)

_grader_proc: "subprocess.Popen | None" = None


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
    and the OCI bundle are present; otherwise fall back to plain Python
    workers (sufficient for dev / single-host testing; production miners
    should install runsc for deterministic-with-validator reward parity).
    """
    global _grader_proc
    from reliquary.constants import GRADER_SOCKET_PATH

    _logger = logging.getLogger("reliquary.cli")

    if _grader_is_running(GRADER_SOCKET_PATH):
        _logger.info("Grader already running at %s; reusing it", GRADER_SOCKET_PATH)
        return

    if use_runsc is None:
        bundle_python = (
            "/opt/reliquary/reliquary/environment/grader/bundle/"
            "rootfs/usr/local/bin/python3"
        )
        use_runsc = bool(shutil.which("runsc")) and os.path.exists(bundle_python)
    if not use_runsc:
        _logger.warning(
            "Grader will run WITHOUT runsc — rewards may diverge from the "
            "validator on edge cases. Install runsc + build the bundle "
            "(scripts/build_grader_bundle.sh) for production miners."
        )

    cmd = ["python", "-m", "reliquary.environment.grader.server"]
    if use_runsc:
        cmd.append("--use-runsc")

    _logger.info("Launching grader server (use_runsc=%s) ...", use_runsc)
    _grader_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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

    # Auto-launch the grader server so reward computation works for
    # OpenCodeInstruct prompts out of the box. Idempotent: skipped if
    # one's already listening.
    _ensure_grader_running()

    async def _run():
        import bittensor as bt
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.environment import load_environments
        from reliquary.infrastructure.chain import get_subtensor, get_metagraph, NETUID
        from reliquary.miner.engine import MiningEngine
        from reliquary.miner.submitter import discover_validator_url, get_window_state_v2

        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
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
        tokenizer = AutoTokenizer.from_pretrained(initial_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Use 2 GPUs when available (vllm on 0, HF proof on 1). Fall back to
        # sharing GPU 0 for test boxes that only expose one device.
        proof_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"

        vllm_model = AutoModelForCausalLM.from_pretrained(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to("cuda:0").eval()

        hf_model = AutoModelForCausalLM.from_pretrained(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
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

    if train:
        _ensure_grader_running()

    env_names = [n.strip() for n in environments.split(",") if n.strip()]
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

        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        if train:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from reliquary.constants import ATTN_IMPLEMENTATION
            from reliquary.validator.service import ValidationService

            logger.info("Loading model from %s...", checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to("cuda:0").eval()

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

    asyncio.run(_run())


if __name__ == "__main__":
    app()
