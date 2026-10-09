"""Local configuration diagnostics without wallet or service access."""

import importlib
import importlib.util
import os
import platform
import sys
from urllib.parse import urlsplit

import typer

from reliquary import __version__
from reliquary.cli.output import CLIGroup, JSON_OPTION, emit_result
from reliquary.cli.platform import platform_app

app = typer.Typer(cls=CLIGroup)
app.add_typer(platform_app)


@app.callback()
def options(ctx: typer.Context, debug: bool = typer.Option(False, "--debug")) -> None:
    ctx.meta["debug"] = debug


def configuration_context() -> dict:
    def origin(value):
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if ":" in hostname:
            hostname = f"[{hostname}]"
        return f"{parsed.scheme}://{hostname}" + (f":{parsed.port}" if parsed.port else "")

    return {
        "version": __version__,
        "network": os.getenv("BT_NETWORK", "finney"),
        "netuid": os.getenv("NETUID", "81"),
        "profile": os.getenv("RELIQUARY_PROTOCOL_PROFILE") or "compiled default",
        "admin_origin": origin(os.getenv("RELIQUARY_ADMIN_URL", "http://127.0.0.1:8790")),
        "api_origin": origin(os.getenv("RELIQUARY_API_URL") or os.getenv("JOBS_API_ORIGIN")
                             or "https://api.reliqua.ai"),
        "credentials": {name: bool(os.getenv(name)) for name in (
            "RELIQUARY_ADMIN_SECRET", "RELIQUARY_EXECUTOR_TOKEN",
            "RELIQUARY_API_KEY", "JOBS_API_KEY",
            "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
        )},
    }


@app.command("context")
def context(as_json: bool = JSON_OPTION) -> None:
    """Show configuration and credential presence; never open a wallet or contact a service."""
    data = configuration_context()
    emit_result(data, as_json=as_json)


@app.command("doctor")
def doctor(
    role: str = typer.Option("client", "--role", help="client or operator dependency checks"),
    as_json: bool = JSON_OPTION,
) -> None:
    """Check local installation and runtime configuration. This is not a live readiness check."""
    if role not in {"client", "operator"}:
        raise typer.BadParameter("--role must be client or operator")
    checks = [{"name": "python", "ok": sys.version_info >= (3, 11),
               "detail": platform.python_version()},
              {"name": "platform", "ok": sys.platform in {"linux", "darwin", "win32"},
               "detail": sys.platform}]
    modules = ["typer", "httpx", "pydantic", "numpy"]
    if role == "operator":
        modules += ["torch", "bittensor", "transformers", "datasets", "aiobotocore"]
    for module in modules:
        checks.append({"name": module, "ok": importlib.util.find_spec(module) is not None})
    try:
        importlib.import_module("reliquary.constants")
        importlib.import_module("reliquary.environment.registry")
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        checks.append({"name": "runtime_configuration", "ok": False,
                       "detail": type(exc).__name__,
                       "hint": "Check protocol/profile variables and installed manifest data; "
                               "help and --version remain available."})
    else:
        checks.append({"name": "runtime_configuration", "ok": True})
    if role == "operator":
        checks.append({"name": "operator_platform", "ok": sys.platform == "linux",
                       "detail": "GPU and sandbox operators require the qualified Linux runtime."})
    ok = all(check["ok"] for check in checks)
    data = {"version": __version__, "role": role, "ok": ok, "checks": checks}
    text = "\n".join(f"{'ok' if c['ok'] else 'FAIL'} {c['name']}"
                     + (f": {c['detail']}" if c.get("detail") else "") for c in checks)
    emit_result(data, text, as_json=as_json)
    if not ok:
        raise typer.Exit(code=1)
