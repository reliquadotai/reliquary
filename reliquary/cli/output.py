"""Management command results and failures for shell callers."""

import json

import typer
from typer.core import TyperGroup

try:
    from typer._click.globals import get_current_context
    from typer._click.exceptions import UsageError
except ImportError:
    from click import get_current_context, UsageError

SCHEMA = "reliquary/cli/v1"


def _json_mode(ctx: typer.Context, value: bool) -> bool:
    ctx.meta["json"] = value
    return value


JSON_OPTION = typer.Option(False, "--json", callback=_json_mode, is_eager=True,
                          help="Print a versioned JSON result; JSON errors go to stderr.")


def emit_result(data, text: str | None = None, *, as_json: bool = False) -> None:
    if as_json:
        typer.echo(json.dumps({"schema": SCHEMA, "data": data}, sort_keys=True))
    elif text is None:
        typer.echo(json.dumps(data, indent=1))
    else:
        typer.echo(text)


def fail(message: str, *, code: str = "operation_failed", exit_code: int = 1) -> None:
    ctx = get_current_context(silent=True)
    if ctx is not None and ctx.meta.get("json"):
        typer.echo(json.dumps({"schema": SCHEMA, "error": {
            "code": code, "message": message}}, sort_keys=True), err=True)
    else:
        typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=exit_code)


class CLIGroup(TyperGroup):
    def parse_args(self, ctx, args):
        # Parse failures may precede the leaf's --json callback.
        if "--json" in args[:args.index("--") if "--" in args else len(args)]:
            ctx.meta["json"] = True
        return super().parse_args(ctx, args)

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (typer.Exit, typer.Abort):
            raise
        except UsageError as exc:
            if not ctx.meta.get("json"):
                raise
            fail(str(exc), code="invalid_input", exit_code=exc.exit_code)
        except Exception as exc:
            if ctx.meta.get("debug"):
                raise
            if isinstance(exc, ValueError):
                fail(str(exc), code="invalid_input", exit_code=2)
            if isinstance(exc, ModuleNotFoundError):
                fail("A runtime dependency is missing. Install this release with the "
                     "[operator] extra for operator commands; use --debug for local details. "
                     "Inspect persisted status before repeating a write.",
                     code="dependency_missing")
            # Library exceptions can contain credentials or remote response bodies.
            fail(f"{type(exc).__name__}: operation failed; use --debug for details. "
                 "Inspect persisted status before repeating a write.")
