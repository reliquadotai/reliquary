"""Keep installation help and local diagnostics independent of runtime startup."""

import sys

from reliquary import __version__

ROOT_HELP = """Usage: reliquary [--debug] COMMAND [ARGS]...

Reliquary — Verifiable Inference Subnet

Options:
  --version             Show installed version.
  --help                Show this message.
  --debug               Show local exception details.
  --install-completion  Install shell completion.
  --show-completion     Print shell completion.

Commands:
  context          Show redacted local configuration.
  doctor           Check installation and runtime configuration.
  platform         Run customer CPU dataset validation jobs.
  tasks            Declare and retire subnet tasks.
  envs             Inspect the environment catalog.
  jobs             Manage corpus generation jobs and exports.
  eval             Create, grade and compare evaluations.
  admin            Serve the signed subnet admin API.
  corpus           Mine corpus jobs and run corpus services.
  mine             Run a miner.
  validate         Run a validator.
  watch-verdicts   Stream final verdicts.
  proof-worker     Run a proof worker.
  train-worker     Run a detached trainer.

Run reliquary COMMAND --help for command options.
"""


def main() -> None:
    args = sys.argv[1:]
    if "--version" in args and all(arg in {"--version", "--debug"} for arg in args):
        print(__version__)
        return
    if not args or ("--help" in args and all(arg in {"--help", "--debug"} for arg in args)):
        print(ROOT_HELP)
        return
    command = args[1] if args[0] == "--debug" and len(args) > 1 else args[0]
    if command in {"doctor", "context", "platform"}:
        from reliquary.cli.diagnostics import app
    else:
        try:
            from reliquary.cli.main import app
        except (ImportError, ValueError, OSError, RuntimeError):
            if "--debug" in args:
                raise
            message = "runtime configuration could not load; run reliquary doctor --json or use --debug for local details."
            if "--json" in args:
                import json
                from reliquary.cli.output import SCHEMA

                print(json.dumps({"schema": SCHEMA, "error": {
                    "code": "runtime_configuration", "message": message}}), file=sys.stderr)
            else:
                print(f"error: {message}", file=sys.stderr)
            raise SystemExit(1) from None
    app()
