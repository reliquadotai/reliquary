# CLI installation and checks

Use Python 3.11 or 3.12. Mining, training and validator workloads run on
Linux and require the additional hardware and runtime configuration in the
[miner](mining.md) and [validator](validating.md) guides. macOS is a client
and local inspection target; it is not a qualified GPU operator runtime.

For a non-editable installation from a reviewed checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip build
package_dir=$(mktemp -d)
python -m build --outdir "$package_dir"
python -m pip install "$package_dir/"*.whl
reliquary --help
```

The default package installs the client dependencies. Linux mining,
training, grading and validator installations additionally require the
`operator` extra:

```bash
python -m pip install '.[operator]'
```

When updating an existing operator installation, retain this extra; a
default client installation does not install GPU or operator dependencies.
Installing either package does not provision capacity, configure
credentials or qualify an operator host.
Use an editable installation for development, as described in the
[README](../README.md#local-development).

The CLI package workflow checks wheel and source distribution installations
on Linux and macOS with Python 3.11 and 3.12. It runs outside the checkout,
without `PYTHONPATH`, and checks every help path, the environment catalog,
unknown-command failure and the integrity of the packaged runtime fixture.
It also checks version output, local diagnostics, credential redaction and
help/diagnostics with an invalid runtime profile.
This resolves the declared client dependencies normally; the Linux validator workflow
separately installs the CPU operator dependencies and runs the validator
tests. Package checks do not exercise GPU execution or live job delivery.

To repeat the client package check in a separate environment, install the
built wheel normally, then run `scripts/check_cli_package.py` from a directory outside the checkout
with `PYTHONPATH` unset. Use a disposable environment: the check temporarily
changes and restores the installed fixture to verify that corruption is
rejected.

Local catalog inspection needs no credentials or service:

```bash
reliquary envs list
reliquary envs show openmathinstruct
reliquary tasks --help
reliquary jobs --help
reliquary eval --help
```

Check the installed version and local configuration before connecting to a
service:

```bash
reliquary --version
reliquary context --json
reliquary doctor --role client --json
reliquary doctor --role operator --json
```

`context` shows credential presence and API/admin origins without credential
values or URL paths. `doctor` checks local dependencies and runtime
configuration; a failed check exits nonzero. Operator checks additionally
require Linux. Root help, version, context and doctor remain available when
the runtime profile cannot load. These commands do not contact a service or
establish deployment readiness.

## Shell output

Management commands exposing `--json` emit one success object on stdout:

```json
{"schema":"reliquary/cli/v1","data":{}}
```

Failures emit an error object on stderr and exit nonzero. Scripts should
check the exit status before reading `data`; progress messages can also
appear on stderr. Use command-specific help to find supported flags.
`envs show` retains its existing plain JSON format.
Interrupted management commands return exit 130; retain the original IDs
and request keys and check status before repeating a write.

```bash
reliquary tasks list --json
reliquary jobs list --json
reliquary jobs status JOB_ID --json
reliquary eval status --job JOB_ID --json
```

Operator evaluation creation and remote grading support `--timeout` to
bound waiting and `--no-wait` to return available IDs and states. A wait
timeout does not cancel the underlying work. Read status and retain the
original identifiers before repeating a command; a qualification returned
by `eval create --no-wait` may still need completion before jobs can be
declared. Remote evaluation downloads require a new or empty output
directory and verify file sizes and digests before committing the bundle.

Unsuccessful commands suppress raw exception details by default. Use the
root `--debug` flag only when local diagnostic details are needed.

## Customer CPU validation

`platform` connects to the existing owner-scoped CPU dataset-validation API.
Supply a workspace key through `RELIQUARY_API_KEY` or `JOBS_API_KEY` using
your normal secret mechanism. Reads need `resources:read` or
`resources:write`; mutations need `resources:write`. Set
`RELIQUARY_API_URL` or `JOBS_API_ORIGIN` to a reviewed HTTPS origin; the
default is `https://api.reliqua.ai`. The group also accepts `--url` and
`--timeout` for the origin and per-request timeout.

```bash
reliquary platform capabilities --json
reliquary platform list --limit 25 --json
```

Capabilities report submission permission, supported controls and owned
worker capacity. An approved account may queue a job while no worker is
available. Execution requires enabled CPU validation, storage and an
enrolled owner worker; installation does not establish live availability.
This group does not provision workers or submit GPU, training or paid
generation orders. Customer commands remain available when the operator
runtime profile cannot load.

Generate and retain distinct UUIDv4 values once for each mutation. The
examples use retained `$UPLOAD_UUID`, `$CREATE_UUID` and `$CONTROL_UUID`
values. Input files contain 1–32768 bytes of UTF-8 JSONL and use
`instruction`, `preference` or `corpus` format.

```bash
reliquary platform import data.jsonl --format instruction --out input-ref.json \
  --idempotency-key "$UPLOAD_UUID" --json
reliquary platform create --input input-ref.json --deadline-seconds 600 \
  --max-attempts 2 --idempotency-key "$CREATE_UUID" --json
reliquary platform status "$JOB_ID" --json
reliquary platform wait "$JOB_ID" --timeout 300 --poll-seconds 2 --json
reliquary platform events "$JOB_ID" --limit 25 --json
```

Keep the returned input reference and job ID. List and event responses
include `next_cursor`; pass it with `--cursor` to read the next page. A
waiting timeout or interruption leaves the job running. Waiting returns
success only for a succeeded job; unsuccessful terminal states and timeouts
exit nonzero and retain the latest observed state when available.
The wait budget is checked between network reads; transport timeouts apply
to individual network phases, so a stalled read can delay the return.

Controls require the revision read from status and a reviewed reason:

```bash
reliquary platform pause "$JOB_ID" --revision "$REVISION" \
  --reason "Pause for review" --idempotency-key "$CONTROL_UUID" --json
reliquary platform download "$JOB_ID" --artifact "$ARTIFACT_ID" \
  --out validation-report.json --json
```

`resume` and `cancel` use the same control flags. Cancellation may need to
drain active work; read status to observe its result. Downloads check
artifact ownership, input identity, size and SHA-256 before atomically
creating a new file. Existing output files are preserved.

Mutations are sent once. An `outcome_unknown` error means the server may
have accepted the request; retain the original file, UUID and reviewed
revision, then reconcile with status or repeat that unchanged request.
Use a new UUID only for a separately reviewed operation.
