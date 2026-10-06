# Corpus validator process configuration

The split validator separates HTTP admission, audit and settlement, and GPU scoring into supervised processes. It reuses the corpus manifests, proof protocol, create-only verdicts and compare-and-swap settlement storage.

## Configuration

Set `RELIQUARY_CORPUS_SPLIT=1` to use the split process runner. With the switch unset, the existing single process runner remains available.

`RELIQUARY_CORPUS_SPLIT_JUDGES` selects groups by task or job ID. Separate groups with `;` and members with `,`. The default `*` assigns each remaining eligible job to its own judge process. A job cannot appear in two groups. Episode jobs stay in the front process because their grading services and routes are hosted there; explicitly assigning an episode job to a judge group is refused. Jobs outside the configured groups are judged in the front.

The split runner requires `--no-set-weights` and refuses remote audit executor mode. It performs manifest, checkpoint fingerprint and proof compatibility checks before starting children.

## Process ownership

- The front serves corpus HTTP routes, commits admission ledgers and durable records, and sends arrival notifications to the configured judges.
- Judges audit and settle their assigned jobs. Status queries use their reported counters; feed coverage protects decisions that depend on complete arrival history.
- The GPU process loads the checkpoint and scores requests from the front and judges through a bounded queue.
- The supervisor restarts an exited child after a bounded backoff while leaving the other children running. Children use the parent-death signal on supported Linux hosts to avoid surviving a dead supervisor.

Internal scoring, feed and status calls use Unix sockets under the directory selected by `RELIQUARY_CORPUS_SPLIT_DIR`. They do not add public TCP listeners.

## Shutdown

The front handles a termination signal through the native HTTP server, allowing tracked requests to finish before service cleanup. Each child cancels and awaits its owned services and their asynchronous descendants, including arrival listings, status backfills and audit or grading work. Custom worker pools are joined before child completion.

The supervisor shares one bounded shutdown deadline across its children. `RELIQUARY_CORPUS_SPLIT_STOP_GRACE_SECONDS` sets that deadline; the default is 20 seconds. The value must be finite and nonnegative. Configure it below the container's stop budget so the supervisor can complete its bounded fallback. A surviving child is killed after the deadline; persistent records and compare-and-swap state remain the recovery source.

Closing the front also stops arrival heartbeats. A later front restart establishes a new feed epoch and requires a complete arrival listing before uncovered records can pass without a drawn audit. Live status, durable accepted records, completed audits and settlement are separate observations.

## Validation

`tests/unit/test_corpus_shutdown.py` exercises native split child entry points with held HTTP record writes, asynchronous service cleanup, feed listings, status backfills and thread completion. `tests/unit/test_corpus_multi_job_validator.py` covers hot job retirement and shutdown ownership. Runtime qualification remains required for the selected checkpoint, proof implementation and deployed configuration.
