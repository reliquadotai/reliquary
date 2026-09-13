# No-reveal accounting and diagnostics

A reveal succeeds when the complete body arrives within its receipt deadline
and matches the committed byte count and SHA256. Parsing, grading, queue
capacity and subsequent proof outcomes are separate decisions. Malformed
bodies still fail validation; an exact malformed body is not a missing reveal.

The circuit is scoped to operator (coldkey) and environment. Its total `entries`
is not an incident count for one miner. `mismatch_count` is historical; the
rolling `failure_events` determine partial debt. This change does not rewrite
existing circuit state or retrospectively attribute historical incidents.

Every newly recorded incident emits `no_reveal_incident` JSON in validator logs:
cause, receipt ID, full hotkey/operator, environment, window/policy window,
prompt, server arrival/start/completion/deadline times and recording time.
Retain these logs for incident attribution; the circuit snapshot is not an
incident archive. Arrival time is the precommit timestamp, not the time the
failure was detected. No body or signing secret is logged.

The existing `rate_limited` JSON response is retained for strict older clients.
A no-reveal circuit refusal additionally includes these HTTP headers:

- `X-Reliquary-Reject-Detail: no_reveal_cooldown`
- `X-Reliquary-Circuit-Scope: operator_environment`
- `X-Reliquary-Circuit-Status`: circuit status, including recovery/probe states
- `X-Reliquary-Retry-After-Window`: retry boundary when supplied by the circuit

Read the headers on `/submit/precommit`; no miner upgrade is required to keep
submitting. These headers diagnose a circuit refusal, not the original incident.

Known deadline and size violations still count once per receipt. Internal
capacity/missing-reservation failures and unclassified disconnects terminate
and release the receipt without assigning blame. Explicit body deadline expiry
still counts. Connection loss alone cannot establish which peer caused it;
existing authentication, quotas, byte limits and receipt bounds remain active.
