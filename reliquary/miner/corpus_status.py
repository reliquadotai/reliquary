"""`reliquary corpus status`: one hotkey's audit state, counts, recent failures
and pay on a corpus job, as the validator reports them."""

from __future__ import annotations

import datetime as _dt
from urllib.parse import quote


class MinerStatusError(RuntimeError):
    """The validator did not answer with a status."""


def fetch_miner_status(validator_url: str, hotkey: str, job_id: str | None = None, *,
                       transport=None, timeout: float = 30.0) -> dict:
    """The job-scoped route with ``job_id``, else the validator's default job."""
    import httpx

    base = validator_url.rstrip("/")
    path = (f"/corpus/jobs/{quote(job_id, safe='')}/miners/{quote(hotkey, safe='')}"
            if job_id is not None else f"/corpus/miners/{quote(hotkey, safe='')}")
    with httpx.Client(base_url=base, timeout=timeout, transport=transport) as client:
        response = client.get(path)
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text[:200]
        raise MinerStatusError(f"{response.status_code} from {base}{path}: {detail}")
    return response.json()


def _when(at) -> str:
    if at is None:
        return "-"
    return _dt.datetime.fromtimestamp(float(at), _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_miner_status(status: dict) -> str:
    """A readable summary of one status document."""
    state = status.get("audit_state")
    if state == "probation":
        state += f" ({status.get('probation_remaining')} audited passes to go)"
    elif state == "suspect":
        state += f" until {_when(status.get('suspect_until'))}"
    elif state == "banned":
        state += f" until {_when(status.get('banned_until'))}"
    lines = [
        f"job {status.get('job_id')}  hotkey {status.get('hotkey')}  as of {_when(status.get('as_of'))}",
        f"audit state: {state}",
        "submissions: accepted {submissions_accepted}  audited {audited}  passed {passed} "
        "(unaudited {passed_unaudited})  failed {failed}  pending {pending_audit}  "
        "voided {voided}".format(**status),
    ]
    if not status.get("counts_complete"):
        lines.append("  (counts still loading from before the validator's last restart)")
    share = status.get("share_last_windows") or {}
    lines.append(
        f"paid: {status.get('verified_tokens_settled')} verified tokens settled; "
        f"{100 * float(share.get('share') or 0.0):.1f}% of the task's rewards over its last "
        f"{share.get('windows', 0)} window(s) "
        f"({share.get('first_window')}..{share.get('last_window')}); task cap {status.get('cap')}")
    failures = status.get("recent_failures") or []
    if failures:
        thresholds = status.get("toploc_thresholds") or {}
        lines.append(
            f"recent failures (thresholds: exp {thresholds.get('exp_mismatch')}, "
            f"mant mean {thresholds.get('mant_mean')}, mant median {thresholds.get('mant_median')}):")
        for failure in failures:
            lines.append(
                f"  {_when(failure.get('audited_at'))}  {failure.get('submission_id', '')[:16]}  "
                f"{failure.get('reason')}  tokens {failure.get('token_count')}  "
                f"exp {failure.get('worst_exp')}  mant mean {failure.get('worst_mant_mean')}  "
                f"mant median {failure.get('worst_mant_median')}")
    else:
        lines.append("recent failures: none")
    return "\n".join(lines)


__all__ = ["MinerStatusError", "fetch_miner_status", "format_miner_status"]
