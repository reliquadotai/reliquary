"""NONADMISSIBLE CPU load complement to authentic proof-capacity measurements.

No admission, payment, storage, signatures or model execution occurs here.
Real kernel outputs remain unchanged. Additional CPU-only fixtures deliberately
exercise scans that a rejected/random full-length proof may exit early. Their
cost is added conservatively; their verdicts never become proof evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from reliquary import constants as c
from reliquary.environment.forced_sampling import u_at
from reliquary.environment.opencodeinstruct import _entry_function_name
from reliquary.shared.modeling import resolve_eos_token_ids, think_close_token_ids
from reliquary.validator import batcher as b
from reliquary.validator import verifier as v
from reliquary.validator.utility_telemetry import utility_telemetry_enabled

STRESS_PROMPT_TOKENS = 24576
STRESS_POLICY_TOKENS = 8192
STRESS_ROLLOUTS = 16
SCOPE = "NONADMISSIBLE_POSTPROOF_CPU_STRESS"
FIXTURE_SCOPE = "NONADMISSIBLE_SYNTHETIC_CPU_ONLY"
REQUIRED_COVERAGE = (
    "policy_token_positions", "rollout_hash", "completion_decode", "token_metrics",
    "sketch_metrics", "seed_u", "seed_u_hash", "pi_old", "eos_padding", "termination",
    "cap_truncation", "natural_bft_cap", "force_span", "termination_classification",
    "logprob", "distribution", "boxed", "token_auth", "all_token_auth", "utility_nll",
    "utility_entropy", "seed_rollout_verdict", "fixture_full_scan_auth",
    "fixture_full_scan_pi_old", "fixture_max_findings_auth", "fixture_full_boxed",
)
_GROUP_COVERAGE = (
    "get_problem", "resolve_eos", "reward_shape", "rewards_std", "seed_group_verdict",
    "fixture_boxed_tokens", "fixture_reward_shape",
)


def runtime_settings() -> dict:
    """Values that select or change the measured native helper work."""
    names = (
        "CHALLENGE_K", "T_PROTO", "MIN_EOS_PROBABILITY", "MAX_NEW_TOKENS_PROTOCOL_CAP",
        "MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV", "BFT_ENABLED", "BFT_THINKING_BUDGET",
        "BFT_ANSWER_BUDGET", "BFT_FORCE_ANSWER", "LOGPROB_IS_EPS", "BOXED_ANSWER_MIN_PROB",
    )
    prefixes = ("TOKEN_AUTH_", "ALL_TOKEN_AUTH_", "CODE_SEMANTIC_AUTH_", "SAMPLING_",
                "FORCED_SEED_", "REWARD_SHAPE_")
    return {
        "constants": {name: getattr(c, name) for name in sorted(set(names) | {
            name for name in vars(c) if name.startswith(prefixes)
        })},
        "numeric_auth_threshold": v.NUMERIC_AUTH_THRESHOLD,
        "utility_telemetry_enabled": utility_telemetry_enabled(),
        "auth_forensics_enabled": b.auth_forensics_enabled(),
        "auth_forensics_max_findings": b.auth_forensics_max_findings_per_rollout(),
        "auth_forensics_context_chars": b.auth_forensics_context_chars(),
        "code_counterfactual_enabled": b.code_semantic_counterfactual_enabled(),
        "code_counterfactual_max_findings": b.code_semantic_counterfactual_max_findings_per_rollout(),
    }


def validate_postproof_measurement(report, *, rollout_count=16, completion_tokens=8192) -> None:
    """Reject an incomplete/differently configured receipt; never infer validity."""
    if (not isinstance(report, dict) or report.get("scope") != SCOPE
            or report.get("admissible") is not False or report.get("complete") is not True
            or report.get("errors") != []):
        raise ValueError("incomplete or incorrectly scoped postproof stress receipt")
    if (report.get("rollout_count") != rollout_count
            or report.get("prompt_tokens") != STRESS_PROMPT_TOKENS
            or report.get("policy_tokens") != completion_tokens
            or report.get("environment") not in {
                "openmathinstruct", "opencodeinstruct", "reliquary_logic_v2"}):
        raise ValueError("postproof stress dimensions or environment mismatch")
    settings = report.get("settings")
    if settings != runtime_settings():
        raise ValueError("postproof stress runtime settings mismatch")
    if settings["code_counterfactual_enabled"] or settings["constants"]["BFT_ENABLED"]:
        raise ValueError("postproof stress excludes counterfactual sandbox and BFT paths")
    times = [report.get(name) for name in (
        "seconds", "real_result_seconds", "synthetic_cpu_fixture_seconds")]
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in times) or not math.isclose(times[0], times[1] + times[2],
                                                   rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError("postproof stress timing is incomplete or not the measured sum")
    counts = report.get("helper_counts", {})
    expected = {name: rollout_count for name in REQUIRED_COVERAGE}
    expected.update({name: 1 for name in _GROUP_COVERAGE})
    is_code = report["environment"] == "opencodeinstruct"
    if is_code:
        expected.update({"code_semantic": rollout_count,
                         "fixture_dense_code_semantic": rollout_count,
                         "fixture_code_tokens": 1})
        # Real OpenCode exposes reward cases; small test/proxy environments
        # without the method do not need this optional entry-name lookup.
        if "admission_reward_cases" in counts or "entry_function_name" in counts:
            expected.update({"admission_reward_cases": 1, "entry_function_name": 1})
    if settings["auth_forensics_enabled"]:
        expected["fixture_forensic_appends"] = 1
        io = report.get("forensic_io") or {}
        minimum_rows = rollout_count * min(completion_tokens, settings["auth_forensics_max_findings"])
        if (type(io.get("rows")) is not int or io["rows"] < minimum_rows
                or (is_code and io["rows"] <= minimum_rows)
                or type(io.get("filesystem_device")) is not int
                or type(io.get("seconds")) not in (int, float)
                or not 0 < io["seconds"] <= report["synthetic_cpu_fixture_seconds"]):
            raise ValueError("native forensic appends were not completely measured")
    if counts != expected or any(type(value) is not int for value in counts.values()):
        raise ValueError("postproof stress helper coverage is incomplete")
    actual, fixtures = report.get("actual_results", []), report.get("cpu_only_fixtures", [])
    if len(actual) != rollout_count or len(fixtures) != rollout_count:
        raise ValueError("postproof stress result rows missing")
    for index, (row, fixture) in enumerate(zip(actual, fixtures, strict=True)):
        if (row.get("rollout_index") != index or fixture.get("rollout_index") != index
                or type(row.get("kernel_verdict", {}).get("all_passed")) is not bool
                or row.get("token_metrics", {}).get("token_count") != completion_tokens
                or row.get("sketch_metrics", {}).get("sketch_count") != (
                    STRESS_PROMPT_TOKENS + completion_tokens)
                or fixture.get("scope") != FIXTURE_SCOPE
                or fixture.get("full_scan_pi_old_count") != completion_tokens):
            raise ValueError("postproof stress full-token evidence missing")
        try:
            full_scan = fixture["full_scan_auth"][0]
            auth = fixture["max_findings_auth"][1]
            boxed = fixture["full_boxed"][1]
            if (full_scan is not True or auth["n_tokens"] != completion_tokens
                    or auth["findings"] != completion_tokens
                    or boxed["n_tokens"] < completion_tokens // 2):
                raise ValueError("CPU fixtures did not reach complete auth/boxed scans")
            if settings["auth_forensics_enabled"] and len(auth["finding_details"]) != min(
                    completion_tokens, settings["auth_forensics_max_findings"]):
                raise ValueError("CPU fixture omitted configured forensic contexts")
            if is_code:
                code = fixture["dense_code_semantic"][1]
                if code["n_spans"] <= 0 or code["findings"] <= 0:
                    raise ValueError("CPU Code fixture did not reach AST/findings branches")
                if settings["auth_forensics_enabled"] and len(code["finding_details"]) != min(
                        code["findings"], settings["auth_forensics_max_findings"]):
                    raise ValueError("CPU Code fixture omitted configured forensic contexts")
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("postproof CPU fixture branch evidence missing") from exc


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value: Any) -> Any:
    """Keep rejection infinities explicit in strict-JSON reports."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_finite(item) for item in value]
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    return value


def _summary(values: list[float]) -> dict:
    # Same scalar utility summary currently nested in _verify_expensive.
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"mean": None, "p50": None, "p90": None}
    return {"mean": sum(ordered) / len(ordered),
            "p50": ordered[round((len(ordered) - 1) * .5)],
            "p90": ordered[round((len(ordered) - 1) * .9)]}


def _fixture_tokens(tokenizer, *, code: bool) -> list[int]:
    """Full-size CPU text only; never paired with a claimed GPU proof.

    Code uses valid repeated assignments (dense AST-sensitive numeric spans).
    Boxed text uses one long numeric answer. Padding follows the closed fence
    or box, so neither parser is relying on syntactically truncated text.
    """
    def encode(repetitions):
        text = ("```python\ndef solve():\n" + "    x = 1 + 2\n" * repetitions
                + "    return x\n```\n") if code else "\\boxed{" + "1234567890" * repetitions + "}"
        return list(tokenizer.encode(text, add_special_tokens=False))

    low, high = 1, STRESS_POLICY_TOKENS
    best = encode(low)
    if not best or len(best) > STRESS_POLICY_TOKENS:
        raise ValueError("tokenizer cannot represent the bounded CPU fixture")
    while low <= high:
        middle = (low + high) // 2
        candidate = encode(middle)
        if len(candidate) <= STRESS_POLICY_TOKENS:
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    padding = list(tokenizer.encode(" x", add_special_tokens=False))
    if not padding:
        raise ValueError("tokenizer cannot encode CPU-only fixture padding")
    return best + [padding[-1]] * (STRESS_POLICY_TOKENS - len(best))


def measure_postproof_stress(*, commits, proofs, tokenizer, environment,
                             randomness, prompt_idx, checkpoint_revision,
                             proof_model=None, forensic_directory=None) -> dict:
    """Measure all native CPU gates without propagating any acceptance.

    ``environment`` is the loaded ENV; ``proof_model`` may be the real remote
    proxy carrying EOS metadata. The caller must retain its real mTLS receipts;
    this function cannot authenticate the origin of a Python ProofResult.
    ``seconds`` includes real results plus separately reported CPU fixtures.
    Any helper error makes ``complete`` false, but remaining helpers still run.
    This is a measured composite workload, not a universal complexity bound.
    """
    if len(commits) != STRESS_ROLLOUTS or len(proofs) != STRESS_ROLLOUTS:
        raise ValueError("stress requires exactly 16 full-length commits and ProofResults")
    name = environment.name
    if name not in {"openmathinstruct", "opencodeinstruct", "reliquary_logic_v2"}:
        raise ValueError("postproof stress is scoped to the three V1 environments")
    if not isinstance(randomness, str) or not randomness or not checkpoint_revision:
        raise ValueError("stress requires explicit randomness and immutable checkpoint context")
    for commit, proof in zip(commits, proofs, strict=True):
        meta = commit.get("rollout", {})
        if (meta.get("prompt_length") != STRESS_PROMPT_TOKENS
                or meta.get("completion_length") != STRESS_POLICY_TOKENS
                or len(commit.get("tokens", ())) != STRESS_PROMPT_TOKENS + STRESS_POLICY_TOKENS
                or meta.get("forced") or meta.get("episode")
                or len(commit.get("commitments", ())) != len(commit["tokens"])):
            raise ValueError("stress requires complete unforced single-turn tokens and sketches")
        if not isinstance(proof, v.ProofResult) or not proof.has_sparse_outputs:
            raise ValueError("stress requires sparse ProofResults, not a pass/fail stub")
        if any(len(getattr(proof, field)) != STRESS_POLICY_TOKENS for field in (
                "completion_chosen_probs", "completion_argmax_probs", "completion_argmax_ids")):
            raise ValueError("stress proof does not cover all policy tokens")
        if (len(proof.challenge_lp_indices) != c.CHALLENGE_K
                or len(proof.challenge_lp_values) != c.CHALLENGE_K
                or len(meta.get("token_logprobs", ())) not in {
                    STRESS_POLICY_TOKENS, len(commit["tokens"])}):
            raise ValueError("stress requires complete challenged and claimed logprobs")

    started = time.perf_counter()
    coverage: dict[str, int] = {}
    errors: list[dict] = []
    settings = runtime_settings()

    def call(label, function, *args, **kwargs):
        coverage[label] = coverage.get(label, 0) + 1
        try:
            return function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- retain helper failure and measure remaining paths
            errors.append({"helper": label, "error_type": type(exc).__name__})
            return None

    problem = call("get_problem", environment.get_problem, prompt_idx)
    entry_name = None
    if name == "opencodeinstruct" and callable(getattr(environment, "admission_reward_cases", None)):
        cases = call("admission_reward_cases", environment.admission_reward_cases, problem)
        entry_name = call("entry_function_name", _entry_function_name, cases)
    eos = call("resolve_eos", resolve_eos_token_ids, proof_model, tokenizer) or set()
    # The V1 profile has no BFT; unavailable atomic think token is benign just
    # as in _verify_expensive, not evidence of incomplete sparse coverage.
    try:
        close_ids = set(think_close_token_ids(tokenizer))
    except (ValueError, TypeError, AttributeError):
        close_ids = set()
    forensic_args = {"include_findings": settings["auth_forensics_enabled"],
                         "max_findings": settings["auth_forensics_max_findings"],
                         "context_chars": settings["auth_forensics_context_chars"]}
    groups = []
    for index, (commit, proof) in enumerate(zip(commits, proofs, strict=True)):
        meta, tokens = commit["rollout"], commit["tokens"]
        positions = call("policy_token_positions", v.policy_token_positions, list(tokens), meta) or []
        completion = [tokens[position] for position in positions]
        row = {"rollout_index": index, "kernel_verdict": {
            "all_passed": proof.all_passed, "passed": proof.passed, "checked": proof.checked,
            "sketch_diff_max": proof.sketch_diff_max, "seed_n_hard_mismatch": proof.seed_n_hard_mismatch}}
        rollout_hash = call("rollout_hash", b.compute_rollout_hash, tokens)
        row["rollout_hash"] = None if rollout_hash is None else rollout_hash.hex()
        row["completion_chars"] = len(call("completion_decode", tokenizer.decode, completion,
                                                skip_special_tokens=True) or "")
        row["token_metrics"] = call("token_metrics", b.token_degeneracy_metrics, completion)
        row["sketch_metrics"] = call("sketch_metrics", b.sketch_commitment_metrics, commit["commitments"])
        uniforms = call("seed_u", lambda index=index, positions=positions: [u_at(randomness, prompt_idx, checkpoint_revision, index, j)
                                           for j in range(len(positions))])
        row["seed_u_sha256"] = call("seed_u_hash", _hash, uniforms)
        pi_old = call("pi_old", b._verify_logprobs_for_training, proof, len(positions))
        row["pi_old_count"] = None if pi_old is None else len(pi_old)
        row["eos_padding"] = call("eos_padding", v.has_eos_padding, commit, tokenizer, proof_model)
        row["termination"] = call("termination", v.verify_termination, commit, tokenizer, proof,
                                  proof_model, env_name=name)
        row["cap_truncation"] = call("cap_truncation", v.is_cap_truncation, commit, tokenizer, proof,
                                     proof_model, env_name=name)
        row["natural_bft_cap"] = call("natural_bft_cap", v.is_natural_bft_cap_candidate,
                                      commit, tokenizer, env_name=name)
        force = call("force_span", v.validate_force_span, tokens, meta, [], STRESS_PROMPT_TOKENS,
                     thinking_budget=c.BFT_THINKING_BUDGET, think_close_ids=set())
        exempt = force[1] if force is not None else set()
        row["force_span_valid"] = None if force is None else force[0]
        row["termination_path"] = call("termination_classification", b.classify_bft_termination,
            tokens, prompt_length=STRESS_PROMPT_TOKENS, completion_length=STRESS_POLICY_TOKENS,
            eos_ids=eos, think_close_ids=close_ids, validated_force_span=None,
            thinking_budget=c.BFT_THINKING_BUDGET, answer_budget=c.BFT_ANSWER_BUDGET)
        common = {"tokens": tokens, "prompt_length": STRESS_PROMPT_TOKENS,
                      "completion_length": STRESS_POLICY_TOKENS, "proof": proof}
        row["logprob"] = call("logprob", v.verify_logprobs_claim, **common,
                              claimed_logprobs=meta["token_logprobs"])
        row["distribution"] = call("distribution", v.evaluate_token_distribution, **common,
                                   exempt_positions=exempt)
        row["boxed"] = call("boxed", v.evaluate_boxed_answer_probability, **common, tokenizer=tokenizer)
        row["token_auth"] = call("token_auth", v.evaluate_token_authenticity, **common,
                                 tokenizer=tokenizer, exempt_positions=exempt)
        row["all_token_auth"] = call("all_token_auth", v.evaluate_all_token_auth_shadow, **common,
                                     tokenizer=tokenizer, exempt_positions=exempt, **forensic_args)
        if name == "opencodeinstruct":
            row["code_semantic"] = call("code_semantic", v.evaluate_code_semantic_token_authenticity,
                                       **common, tokenizer=tokenizer, entry_name=entry_name, **forensic_args)
        row["chosen_nll"] = call("utility_nll", lambda proof=proof: _summary([
            -math.log(max(float(probability), 1e-45)) for probability in proof.completion_chosen_probs]))
        row["entropy"] = call("utility_entropy", _summary, list(proof.completion_entropies))
        groups.append(row)
    seed_counts = [(proof.seed_n_stochastic, proof.seed_n_match) for proof in proofs]
    rewards = [float(commit["rollout"].get("total_reward", 0.)) for commit in commits]
    reward_shape = call("reward_shape", b.detect_reward_shape_manipulation, rewards,
                        [STRESS_POLICY_TOKENS] * STRESS_ROLLOUTS,
                        [bool(row["cap_truncation"]) for row in groups])
    group_result = {
        "reward_sigma": call("rewards_std", b.rewards_std, rewards),
        "reward_shape": reward_shape.to_log_dict() if reward_shape else None,
        "seed_group_would_reject": call("seed_group_verdict", b._forced_seed_verdict,
                                        sum(x for x, _ in seed_counts), sum(y for _, y in seed_counts), True),
        # Evaluate each rollout independently as well: native aggregate exits
        # at the first failing rollout, which does not cover the full group.
        "seed_rollout_would_reject": [call("seed_rollout_verdict", b._forced_seed_rollout_reject,
                                           [pair], True) for pair in seed_counts],
    }
    actual_seconds = time.perf_counter() - started

    fixture_started = time.perf_counter()
    boxed_tokens = call("fixture_boxed_tokens", _fixture_tokens, tokenizer, code=False)
    code_tokens = call("fixture_code_tokens", _fixture_tokens, tokenizer, code=True) if name == "opencodeinstruct" else None
    fixtures = []
    for index, (commit, proof) in enumerate(zip(commits, proofs, strict=True)):
        # dataclasses.replace makes distinct vectors and retains the original
        # GRAIL verdict. These probability arrays are CPU fixtures only.
        scan = replace(proof, completion_chosen_probs=[.5] * STRESS_POLICY_TOKENS,
                       completion_argmax_probs=[1.] * STRESS_POLICY_TOKENS)
        findings = replace(proof, completion_chosen_probs=[0.] * STRESS_POLICY_TOKENS,
                           completion_argmax_probs=[1.] * STRESS_POLICY_TOKENS)
        row = {"rollout_index": index, "scope": FIXTURE_SCOPE}
        common = {"tokens": commit["tokens"], "prompt_length": STRESS_PROMPT_TOKENS,
                      "completion_length": STRESS_POLICY_TOKENS, "tokenizer": tokenizer}
        row["full_scan_auth"] = call("fixture_full_scan_auth", v.evaluate_token_authenticity,
                                     scan, **common)
        scanned_lp = call("fixture_full_scan_pi_old", b._verify_logprobs_for_training,
                          scan, STRESS_POLICY_TOKENS)
        row["full_scan_pi_old_count"] = None if scanned_lp is None else len(scanned_lp)
        row["max_findings_auth"] = call("fixture_max_findings_auth", v.evaluate_all_token_auth_shadow,
                                        findings, **common, **forensic_args)
        if boxed_tokens is not None:
            row["full_boxed"] = call("fixture_full_boxed", v.evaluate_boxed_answer_probability,
                tokens=commit["tokens"][:STRESS_PROMPT_TOKENS] + boxed_tokens,
                prompt_length=STRESS_PROMPT_TOKENS, completion_length=STRESS_POLICY_TOKENS,
                proof=scan, tokenizer=tokenizer)
        if code_tokens is not None:
            row["dense_code_semantic"] = call("fixture_dense_code_semantic", v.evaluate_code_semantic_token_authenticity,
                tokens=commit["tokens"][:STRESS_PROMPT_TOKENS] + code_tokens,
                prompt_length=STRESS_PROMPT_TOKENS, completion_length=STRESS_POLICY_TOKENS,
                proof=findings, tokenizer=tokenizer, entry_name="solve", **forensic_args)
        fixtures.append(row)
    shape = call("fixture_reward_shape", b.detect_reward_shape_manipulation,
                 [1.] * (STRESS_ROLLOUTS // 2) + [0.] * (STRESS_ROLLOUTS // 2),
                 [STRESS_POLICY_TOKENS] * STRESS_ROLLOUTS, [True] * STRESS_ROLLOUTS)
    forensic_io = None
    if settings["auth_forensics_enabled"]:
        forensic_io = call("fixture_forensic_appends", _measure_forensic_appends,
            fixtures, environment=name, prompt_idx=prompt_idx, directory=forensic_directory)
    fixture_seconds = time.perf_counter() - fixture_started
    report = _finite({
        "scope": SCOPE, "admissible": False,
        "complete": not errors, "seconds": actual_seconds + fixture_seconds,
        "real_result_seconds": actual_seconds, "synthetic_cpu_fixture_seconds": fixture_seconds,
        "rollout_count": len(commits), "prompt_tokens": STRESS_PROMPT_TOKENS,
        "policy_tokens": STRESS_POLICY_TOKENS, "environment": name,
        "settings": settings, "helper_counts": coverage, "errors": errors,
        "forensic_io": forensic_io,
        "actual_results": groups, "group_results": group_result,
        "cpu_only_fixtures": fixtures, "synthetic_reward_shape": shape.to_log_dict() if shape else None,
        "limits": ["Caller must retain authentic remote proof receipts; Python inputs do not prove provenance.",
                   "Native forensic appends measured only in disposable private files; external logs and Code counterfactual sandbox excluded.",
                   "V1 single-turn/unforced profile only; short-answer fallback and BFT/episodes are separate paths.",
                   "Actual results may return early internally; CPU fixtures cover full auth/pi_old, boxed and Code AST scans.",
                   "Composite workload is conservative sampled coverage, not a universal worst-case complexity proof."],
    })
    if report["complete"]:
        try:
            validate_postproof_measurement(report, rollout_count=STRESS_ROLLOUTS,
                                          completion_tokens=STRESS_POLICY_TOKENS)
        except ValueError as exc:
            report["complete"] = False
            report["errors"].append({"helper": "coverage", "error_type": "ValueError",
                                     "message": str(exc)})
    return report


def _measure_forensic_appends(fixtures, *, environment, prompt_idx, directory):
    """Real production serializers/appends; temporary files never enter history."""
    from reliquary.validator.auth_forensics import (
        record_all_token_auth_findings,
        record_code_semantic_auth_findings,
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nonadmissible-proof-stress-", dir=directory) as temporary:
        root = Path(temporary)
        expected = 0
        for index, fixture in enumerate(fixtures):
            fields = {"window_start": 1, "env_name": environment, "miner_hotkey": "NONADMISSIBLE_STRESS",
                "prompt_idx": prompt_idx, "rollout_idx": index, "rollout_reward": 1., "reward_positive": True,
                "prompt_length": STRESS_PROMPT_TOKENS, "completion_length": STRESS_POLICY_TOKENS}
            for key, writer, filename in (
                ("max_findings_auth", record_all_token_auth_findings, "all-token.jsonl"),
                ("dense_code_semantic", record_code_semantic_auth_findings, "code.jsonl"),
            ):
                if key not in fixture:
                    continue
                metrics = fixture[key][1]
                expected += len(metrics["finding_details"])
                writer(metrics=metrics, path=root/filename, **fields)
        # Production writers are intentionally fail-soft. Qualification is not:
        # read back every actual line so a swallowed write error cannot pass.
        rows = [json.loads(line) for path in root.glob("*.jsonl") for line in path.read_text().splitlines()]
        if len(rows) != expected or expected <= 0:
            raise ValueError("native forensic writer did not persist all fixture findings")
        device = os.stat(root).st_dev
    return {"rows": len(rows), "filesystem_device": device, "seconds": time.perf_counter()-started}
