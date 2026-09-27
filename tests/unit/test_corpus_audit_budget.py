"""The audit's per-pass token budget is set by the operator for the card and
checkpoint: 131,072 tokens overran an H100 beside Qwen3.8-27B (2026-09-27)."""

import importlib


def test_the_audit_budget_follows_the_environment(monkeypatch):
    import reliquary.validator.corpus_auditor as auditor

    monkeypatch.setenv("RELIQUARY_CORPUS_AUDIT_BATCH_TOKENS", "32768")
    assert importlib.reload(auditor).AUDIT_BATCH_TOKENS == 32768
    monkeypatch.delenv("RELIQUARY_CORPUS_AUDIT_BATCH_TOKENS")
    assert importlib.reload(auditor).AUDIT_BATCH_TOKENS == 131072
