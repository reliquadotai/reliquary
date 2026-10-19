from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from reliquary.validator.gpu_registry import AdmissionError, GpuRegistry, registry_from_env
from .conftest import NOW


@pytest.fixture
def registry(tmp_path, authority, now):
    value = GpuRegistry(tmp_path / "gpu.sqlite3", validator="validator-test", run="run-test", mode="required",
                        verifier=authority.verifier, clock=lambda: now[0])
    yield value
    value.close()


def admit(registry, authority, hotkey="miner-1", **kwargs):
    challenge = registry.challenge(hotkey)
    return registry.attest(hotkey, challenge["challenge_id"], authority.bundle(challenge["nonce"], **kwargs),
                           {"gpu_name": "declared-name-not-a-hardware-claim", "vram_bytes": 80 * 1024**3})


def test_required_admission_and_signed_declarations_are_distinct(registry, authority):
    with pytest.raises(AdmissionError, match="required"):
        registry.require_admission("miner-1")
    status = admit(registry, authority)
    assert registry.require_admission("miner-1")["attested"]
    assert status["hardware"]["hardware_model"] == "GH100 TEST ONLY"
    assert status["declared"]["gpu_name"] == "declared-name-not-a-hardware-claim"
    assert status["execution_attested"] is False and status["declarations_attested"] is False
    assert "ueid" not in status["hardware"]
    assert registry.fleet()["active_attested_devices"] == 1
    assert "miner-1" not in str(registry.fleet())


def test_challenge_is_single_use_even_under_concurrent_requests(registry, authority):
    challenge = registry.challenge("miner-1")
    bundle = authority.bundle(challenge["nonce"])
    def attempt(_):
        try:
            registry.attest("miner-1", challenge["challenge_id"], bundle, {})
            return "accepted"
        except AdmissionError as exc:
            return exc.reason
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(attempt, range(2)))
    assert sorted(results) == ["accepted", "gpu_challenge_used_or_expired"]


def test_challenge_cannot_be_stolen_or_replayed_after_restart(tmp_path, registry, authority, now):
    challenge = registry.challenge("miner-1")
    bundle = authority.bundle(challenge["nonce"])
    with pytest.raises(AdmissionError, match="unknown"):
        registry.attest("other-miner", challenge["challenge_id"], bundle, {})
    registry.attest("miner-1", challenge["challenge_id"], bundle, {})
    other = GpuRegistry(tmp_path / "gpu.sqlite3", validator="validator-test", run="run-test", mode="required",
                        verifier=authority.verifier, clock=lambda: now[0])
    try:
        assert other.status("miner-1")["attested"]
        with pytest.raises(AdmissionError, match="used"):
            other.attest("miner-1", challenge["challenge_id"], bundle, {})
    finally:
        other.close()


def test_invalid_proof_consumes_challenge_without_creating_admission(registry, authority):
    challenge = registry.challenge("miner-1")
    with pytest.raises(AdmissionError, match="nonce"):
        registry.attest("miner-1", challenge["challenge_id"], authority.bundle("ab" * 32), {})
    with pytest.raises(AdmissionError, match="used"):
        registry.attest("miner-1", challenge["challenge_id"], authority.bundle(challenge["nonce"]), {})
    assert registry.fleet()["active_attested_devices"] == 0


def test_same_gpu_cannot_have_two_active_hotkeys(registry, authority):
    admit(registry, authority, "miner-1")
    with pytest.raises(AdmissionError, match="gpu_already_admitted"):
        admit(registry, authority, "miner-2")
    assert not registry.status("miner-2")["attested"]


def test_two_database_connections_cannot_admit_the_same_gpu_concurrently(tmp_path, registry, authority, now):
    other = GpuRegistry(tmp_path / "gpu.sqlite3", validator="validator-test", run="run-test", mode="required",
                        verifier=authority.verifier, clock=lambda: now[0])
    first, second = registry.challenge("miner-1"), other.challenge("miner-2")
    def attempt(args):
        store, hotkey, challenge = args
        try:
            store.attest(hotkey, challenge["challenge_id"], authority.bundle(challenge["nonce"]), {})
            return "accepted"
        except AdmissionError as exc:
            return exc.reason
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(attempt, ((registry, "miner-1", first), (other, "miner-2", second))))
        assert sorted(results) == ["accepted", "gpu_already_admitted"]
        assert registry.fleet()["active_attested_devices"] == 1
    finally:
        other.close()


def test_hotkey_cannot_replace_active_gpu_but_can_refresh_it(registry, authority):
    first = admit(registry, authority)
    assert admit(registry, authority)["device_id"] == first["device_id"]
    with pytest.raises(AdmissionError, match="hotkey_already_has_gpu"):
        admit(registry, authority, device={"ueid": "different-device"})


def test_expiry_and_renewal_never_rewrite_lease_provenance(registry, authority, now):
    first = admit(registry, authority)
    registry.bind_leases("miner-1", [SimpleNamespace(lease_id="lease-test")], first)
    now[0] += 301
    assert registry.fleet()["active_attested_devices"] == 0
    with pytest.raises(AdmissionError):
        registry.require_admission("miner-1")
    admit(registry, authority, device={"iat": NOW + 301, "nbf": NOW + 301, "exp": NOW + 3600},
          overall={"iat": NOW + 301, "nbf": NOW + 301, "exp": NOW + 3600})
    assert registry.lease_hardware("lease-test")["verified_at"] == first["verified_at"]
    assert registry.status("miner-1")["verified_at"] > first["verified_at"]


def test_expired_challenge_cannot_be_used(registry, authority, now):
    challenge = registry.challenge("miner-1")
    now[0] += 121
    with pytest.raises(AdmissionError, match="expired"):
        registry.attest("miner-1", challenge["challenge_id"], authority.bundle(challenge["nonce"]), {})


def test_slow_verification_cannot_extend_challenge_or_token_lifetime(registry, authority, now):
    verifier = registry.verifier
    class Slow:
        def verify(self, *args, **kwargs):
            report = verifier.verify(*args, **kwargs)
            now[0] += 121
            return report
    registry.verifier = Slow()
    with pytest.raises(AdmissionError, match="expired"):
        admit(registry, authority)
    assert not registry.status("miner-1")["attested"]


def test_observe_collects_information_without_blocking_unattested_miners(tmp_path, authority):
    registry = GpuRegistry(tmp_path / "observe.db", validator="v", run="r", mode="observe", verifier=authority.verifier)
    try:
        assert registry.require_admission("unknown")["admission_allowed"]
        assert registry.require_admission("unknown")["attested"] is False
    finally:
        registry.close()


def test_challenges_are_bound_to_validator_and_run(tmp_path, registry, authority, now):
    challenge = registry.challenge("miner-1")
    for validator, run in (("other-validator", "run-test"), ("validator-test", "other-run")):
        other = GpuRegistry(tmp_path / "gpu.sqlite3", validator=validator, run=run, mode="required",
                            verifier=authority.verifier, clock=lambda: now[0])
        try:
            with pytest.raises(AdmissionError, match="unknown"):
                other.attest("miner-1", challenge["challenge_id"], authority.bundle(challenge["nonce"]), {})
        finally:
            other.close()


def test_challenge_rate_is_persisted(registry):
    for _ in range(6):
        registry.challenge("miner-1")
    with pytest.raises(AdmissionError, match="rate") as error:
        registry.challenge("miner-1")
    assert error.value.status == 429


def test_invalid_config_fails_startup_and_default_off_needs_no_extra(tmp_path):
    assert registry_from_env(tmp_path / "unused", validator="v", run="r", env={}) is None
    assert not (tmp_path / "unused").exists()
    with pytest.raises(ValueError):
        registry_from_env(tmp_path / "unused", validator="v", run="r", env={"RELIQUARY_RL_GPU_ADMISSION": "typo"})
