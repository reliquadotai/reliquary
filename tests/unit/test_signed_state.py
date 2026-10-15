"""`signed_state.swe_patch`: the git patch a signed SWE episode's export row carries,
read from the verifiers-bridge state the gateway graded. The fixtures are states
captured on a live gVisor gateway (reliquary-swe 0.3.0, SWE-smith oauthlib tasks):
a gold patch, an untouched repository and a planted conftest.py."""

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("reliquary_sandbox_service.episodes.state_archive")

from reliquary.corpus.signed_state import (  # noqa: E402
    BRIDGE_STATE_MAGIC, SignedStateUnreadable, swe_patch,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "signed_swe"


def captured(index: int) -> str:
    # The submission's final_diff: the state bytes as UTF-8 text (its sha256 is the
    # final record's state_sha256).
    return gzip.decompress((FIXTURES / f"state-{index}.bin.gz").read_bytes()).decode("utf-8")


@pytest.mark.parametrize("index,sha256,head", [
    (0, "0c110bb116aa5b6b42b4b2698c895e7cbe18fe8f8c199927d6316c7ed0ab16e7",
     "diff --git a/oauthlib/oauth2/rfc6749/utils.py b/oauthlib/oauth2/rfc6749/utils.py\n"),
    (3, "94579153a0745dedf7e925c2654c1eabc755e9e2b76b70a0fb8498596c284424",
     "diff --git a/conftest.py b/conftest.py\nnew file mode 100644\n"),
])
def test_a_captured_state_yields_its_git_patch(index, sha256, head):
    state = captured(index)
    assert state.startswith(BRIDGE_STATE_MAGIC.decode())
    patch = swe_patch(state)
    assert patch.startswith(head) and patch.endswith("\n")
    assert hashlib.sha256(patch.encode()).hexdigest() == sha256
    assert "\x00" not in patch and BRIDGE_STATE_MAGIC.decode() not in patch


def test_an_untouched_repository_exports_an_empty_patch():
    assert swe_patch(captured(1)) == ""


def test_no_state_and_a_non_bridge_state():
    assert swe_patch("") == ""
    plain = "diff --git a/x b/x\n"
    assert swe_patch(plain) == plain


def _state(header: bytes, archive: bytes) -> str:
    return (BRIDGE_STATE_MAGIC + header + b"\n" + archive).decode("utf-8")


def test_a_state_without_the_artifact_root_is_an_empty_patch():
    header = json.dumps({"missing_required": ["/logs/artifacts"], "present": []}).encode()
    assert swe_patch(_state(header, b"")) == ""


@pytest.mark.parametrize("header,archive", [
    (b"not json", b""),
    (json.dumps({"present": "/logs/artifacts"}).encode(), b""),
    (json.dumps({"missing_required": [], "present": ["/logs/artifacts"]}).encode(), b"junk"),
])
def test_an_unreadable_state_is_refused(header, archive):
    with pytest.raises(SignedStateUnreadable):
        swe_patch(_state(header, archive))


def test_a_member_outside_its_root_is_refused():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        manifest = json.dumps({"roots": ["/logs/artifacts"], "version": 1}).encode()
        for name, data in ((".reliquary-archive.json", manifest), ("0/../../etc/x", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    header = json.dumps({"missing_required": [], "present": ["/logs/artifacts"]}).encode()
    with pytest.raises(SignedStateUnreadable):
        swe_patch(_state(header, buffer.getvalue()))
