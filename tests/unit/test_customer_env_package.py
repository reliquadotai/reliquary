import hashlib
import io
import json
import tarfile

import pytest

from reliquary.customer_env import check_order_env, env_id, parse_env_id
from reliquary.customer_env.package import PackageInvalid, read_package

H = "a" * 64
PYPROJECT = '[project]\nname = "my-env"\nversion = "1"\n[tool.reliquary]\ntaskset = "my-env"\nargs = {split = "test"}\n'
LOCK = f"verifiers==0.2.0 \\\n    --hash=sha256:{H}\nnumpy==2.0.0 --hash=sha256:{H} --hash=sha256:{'b' * 64}  # numeric\n"
EXAMPLES = "\n".join(json.dumps(e) for e in (
    {"key": "t0", "answer": "4", "expected_score": 1.0},
    {"key": "t0", "answer": "5", "expected_score": 0.0},
))


def archive(files: dict, *, root: str | None = "my-env", extra=None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            body = text.encode()
            info = tarfile.TarInfo(f"{root}/{name}" if root else name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
        for info in extra or ():
            tar.addfile(info)
    return buf.getvalue()


def good(**override):
    return {"pyproject.toml": PYPROJECT, "requirements.lock": LOCK, "examples.jsonl": EXAMPLES,
            "src/my_env/__init__.py": "", **override}


@pytest.mark.parametrize("root", ["my-env", None])
def test_a_good_package_reads_its_manifest(root):
    data = archive(good(), root=root)
    pkg = read_package(data)
    assert pkg.sha256 == hashlib.sha256(data).hexdigest()
    assert (pkg.taskset, pkg.args, pkg.kind) == ("my-env", {"split": "test"}, "single_turn")
    assert pkg.requirements[0] == f"verifiers==0.2.0 --hash=sha256:{H}"
    assert len(pkg.requirements) == 2 and len(pkg.examples) == 2
    assert pkg.root == (root or "")


@pytest.mark.parametrize("files, reason", [
    (good(**{"requirements.lock": "verifiers==0.2.0\n"}), "--hash"),
    (good(**{"requirements.lock": "verifiers>=0.2 --hash=sha256:" + H}), "pinned"),
    (good(**{"requirements.lock": "--index-url https://evil\n"}), "not allowed"),
    (good(**{"requirements.lock": "git+https://x/y.git --hash=sha256:" + H}), "pinned"),
    (good(**{"pyproject.toml": "[project]\nname='x'\n"}), "tool.reliquary"),
    (good(**{"pyproject.toml": PYPROJECT + "image = 'x'\n"}), "unknown keys"),
    (good(**{"pyproject.toml": PYPROJECT.replace('taskset = "my-env"', 'taskset = "../x"')}), "taskset"),
    (good(**{"examples.jsonl": json.dumps({"key": "a", "answer": "x", "expected_score": 1})}), "2 to 64"),
    (good(**{"examples.jsonl": EXAMPLES.replace("0.0", "1.0")}), "below"),
    (good(**{"examples.jsonl": EXAMPLES.replace("1.0", "true")}), "expected_score"),
    ({"pyproject.toml": PYPROJECT, "requirements.lock": LOCK}, "examples.jsonl is missing"),
])
def test_a_bad_package_is_refused_with_its_reason(files, reason):
    with pytest.raises(PackageInvalid, match=reason):
        read_package(archive(files))


def test_links_devices_and_escaping_paths_are_refused():
    link = tarfile.TarInfo("my-env/evil")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    with pytest.raises(PackageInvalid, match="only regular files"):
        read_package(archive(good(), extra=[link]))
    with pytest.raises(PackageInvalid, match="unsafe path"):
        read_package(archive({**good(), "../../etc/x": "x"}))
    with pytest.raises(PackageInvalid, match="unsafe path"):
        read_package(archive({"/abs": "x", **good()}, root=None))


def test_a_bomb_is_refused_on_its_declared_sizes(monkeypatch):
    from reliquary.customer_env import package

    monkeypatch.setattr(package, "MAX_UNPACKED_BYTES", 1000)
    with pytest.raises(PackageInvalid, match="unpacks to over"):
        read_package(archive(good(**{"big.bin": "x" * 2000})))
    with pytest.raises(PackageInvalid, match="tar.gz"):
        read_package(b"not an archive")


def test_env_ids_and_order_checks():
    e = env_id("acct_1", "my-env", H)
    assert parse_env_id(str(e)) == e and str(e) == f"customer:acct_1/my-env@{H}"
    for bad in (f"customer:acct/My_Env@{H}", "customer:acct/x@123", f"acct/x@{H}"):
        with pytest.raises(ValueError):
            parse_env_id(bad)
    assert check_order_env(str(e), account="acct_1", product="eval", uses=("eval",)) == e
    with pytest.raises(ValueError, match="not_yours"):
        check_order_env(str(e), account="other", product="eval", uses=("eval",))
    with pytest.raises(ValueError, match="not_qualified_for: rl"):
        check_order_env(str(e), account="acct_1", product="rl", uses=("eval",))
