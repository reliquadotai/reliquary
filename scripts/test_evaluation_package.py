"""Check an installed evaluation wheel from outside its source checkout."""

import importlib
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path


def test_installed_wheel_retains_manifest_resources():
    installed = distribution("reliquary")
    package = Path(installed.locate_file("reliquary")).resolve()
    assert Path(find_spec("reliquary").origin).resolve().parent == package
    assert "tests/fixtures/reliquarylogic_v1.jsonl" in {
        str(path) for path in installed.files or ()
    }
    # Importing the real registry checks every installed manifest and bound file digest.
    importlib.import_module("reliquary.environment.registry")


if __name__ == "__main__":
    test_installed_wheel_retains_manifest_resources()
    print("Installed evaluation wheel manifest resources passed.")
