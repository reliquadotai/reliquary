"""Run one SciCode step's test script; print {"passed": bool} as the last line.

argv: <spec.json> with {"script": ..., "timeout": seconds}. The script is the
upstream test file (dependencies, earlier steps' code, this step's code, the
step's asserts against the HDF5 targets); it passes when it exits 0, as in
scicode-bench's test_generated_code.py. Needs the SciCode grading image's
Python (numpy/scipy/sympy/h5py, `scicode`, the targets file).
"""

import json
import subprocess
import sys


def main():
    with open(sys.argv[1]) as handle:
        spec = json.load(handle)
    try:
        proc = subprocess.run([sys.executable, "-c", spec["script"]], input="",
                              capture_output=True, text=True, timeout=spec["timeout"])
        passed = proc.returncode == 0
    except subprocess.TimeoutExpired:
        passed = False
    print(json.dumps({"passed": passed}))


if __name__ == "__main__":
    main()
