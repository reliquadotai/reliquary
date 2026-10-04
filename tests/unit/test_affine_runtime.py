"""CPU-only boundary checks: fake upstream execution is never mining evidence."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from reliquary.integrations import affine
from reliquary.integrations.affine import (
    AffineConfig, AffineRunner, AffineRuntimeError, load_config, validate_config, verify_checkout,
    reconcile_state,
)


class AffineRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.checkout = self.root / "upstream"
        (self.checkout / "subnet").mkdir(parents=True)
        self.credential = self.root / "credential"
        self.credential.write_bytes(b"fixture-only-do-not-read")
        self.credential.chmod(0o600)

    def config(self, source):
        (self.checkout / "subnet/source_bootstrap.py").write_text(source)
        def git(*args):
            return subprocess.check_output(["git", "-C", str(self.checkout), *args], stderr=subprocess.DEVNULL)
        git("init", "-q")
        git("add", "subnet/source_bootstrap.py")
        git("-c", "user.name=Runtime Check", "-c", "user.email=check@example.invalid", "commit", "-qm", "runtime fixture")
        return AffineConfig(
            upstream_checkout=self.checkout, upstream_revision=git("rev-parse", "HEAD").decode().strip(),
            python=Path(sys.executable), authority="a" * 64,
            current_url="https://fixture.r2.cloudflarestorage.com/current?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=private-signature",
            state_dir=self.root / "state", source_cache_dir=self.root / "source-cache",
            key_file=self.credential, env_id="math", indices=(2, 7),
            retry_seconds=0.01, process_timeout_seconds=5,
        )

    def test_native_process_is_isolated_and_completion_is_not_acceptance(self):
        config = self.config("""
import json, os, pathlib, sys
assert sys.flags.isolated and sys.dont_write_bytecode
assert 'PYTHONPATH' not in os.environ and 'UNRELATED_SECRET' not in os.environ
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
state = pathlib.Path(sys.argv[sys.argv.index('--state') + 1])
(state / 'observed.json').write_text(json.dumps({'args': sys.argv[1:], 'home': os.environ.get('HOME'),
                                                'cache': os.environ['HF_HOME']}))
print(sys.argv[sys.argv.index('--current-url') + 1])
""")
        old_env = os.environ.copy()
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(old_env)))
        os.environ.update(PYTHONPATH=str(self.root), UNRELATED_SECRET="host-only")
        result = AffineRunner(config).run()
        self.assertEqual(result["stage"], "bootstrap_completed")
        self.assertIsNone(result["uploaded"])
        self.assertIsNone(result["accepted"])
        self.assertIsNone(result["trained"])
        self.assertNotIn("private-signature", json.dumps(result) + repr(config))
        self.assertEqual(self.credential.read_bytes(), b"fixture-only-do-not-read")
        observed = json.loads((config.state_dir / "observed.json").read_text())
        self.assertEqual(observed["home"], os.environ.get("HOME"))
        self.assertEqual(observed["cache"], str(config.state_dir / "model-cache"))
        args = observed["args"]
        self.assertIn("--once", args)
        self.assertEqual(args[args.index("--indices") + 1:], ["2", "7"])
        self.assertEqual((config.state_dir / "runtime.log").stat().st_mode & 0o777, 0o600)
        self.assertFalse(list(self.checkout.rglob("__pycache__")))
        first, second = AffineRunner(config)._command(), AffineRunner(config)._command()
        caches = [command[command.index("-X") + 1] for command in (first, second)]
        self.assertNotEqual(*caches)
        for prefix in caches:
            self.assertTrue(prefix.startswith("pycache_prefix=" + str(config.state_dir)))
            self.assertFalse(Path(prefix.split("=", 1)[1]).exists())

    def test_refuses_dirty_checkout_untrusted_paths_and_ambiguous_config(self):
        config = self.config("pass\n")
        validate_config(config)
        verify_checkout(config)
        for changed in (replace(config, authority="bad"), replace(config, cap_file=self.credential),
                        replace(config, state_dir=self.checkout / "state"),
                        replace(config, indices=(2, 2)), replace(config, search_budget=129),
                        replace(config, manifest_snapshot_file=self.credential)):
            with self.assertRaises(AffineRuntimeError):
                validate_config(changed)
        self.credential.chmod(0o644)
        with self.assertRaises(AffineRuntimeError):
            validate_config(config)
        self.credential.chmod(0o600)
        path = self.root / "config.affine-private.json"
        path.write_text('{"current_url":"secret-one","current_url":"secret-two"}')
        path.chmod(0o600)
        with self.assertRaisesRegex(AffineRuntimeError, "malformed"):
            load_config(path)
        (self.checkout / "subnet/source_bootstrap.py").write_text("raise RuntimeError('modified')\n")
        with self.assertRaises(AffineRuntimeError):
            AffineRunner(config).run()
        self.assertFalse(config.state_dir.exists())

    def test_retry_limit_and_state_lock(self):
        config = self.config("raise SystemExit(9)\n")
        config = replace(config, retry_attempts=2)
        result = AffineRunner(config).run()
        self.assertEqual((result["stage"], result["attempts"], result["exit_code"]), ("failed", 2, 9))
        import fcntl
        with (config.state_dir / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(AffineRuntimeError, "another runner"):
                AffineRunner(config).run()

    def test_budget_and_cancel_remove_child_processes(self):
        config = self.config("""
import pathlib, subprocess, sys, time
state = pathlib.Path(sys.argv[sys.argv.index('--state') + 1])
subprocess.Popen([sys.executable, '-I', '-B', '-c',
                 'import pathlib,time,sys;time.sleep(1);pathlib.Path(sys.argv[1]).write_text("orphan")',
                 str(state / 'orphan')])
(state / 'started').write_text('ready')
time.sleep(30)
""")
        result = AffineRunner(config).run(max_seconds=0.3)
        self.assertEqual(result["stage"], "budget_exhausted")
        time.sleep(1.1)
        self.assertFalse((config.state_dir / "orphan").exists())
        stop = threading.Event()
        timer = threading.Timer(0.3, stop.set)
        timer.start()
        try:
            result = AffineRunner(config).run(stop_event=stop)
        finally:
            timer.cancel()
        self.assertEqual(result["stage"], "cancelled")
        # Cancellation released the state lock; a subsequent run can acquire it.
        stop.clear()
        self.assertEqual(AffineRunner(config).run(max_seconds=0.1)["stage"], "budget_exhausted")

    def test_crashed_parent_refuses_resume_and_terminal_state_follows_cleanup(self):
        config = self.config("""
import pathlib, sys, time
state = pathlib.Path(sys.argv[sys.argv.index('--state') + 1])
(state / 'started').write_text('ready')
time.sleep(30)
""")
        original_terminate = affine._terminate
        observed = []

        def check_cleanup(process):
            before = json.loads((config.state_dir / "runner.json").read_text())
            self.assertEqual((before["stage"], before["process_group"]), ("running", process.pid))
            original_terminate(process)
            self.assertEqual(json.loads((config.state_dir / "runner.json").read_text())["stage"], "running")
            observed.append(process.pid)

        with mock.patch.object(affine, "_terminate", side_effect=check_cleanup):
            self.assertEqual(AffineRunner(config).run(max_seconds=0.2)["stage"], "budget_exhausted")
        self.assertEqual(len(observed), 1)
        self.assertEqual(json.loads((config.state_dir / "runner.json").read_text())["stage"], "budget_exhausted")

        path = self.root / "config.affine-private.json"
        path.write_text(json.dumps(asdict(config), default=str))
        path.chmod(0o600)
        (config.state_dir / "started").unlink()
        parent = subprocess.Popen([sys.executable, "-c",
            "import sys;from reliquary.integrations.affine import AffineRunner,load_config;"
            "AffineRunner(load_config(sys.argv[1])).run()", str(path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        group = None
        try:
            limit = time.monotonic() + 5
            while time.monotonic() < limit:
                journal = json.loads((config.state_dir / "runner.json").read_text())
                if journal.get("owner_pid") == parent.pid and journal.get("process_group") \
                        and (config.state_dir / "started").exists():
                    group = journal["process_group"]
                    break
                time.sleep(0.02)
            self.assertIsNotNone(group, "fixture parent did not launch the native child")
            parent.kill()
            parent.wait(timeout=5)
            saved = (config.state_dir / "runner.json").read_bytes()
            with self.assertRaisesRegex(AffineRuntimeError, "explicit reconciliation"):
                AffineRunner(config).run()
            self.assertEqual((config.state_dir / "runner.json").read_bytes(), saved)
            with self.assertRaisesRegex(AffineRuntimeError, "still exists"):
                reconcile_state(config)
            os.killpg(group, signal.SIGKILL)
            limit = time.monotonic() + 5
            while time.monotonic() < limit:
                try:
                    reconciled = reconcile_state(config)
                    break
                except AffineRuntimeError:
                    time.sleep(0.02)
            else:
                self.fail("terminated fixture group remained unreconciled")
            self.assertEqual(reconciled["stage"], "reconciled")
            self.assertEqual(AffineRunner(config).run(max_seconds=0.1)["stage"], "budget_exhausted")
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if group is not None:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_logging_is_bounded_and_cleanup_failure_keeps_running_state(self):
        config = self.config("import sys,time;sys.stdout.write('x'*100000);sys.stdout.flush();time.sleep(30)\n")
        config = replace(config, max_log_bytes=1024)
        result = AffineRunner(config).run()
        self.assertEqual(result["stage"], "log_limit_exceeded")
        self.assertEqual((config.state_dir / "runtime.log").stat().st_size, 1024)
        self.assertEqual(AffineRunner(config).run()["stage"], "log_limit_exceeded")
        (config.state_dir / "runtime.log").unlink()
        processes = []

        def fail_cleanup(process):
            processes.append(process)
            raise PermissionError("unconfirmed")

        try:
            with mock.patch.object(affine, "_terminate", side_effect=fail_cleanup):
                with self.assertRaisesRegex(AffineRuntimeError, "cleanup is unconfirmed"):
                    AffineRunner(config).run()
            journal = json.loads((config.state_dir / "runner.json").read_text())
            self.assertEqual(journal["stage"], "running")
            with self.assertRaisesRegex(AffineRuntimeError, "explicit reconciliation"):
                AffineRunner(config).run()
        finally:
            for process in processes:
                affine._terminate(process)

    def test_isolated_epoch_guard_refuses_rotating_discovery_and_tampered_snapshot(self):
        import base64
        import hashlib
        from nacl.signing import SigningKey
        from reliquary.integrations.affine_epoch import canonical

        config = self.config("raise AssertionError('the discovery bootstrap must not run in stage two')\n")
        key = SigningKey.generate()
        authority = key.verify_key.encode().hex()
        source_hash = hashlib.sha256(b"fixture source archive").hexdigest()
        source = config.source_cache_dir / source_hash
        (source / "subnet").mkdir(parents=True)
        (source / "subnet/__init__.py").write_text("")
        (source / "subnet/client.py").write_text("def fetch_signed(*args):\n raise RuntimeError('discovery has rotated to successor')\n")
        (source / "subnet/cli.py").write_text("""
import json, pathlib, sys
from .client import fetch_signed
assert sys.flags.isolated and sys.dont_write_bytecode
assert 'approved_source_bootstrap' not in sys.modules
assert '--manifest-url' in sys.argv and '--current-url' not in sys.argv
authority = sys.argv[sys.argv.index('--authority') + 1]
manifest = fetch_signed(sys.argv[sys.argv.index('--manifest-url') + 1], authority)
try:
    fetch_signed('https://rotated.invalid/current', authority)
except ValueError:
    refused = True
else:
    raise AssertionError('rotating discovery must be refused')
state = pathlib.Path(sys.argv[sys.argv.index('--state') + 1])
state.mkdir(parents=True, exist_ok=True)
(state / 'executed.json').write_text(json.dumps({'epoch': manifest['epoch'],
    'checkpoint': manifest['checkpoint']['id'], 'rotated_refused': refused, 'model_execution': False}))
""")
        config.source_cache_dir.chmod(0o700)
        route = config.current_url.replace("/current?", "/initial-manifest?")
        manifest = dict(epoch="test-initial", deadline=time.time() + 30, transport_policy="direct-r2-v1",
                        checkpoint={"id": "1" * 64}, source_bundle={"sha256": source_hash})

        def envelope(payload):
            return dict(payload=payload, signer=authority,
                        signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())

        snapshot = dict(current_envelope=envelope(dict(epoch="test-initial", transport_policy="direct-r2-v1",
                                                       manifest_url=route)), manifest_envelope=envelope(manifest))
        path = self.root / "snapshot.affine-private.json"
        path.write_bytes(canonical(snapshot))
        path.chmod(0o600)
        config = replace(config, authority=authority, manifest_snapshot_file=path)
        runner = AffineRunner(config)
        command = runner._command() + ["--source-root", str(source)]
        self.assertNotIn(config.current_url, command)
        child = subprocess.run(command, capture_output=True, timeout=5)
        self.assertEqual(child.returncode, 0, child.stderr.decode())
        observed = json.loads((config.state_dir / "executed.json").read_text())
        self.assertEqual((observed["epoch"], observed["checkpoint"]), ("test-initial", "1" * 64))
        self.assertTrue(observed["rotated_refused"])
        self.assertFalse(observed["model_execution"])
        snapshot["manifest_envelope"]["payload"]["checkpoint"]["id"] = "2" * 64
        path.write_bytes(canonical(snapshot))
        # The original command refuses modified bytes before native imports.
        child = subprocess.run(command, capture_output=True, timeout=5)
        self.assertEqual(child.returncode, 1)
        self.assertNotIn(b"private-signature", child.stderr)
        # A freshly hashed file still cannot change the authority-signed payload.
        child = subprocess.run(runner._command() + ["--source-root", str(source)], capture_output=True, timeout=5)
        self.assertEqual(child.returncode, 1)


if __name__ == "__main__":
    unittest.main()
