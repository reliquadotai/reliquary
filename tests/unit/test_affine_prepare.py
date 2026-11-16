"""Signed metadata preparation, with no model, source installation or upload."""
import base64
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from nacl.public import SealedBox
from nacl.signing import SigningKey

from reliquary.integrations.affine import AffineConfig, AffineRunner, load_config
from reliquary.integrations.affine_cli import main
from reliquary.integrations.affine_evidence import canonical, inspect_bootstrap


class PreparedNativeTask(unittest.TestCase):
    def test_delegated_preparation_is_private_bound_and_disabled_until_qualification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkout = root / "upstream"
            (checkout / "subnet").mkdir(parents=True)
            (checkout / "subnet/__init__.py").write_text("")
            (checkout / "subnet/artifact_budget.py").write_text(
                "def for_manifest(manifest):\n return dict(compressed_bytes=2000000000,raw_bytes=3000000000,tensor_rows=2048)\n")
            (checkout / "subnet/storage.py").write_text("""import json,base64
from nacl.public import SealedBox
from nacl.signing import SigningKey
class Identity:
 def __init__(self,seed):
  self.key=SigningKey(seed);self.id=self.key.verify_key.encode().hex()
 def decrypt(self,value):
  return json.loads(SealedBox(self.key.to_curve25519_private_key()).decrypt(base64.b64decode(value)))
""")
            authority_key, miner_key = SigningKey.generate(), SigningKey.generate()
            authority, miner = authority_key.verify_key.encode().hex(), miner_key.verify_key.encode().hex()
            url = lambda name: "https://fixture.r2.cloudflarestorage.com/" + name + "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=fixture-secret"
            deadline = int(time.time()) + 300
            manifest = dict(epoch="fixture-epoch", deadline=deadline, transport_policy="direct-r2-v1", K=1, L=1,
                            max_batches=3, checkpoint={"id": "1" * 64}, source_bundle={"sha256": "2" * 64},
                            sampling_contract=dict(version="forced-inverse-cdf-replay-v1",
                                                   generation="uncached-eager-inverse-cdf", verification="exact-token-replay",
                                                   max_attempts=128, randomness="3" * 64), sampling_source_hash="4" * 64,
                            capabilities={miner: "encrypted-fixture"},
                            environments=[dict(env_id="math", indices=[2, 7, 8], spec={"version": "v1"},
                                               harness={"max_output_tokens": 1024})])
            capability = dict(epoch=manifest["epoch"], identity=miner, deadline=deadline,
                              transport="direct-r2-v1", headers={"Content-Type": "application/octet-stream"},
                              put_url=url("private/fixture-epoch/staging/" + miner + ".zip"))
            upload_fields = {name: capability[name] for name in ("deadline", "transport", "headers", "put_url")}
            sealed = SealedBox(miner_key.verify_key.to_curve25519_public_key())
            manifest["capabilities"][miner] = base64.b64encode(sealed.encrypt(canonical(upload_fields))).decode()
            cap = root / "cap.affine-private.json"
            cap.write_bytes(canonical(capability))
            cap.chmod(0o600)

            def git(*args):
                return subprocess.check_output(["git", "-C", str(checkout), *args], stderr=subprocess.DEVNULL)

            git("init", "-q")

            def pin(value):
                def envelope(payload):
                    return dict(payload=payload, signer=authority,
                                signature=base64.b64encode(authority_key.sign(canonical(payload)).signature).decode())
                documents = {url("current"): canonical(envelope(dict(epoch=value["epoch"], transport_policy="direct-r2-v1",
                                                                   manifest_url=url("manifest")))).decode(),
                             url("manifest"): canonical(envelope(value)).decode()}
                # The fixture performs real Ed25519 verification in the isolated
                # metadata interpreter. No inference package exists in this source.
                source = """import json,base64
from nacl.signing import VerifyKey
JSON_LIMIT=33554432
COMPRESSED_LIMIT=33554432
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def r2_url(url):
 if not isinstance(url,str) or not url.startswith('https://fixture.r2.cloudflarestorage.com/'):raise ValueError('url')
 return url
def signed(raw,authority):
 value=json.loads(raw)
 assert value['signer']==authority
 VerifyKey(bytes.fromhex(authority)).verify(canonical(value['payload']),base64.b64decode(value['signature']))
 return value['payload']
def download(url,limit):return DOCUMENTS[url].encode()
""" + "\nDOCUMENTS=" + repr(documents) + "\n"
                (checkout / "subnet/source_bootstrap.py").write_text(source)
                git("add", "subnet")
                git("-c", "user.name=Preparation Check", "-c", "user.email=check@example.invalid", "commit", "-qm", "metadata fixture")
                return git("rev-parse", "HEAD").decode().strip()

            config = AffineConfig(upstream_checkout=checkout, upstream_revision=pin(manifest), python=Path(sys.executable),
                                  authority=authority, current_url=url("current"), state_dir=root / "state",
                                  source_cache_dir=root / "cache", cap_file=cap, env_id="math", indices=(2, 7),
                                  search_budget=128, max_batches=3, process_timeout_seconds=5)
            config_path = root / "input.affine-private.json"

            def invoke(command, *args):
                config_path.write_bytes(canonical({name: str(value) if isinstance(value, Path) else value
                                                   for name, value in asdict(config).items()}))
                config_path.chmod(0o600)
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = main([command, "--config", str(config_path), *map(str, args)])
                raw = stdout.getvalue()
                self.assertNotIn(miner, raw)
                self.assertNotIn("fixture-secret", raw)
                self.assertNotIn(str(root), raw)
                return code, json.loads(raw)

            code, checked = invoke("check")
            self.assertEqual(code, 0)
            self.assertTrue(checked["epoch_open"])
            self.assertEqual(checked["credential_mode"], "delegated_capability")
            self.assertEqual(checked["constraints"]["effective_max_batches"], 2)
            self.assertEqual(checked["constraints"]["signed_max_attempts"], 128)
            self.assertEqual(checked["constraints"]["artifact_budget"]["tensor_rows_per_array"], 2048)
            self.assertEqual(config.state_dir.stat().st_mode & 0o777, 0o700)
            key = root / "client-key"
            original_key = miner_key.encode().hex().encode()
            key.write_bytes(original_key)
            key.chmod(0o600)
            delegated = root / "delegated.affine-private.json"
            config = replace(config, key_file=key, cap_file=None)
            code, receipt = invoke("delegate", "--out", delegated)
            self.assertEqual(code, 0)
            self.assertEqual(receipt["stage"], "delegated")
            self.assertEqual(json.loads(delegated.read_bytes()), capability)
            self.assertEqual(delegated.stat().st_mode & 0o777, 0o600)
            self.assertEqual(key.read_bytes(), original_key)
            self.assertEqual(invoke("delegate", "--out", delegated)[0], 1)
            key.write_bytes(SigningKey.generate().encode().hex().encode())
            refused = root / "wrong-key.affine-private.json"
            self.assertEqual(invoke("delegate", "--out", refused)[0], 1)
            self.assertFalse(refused.exists())
            key.write_bytes(original_key)
            key.write_bytes(b"not-a-native-seed")
            self.assertEqual(invoke("delegate", "--out", refused)[0], 1)
            key.write_bytes(original_key)
            approved = config.authority
            config = replace(config, authority="e" * 64)
            self.assertEqual(invoke("delegate", "--out", refused)[0], 1)
            config = replace(config, authority=approved)
            encrypted = manifest["capabilities"][miner]
            damaged = bytearray(base64.b64decode(encrypted))
            damaged[-1] ^= 1
            manifest["capabilities"][miner] = base64.b64encode(damaged).decode()
            config = replace(config, upstream_revision=pin(manifest))
            self.assertEqual(invoke("delegate", "--out", refused)[0], 1)
            self.assertFalse(refused.exists())
            manifest["capabilities"][miner] = encrypted
            config = replace(config, upstream_revision=pin(manifest))
            config = replace(config, key_file=None, cap_file=delegated)
            self.assertEqual(invoke("delegate", "--out", refused)[0], 1)
            output = root / "prepared.affine-private.json"
            code, prepared = invoke("prepare", "--out", output)
            self.assertEqual(code, 0)
            self.assertFalse(prepared["runnable"] or prepared["hardware_qualified"] or prepared["paid"])
            loaded = load_config(output)
            self.assertEqual(loaded.max_batches, 2)
            command = AffineRunner(loaded)._command()
            self.assertNotIn(config.current_url, command)
            self.assertEqual(command[command.index("--cap-file") + 1], str(delegated))
            self.assertNotIn("--key", command)
            task = json.loads(output.with_name(output.stem + "-task.affine-private.json").read_bytes())
            self.assertEqual(task["miner_id"], miner)
            self.assertEqual(task["readiness"]["bindings"], checked["bindings"])
            for path in (output, loaded.manifest_snapshot_file, output.with_name(output.stem + "-task.affine-private.json")):
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            original = output.read_bytes()
            self.assertEqual(invoke("prepare", "--out", output)[0], 1)
            self.assertEqual(output.read_bytes(), original)

            journal = config.state_dir / "runner.json"
            journal.write_bytes(canonical(dict(schema="affine-runtime/v1", stage="running", process_group=999999)))
            journal.chmod(0o600)
            blocked = root / "blocked.affine-private.json"
            self.assertEqual(invoke("prepare", "--out", blocked)[0], 1)
            self.assertFalse(blocked.exists())
            self.assertEqual(json.loads(journal.read_bytes())["stage"], "running")
            journal.unlink()
            config = replace(config, cap_file=cap)

            config = replace(config, indices=(9,))
            self.assertEqual(invoke("check")[0], 1)
            config = replace(config, indices=(2, 7))

            for change in ({"deadline": deadline + 1}, {"transport": "other"}, {"headers": {}},
                           {"put_url": url("private/other/staging/" + miner + ".zip")}, {"identity": "f" * 64}):
                cap.write_bytes(canonical(dict(capability, **change)))
                self.assertEqual(invoke("check")[0], 1)
            cap.write_bytes(canonical(capability))
            manifest["deadline"] = int(time.time()) - 1
            capability["deadline"] = manifest["deadline"]
            cap.write_bytes(canonical(capability))
            config = replace(config, upstream_revision=pin(manifest))
            self.assertFalse(invoke("check")[1]["epoch_open"])
            with self.assertRaises(RuntimeError):
                inspect_bootstrap(config)
            closed = root / "closed.affine-private.json"
            code, prepared = invoke("prepare", "--out", closed)
            self.assertEqual(code, 0)
            self.assertFalse(prepared["epoch_open"] or prepared["runnable"])
            config = replace(config, key_file=key, cap_file=None)
            self.assertEqual(invoke("delegate", "--out", root / "expired.affine-private.json")[0], 1)


if __name__ == "__main__":
    unittest.main()
