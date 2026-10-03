"""Signed synthetic receipts exercise lifecycle and exact request bindings."""

import base64
import copy
import hashlib
import json
import unittest

from nacl.signing import SigningKey, VerifyKey

from reliquary.integrations.affine_evidence import canonical, verify_bundle


class EvidenceCheck(unittest.TestCase):
    def test_signed_lifecycle_and_bindings(self):
        key = SigningKey.generate()
        authority = key.verify_key.encode().hex()
        miner = SigningKey.generate().verify_key.encode().hex()
        sha = lambda value: hashlib.sha256(value).hexdigest()

        def envelope(payload):
            return dict(payload=payload, signer=authority,
                        signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())

        def signed(raw, known):
            value = json.loads(raw)
            self.assertEqual(value["signer"], known)
            VerifyKey(bytes.fromhex(known)).verify(canonical(value["payload"]),
                                                    base64.b64decode(value["signature"], validate=True))
            return value["payload"]

        old = {"model.safetensors": sha(b"old weights"), "tokenizer.json": sha(b"tokenizer")}
        new = dict(old, **{"model.safetensors": sha(b"new weights")})
        checkpoint = dict(id=sha(canonical(old)), files=old)
        successor = dict(id=sha(canonical(new)), files=new)
        source = dict(sha256=sha(b"source"), size=6, key="public/source.tar.gz")
        manifest = dict(epoch="nonpayable-fixture-0", checkpoint=checkpoint, source_bundle=source,
                        start=1, deadline=2, payable=False, transport_policy="direct-r2-v1", K=1, L=1,
                        environments=[dict(env_id="math", indices=[1, 2], spec={"version": "v1"}, harness={})])
        rolls = [dict(env_id="math", index=1, classification=kind,
                      turns=[dict(prompt=[1], output=[number])])
                 for number, kind in enumerate(("positive", "negative"), 2)]
        batch = dict(epoch=manifest["epoch"], checkpoint=checkpoint["id"], env_id="math",
                     index=1, sample_index=1, environment_version="v1", rollouts=rolls)
        frozen_sha = sha(b"synthetic zip")
        receipts = {miner: dict(sha256=frozen_sha, received_at=1.5)}
        scores = dict(epoch_id=manifest["epoch"], checkpoint=checkpoint["id"], finalized_at=3,
                      payable=False, receipts=receipts, total=0, weights={}, points={miner: 0})
        audit = dict(epoch=manifest["epoch"], submission_sha256=frozen_sha, accepted=[batch],
                     outcomes=[dict(env_id="math", index=1, valid=True, fully_audited=True)])
        training = dict(source_epoch=manifest["epoch"], input_checkpoint=checkpoint["id"],
                        checkpoint=successor["id"], steps=1, weights_changed=True,
                        updates=[dict(env_id="math", index=1,
                                      positive_rollout_sha256=sha(canonical(rolls[0])),
                                      negative_rollout_sha256=sha(canonical(rolls[1])))])
        payloads = dict(manifest=manifest, scores=scores, audit=audit,
                        audit_challenge=dict(generated_after_freeze_at=3, receipts=receipts, seed="fixture"),
                        training=training, checkpoint_descriptor=successor,
                        next_manifest=dict(manifest, epoch="nonpayable-fixture-1", start=4, checkpoint=successor))
        bundle = dict(schema="affine-evidence-bundle/v1", authority=authority,
                      epoch_id=manifest["epoch"], miner_id=miner, submission_sha256=frozen_sha,
                      source_sha256=source["sha256"], env_id="math", requested_indices=[1],
                      envelopes={name: envelope(payload) for name, payload in payloads.items()},
                      byte_checks=dict(source_sha256=source["sha256"], frozen_sha256=frozen_sha,
                                       checkpoint_files=new))
        result = verify_bundle(bundle, signed)
        self.assertEqual(result["stage"], "cycle_verified")
        self.assertEqual((result["accepted_batches"], result["points"], result["consumed_batches"]), (1, 0, 1))
        self.assertTrue(result["handover_verified"])
        self.assertTrue(result["qualified_successor"])
        self.assertEqual(result["next_bindings"]["checkpointDigest"], successor["id"])
        self.assertNotEqual(result["next_bindings"]["bootstrapDigest"], result["bindings"]["bootstrapDigest"])
        self.assertFalse(result["paid"] or result["inference_recomputed"] or result["model_execution_verified"])
        self.assertEqual(verify_bundle(bundle, signed, result["bindings"])["stage"], "cycle_verified")

        for removed, stage in (("next_manifest", "checkpoint_published"),
                               ("checkpoint_descriptor", "training_reported"),
                               ("training", "accepted")):
            partial = copy.deepcopy(bundle)
            for name in list(partial["envelopes"]):
                if list(payloads).index(name) >= list(payloads).index(removed):
                    partial["envelopes"].pop(name)
            self.assertEqual(verify_bundle(partial, signed)["stage"], stage)
        no_bytes = copy.deepcopy(bundle)
        no_bytes["byte_checks"] = {}
        self.assertEqual(verify_bundle(no_bytes, signed)["stage"], "handover_reported")
        self.assertFalse(verify_bundle(no_bytes, signed)["handover_verified"])
        self.assertFalse(verify_bundle(no_bytes, signed)["qualified_successor"])
        no_attribution = copy.deepcopy(bundle)
        no_attribution["envelopes"]["training"] = envelope(dict(training, updates=[]))
        self.assertFalse(verify_bundle(no_attribution, signed)["training_verified"])
        other_pair = dict(training["updates"][0], positive_rollout_sha256=sha(b"another miner pair"))
        unrelated = copy.deepcopy(bundle)
        unrelated["envelopes"]["training"] = envelope(dict(training, updates=[other_pair]))
        self.assertEqual(verify_bundle(unrelated, signed)["consumed_batches"], 0)
        self.assertFalse(verify_bundle(unrelated, signed)["handover_verified"])
        self.assertFalse(verify_bundle(unrelated, signed)["qualified_successor"])
        shared_index = copy.deepcopy(bundle)
        shared_index["envelopes"]["training"] = envelope(dict(training, updates=[other_pair, *training["updates"]]))
        self.assertEqual(verify_bundle(shared_index, signed)["stage"], "cycle_verified")
        self.assertEqual(verify_bundle(shared_index, signed)["consumed_batches"], 1)
        for changed in ("environment", "harness", "source", "runtime_profile"):
            different = copy.deepcopy(bundle)
            next_payload = copy.deepcopy(payloads["next_manifest"])
            if changed == "source":
                next_payload["source_bundle"]["sha256"] = sha(b"changed source")
            elif changed == "runtime_profile":
                next_payload["runtime_profile"] = {"device": "changed"}
            else:
                field = "spec" if changed == "environment" else "harness"
                next_payload["environments"][0][field] = {"version": "changed"}
            different["envelopes"]["next_manifest"] = envelope(next_payload)
            checked = verify_bundle(different, signed)
            with self.subTest(continuity=changed):
                self.assertEqual(checked["stage"], "cycle_verified")
                self.assertTrue(checked["handover_verified"])
                self.assertFalse(checked["qualified_successor"])
                self.assertNotIn("next_bindings", checked)
        missing = copy.deepcopy(bundle)
        missing["envelopes"]["scores"] = envelope(dict(scores, receipts={}))
        missing["envelopes"]["audit_challenge"] = envelope(dict(payloads["audit_challenge"], receipts={}))
        self.assertEqual(verify_bundle(missing, signed)["stage"], "not_submitted")

        changes = [
            ("scores", dict(scores, checkpoint=successor["id"])),
            ("audit", dict(audit, submission_sha256=sha(b"other submission"))),
            ("training", dict(training, input_checkpoint=successor["id"])),
            ("next_manifest", dict(payloads["next_manifest"], checkpoint=checkpoint)),
            ("audit_challenge", dict(payloads["audit_challenge"], generated_after_freeze_at=1)),
        ]
        for name, payload in changes:
            altered = copy.deepcopy(bundle)
            altered["envelopes"][name] = envelope(payload)
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_bundle(altered, signed)
        altered = copy.deepcopy(bundle)
        altered["envelopes"]["audit"]["payload"]["accepted"] = []
        with self.assertRaises(Exception):
            verify_bundle(altered, signed)
        with self.assertRaises(ValueError):
            verify_bundle(bundle, signed, dict(result["bindings"], bootstrapDigest=sha(b"another manifest")))
        heldout = copy.deepcopy(bundle)
        heldout["requested_indices"] = [3]
        with self.assertRaises(ValueError):
            verify_bundle(heldout, signed)


if __name__ == "__main__":
    unittest.main()
