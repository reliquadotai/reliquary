# Independent Affine benchmark evaluation on SN81

`reliquary affine evaluation prepare` freezes an operator-reviewed evaluation
offer for the existing platform evaluation API. SN81 miners generate completions
for a public, pinned checkpoint; the evaluation control qualifies verification,
audits every accepted completion, drains the corpus job and grades its records.
This uses the existing evaluation task and payment lifecycle.

The compatibility label is `independent-subnet-evaluation`. Affine's native
cached sampler, draw policy, chat template, runtime and answer projection are
separate experiment identities. This path does not establish native numerical
or scoring parity. V9 gradient-dot computation is a separate operation described
in [AFFINE_SCORING.md](AFFINE_SCORING.md).

## Prepare a private offer

Use an existing frozen single-turn set built with `reliquary eval build-set`.
Installed deterministic Verifiers tasksets retain their task identities and
native graders through that existing command. Tools, image prompts and LLM
judges are outside this single-turn intake. See the
[evaluation runbook](runbooks/evaluation-on-our-validator.md).

The set must include its reviewed source artifact digests, license-body digests
and explicit overlap evidence in its frozen protocol. These records describe
reviewed evidence; they do not establish that an Affine task is held out.

All input and output paths must be absolute, owner-only and outside repositories.
The preparation JSON has exactly these fields:

| Field | Required value |
| --- | --- |
| `schema` | `affine-subnet-evaluation-intake/v1` |
| `setDirectory` | Directory containing frozen `set.json`, `prompts.jsonl`, `grading.jsonl` |
| `modelSnapshot` | Saved public model metadata with `id`, full 40-hex `sha`, `private: false`, `gated: false` |
| `modelConfig` | Saved model `config.json`; its architecture must be supported by SN81 |
| `nativeSuite` | Original manifest/source/suite and reviewed task-to-prompt mapping below |
| `request` | Bounded existing evaluation request parameters below |
| `review` | Explicit approvals listed below |

`nativeSuite` contains `manifestSha256`, 40-hex `sourceRevision`, `suite` and
`taskMapping`. The suite retains its original `name`, `env_id`, ordered `indices`,
`seed` and `repeats`. Each mapping row contains `nativeIndex`, `taskSha256`,
`problemId` and `promptSha256`. The problem identity must match the frozen set
row in the same order; `promptSha256` hashes the row's `messages` using
`canonical_sha256`. The native task hashes remain operator-provided provenance:
preparation does not authenticate an upstream signing authority or prove prompt
equivalence to the native task.

`request` contains `problems`, `samples`, `max_new_tokens`, `thinking` and
`sampling`. Sampling explicitly supplies `temperature`, `top_p`, `top_k`, `seed`.
Subnet generation requires a positive temperature; `top_k` is `-1` or `1–1000`.
The request may select a prefix of the frozen set and independently choose its
sample count; the original native suite remains unchanged in provenance.

`review` contains exactly `sourceApproved`, `rightsApproved`, `overlapReviewed`,
`publicModelApproved` and `promptsVisibleToMinersApproved`, each `true`.
The public model and miner visibility approvals matter: SN81 participants can
read an evaluation job's prompts. This is unsuitable for confidential prompts
or a checkpoint that cannot be distributed publicly.

```sh
reliquary affine evaluation prepare --config "$PRIVATE_INTAKE" --out "$PRIVATE_OFFER"
```

Preparation checks the frozen set bytes, current harness and grader identities,
model snapshot, architecture, bounds and ordered provenance mapping. It writes
a new private file without overwriting existing state; stdout contains only a
sanitized digest/status summary. It never contacts a provider, publishes a set,
qualifies hardware or creates a billable order.

The resulting offer includes an immutable `offerDigest`, frozen file/protocol
hashes and `quoteRequest` for `POST /api/v1/evaluations/quotes` with
`mode: "subnet"`. Its availability remains `withdrawn`; live public model access,
publication, GPU qualification, native task mapping and native parity remain
explicitly unverified. A synthetically prepared offer is preparation evidence
only.

## Check a local Affine checkpoint export

`reliquary affine evaluation check-assets` checks local checkpoint bytes against
an original signed Affine opening without fetching assets or running a model.
Use a private JSON file with exactly `schema: "affine-evaluation-assets/v1"`,
`preparedOffer`, `manifestEnvelope`, `authority`, `modelSnapshot` and
`modelDirectory`. The four paths must be absolute and owner-only outside Git;
`authority` is the independently trusted operator's Ed25519 public key.

The saved model snapshot must match the preparation's byte hash, public model
name and immutable revision. The opening uses Affine's existing
`{payload, signer, signature}` envelope. Its `manifestSha256` is the SHA-256 of
the native canonical payload, and its exact signed evaluation suite must match
the prepared provenance. The native checkpoint descriptor is
`{id, files: {filename: sha256}}`, where `id` hashes the native canonical file
map. Existing checkpoint read capabilities are never used or exported.

`modelDirectory` must be a dedicated export containing exactly the descriptor's
files, including config and safetensors. The checker streams their hashes and
requires config bytes to match the preparation's `modelConfigSha256`. It
rejects changed, missing, extra or symlinked files. It writes a new private
receipt binding the preparation, authenticated opening, checkpoint, model
revision and local inventory; stdout contains only a sanitized status summary.

```sh
reliquary affine evaluation check-assets --config "$PRIVATE_ASSET_CONFIG" --out "$PRIVATE_ASSET_RECEIPT"
```

This verifies an authenticated opening and local byte equivalence. It does not
prove that a remote public model repository contains those bytes, authenticate
task-to-prompt equivalence, qualify a GPU or establish native evaluation parity.
Those flags remain false, and the offer remains withdrawn. A real pilot still
needs the approved checkpoint publication, native cohort/prompt/grader review,
actual subnet execution and an independently checked delivered report.

## Activation and service delivery

The existing reviewed set publication command emits a withdrawn registration
template. Actual publication, platform catalog qualification, model access,
executor availability and GPU verification must be established before an offer
is enabled. A local metadata snapshot cannot prove current remote availability.

The platform's authoritative quote reserves existing account credits. An owned,
idempotent evaluation order then uses the established qualification, SN81 job,
audit, drain, grading and private result-download lifecycle. Prepared offers do
not reserve funds or promise a GPU price, runtime, quality result or Affine
acceptance. Compare results only under matching frozen experiment conditions.

```sh
python -m pytest --noconftest tests/unit/test_affine_evaluation.py -q
```
