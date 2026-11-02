# Offline video-event benchmark

`python -m reliquary.vision.retail_benchmark` validates a frozen local video
manifest and scores externally supplied event intervals. It uses no model,
accelerator, provider, network, platform profile or native worker.

The manifest schema is `score-retail-benchmark/v1`, with `mode: offline_video`.
It contains `classes`, fixed `thresholds` (`score_threshold`, `tiou_threshold`)
and `cases`. Each case requires:

- `id`, `source_group`, `source_start_us`, `split` and `domain`;
- `duration_us`, local `media: {path, sha256}` and `annotation_complete`;
- `rights: {license, source_url, attribution, commercial_evaluation,
  commercial_training}`;
- `events: [{label, start_us, end_us}]`.

Intervals use clip-relative integer microseconds and a half-open end. A source
group identifies one physical camera/recording timeline; source start is the
clip's offset on that timeline. Overlapping excerpts, identical media and source
groups crossing splits are rejected. Related views or sessions can still leak
information despite different bytes: the dataset curator must also separate
those before freezing a benchmark. Camera-hours use the declared duration;
verify duration, clock, cuts and annotation alignment during preparation.

Rights records currently accept `CC-BY-4.0` or `owner-permission`, require
commercial evaluation permission, and retain the separate training flag.
They are declarations to inspect against the source licence or permission;
validation does not authenticate a grant or establish other rights. Incomplete
annotations may be prepared but cannot be scored. Missing annotations must not
be represented as negative examples.

The prediction envelope has `schema: score-retail-predictions/v1`, the exact
`benchmark_sha256`, `model: {id, revision}` and `predictions`. Each row has
`case_id`, `status`, `raw_output` and `events`. Status is `ok`, `unobservable` or
`invalid`; omitted rows are missing. An `ok` raw output must be exactly JSON
`{"events":[...]}` and match its imported events. Each predicted event requires
`label`, `start_us`, `end_us` and finite `score` in [0,1]. Non-ok rows have no
events. Retain invalid or truncated bytes without silently converting them into
a successful empty response.

```sh
python -m reliquary.vision.retail_benchmark validate "$MANIFEST" \
  --media-root "$MEDIA_ROOT"

python -m reliquary.vision.retail_benchmark score "$MANIFEST" "$PREDICTIONS" \
  --media-root "$MEDIA_ROOT" --split test --output "$PRIVATE_REPORT"
```

Use a new report path in an existing owner-private directory outside Git.
Outputs are exclusive-create and mode 0600. The manifest is parsed and hashed
from one byte snapshot. Model identity and inference execution remain external
declarations; the scorer does not authenticate a runtime receipt.

Scoring uses class-aware maximum-cardinality one-to-one matching at the frozen
temporal IoU threshold. Duplicate detections count as extra predictions. Reports
include per-class and aggregate TP/FP/FN, precision/recall/F1, matched mean tIoU,
processed coverage, reviewed normal exposure and false alerts per processed
camera-hour. Missing, invalid and unobservable cases keep their positives in the
recall denominator and contribute no processed exposure. Undefined metrics are
null. Normal exposure subtracts the union of all target intervals, so overlapping
labels do not remove the same time twice.

The zero-false-positive Poisson upper 95% rate is `-log(0.05)/hours`; no general
nonzero-count confidence interval is computed. A tiny zero-error clip is not
evidence of a low operational false-alert rate. The statistical assumptions and
camera/site dependence need review before customer acceptance.

This is an offline localization benchmark. It does not establish causal online
alert delay, delivery latency, inference cost, native retail training support or
station-domain quality. Those report fields remain null or false. The existing
vision worker's grounding and football contracts are unchanged.

For a lightweight local check without the unrelated validator server fixtures:

```sh
python -m pytest --noconftest -c /dev/null \
  tests/unit/test_vision_retail_benchmark.py -q -p no:cacheprovider
```
