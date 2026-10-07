"""The grading adapter preserves missing/error rows and audit confidence."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary.corpus.delivery import LocalDirectorySink
from reliquary.eval.grading import collect_job_records, grade_evaluation
from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.mapping import environment_version
from tests.unit.test_eval_grading import _line, _plain_scorer, _publish, open_grading


def test_uploaded_mapping_is_complete_but_generation_unverified(tmp_path):
    card, rows = _publish(tmp_path, "code", 2, 3)
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["dataset"] = {"id": card["set_id"], "sha256": card["prompts_sha256"]}
    value["environment"] = {"id": card["env"], "version": environment_version(card)}
    contract = ServiceContract.from_dict(value)
    provenance = {"model": value["checkpoint"]["repo"], "revision": value["checkpoint"]["revision"],
                  "checkpoint_sha256": value["checkpoint"]["sha256"],
                  "generation_contract_sha256": value["generation_contract_sha256"]}
    folder = tmp_path / "platform" / "evaluations" / "mapping"
    folder.mkdir(parents=True)
    lines = [_line(row["problem_id"], i, f"```python\n{answer}\n```")
             for row in rows for i, answer in enumerate((f"={row['source_index']}", "wrong"))]
    (folder / "completions-00000.jsonl").write_text("".join(lines))
    manifest = asyncio.run(grade_evaluation(
        eval_id="mapping", set_ids=[card["set_id"]],
        completion_keys=["evaluations/mapping/completions-00000.jsonl"],
        problems_per_set={card["set_id"]: 2}, samples_per_set={card["set_id"]: 2},
        platform=LocalDirectorySink(tmp_path / "platform"), subnet=LocalDirectorySink(tmp_path / "subnet"),
        provenance=provenance, service_contract=contract, open_environment=open_grading,
        scorer_for=_plain_scorer, require_sandbox=lambda _: None, work_dir=tmp_path / "work"))
    assert manifest["service_contract_sha256"] == contract.sha256
    mapping = json.loads((folder / "mapping-manifest.json").read_text())
    assert mapping["complete"] and mapping["category_counts"] == {"in-zone": 2}
    assert mapping["generation_verified"] is False and mapping["sampling_verified"] is False
    assert mapping["provenance"]["rl_group_comparable"] is False
    assert {f["name"] for f in manifest["files"]} >= {"mapping.jsonl", "mapping-manifest.json"}


@pytest.mark.parametrize("audited,expected", [(True, True), (False, False)])
def test_generation_confidence_requires_every_passing_record_audited(audited, expected):
    class Records:
        async def list_verdict_ids(self, _):
            return ["first", "second"]
        async def read_verdict(self, _, sid):
            return {"passed": True, "audited": True if sid == "first" else audited}
        async def read_submission(self, _, sid):
            return {"prompt_index": 0, "completions": [{"text": sid}]}
    job = SimpleNamespace(job_id="fixture")
    collected = asyncio.run(collect_job_records(job, Records(), include_generation_status=True))
    assert collected["generation_verified"] is expected
    assert "generation_verified" not in asyncio.run(collect_job_records(job, Records()))
