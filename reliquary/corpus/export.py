"""The dataset a corpus job delivers: every completion of every audited-and-passed
submission. The job's filter is applied here, as an annotation, because it
never decides payment."""

from __future__ import annotations


async def export_rows(*, job, records, grade=None):
    for submission_id in await records.list_verdict_ids(job.job_id):
        verdict = await records.read_verdict(job.job_id, submission_id)
        if not verdict or not verdict.get("passed"):
            continue
        record = await records.read_submission(job.job_id, submission_id)
        for completion in record["completions"]:
            row = {
                "prompt": record["rendered_prompt"],
                "completion": completion["text"],
                "prompt_index": record["prompt_index"],
                "hotkey": record["hotkey"],
                "submission_id": submission_id,
            }
            if grade is not None:
                accepted, score = grade(record["prompt_index"], completion["text"])
                row["accepted"], row["score"] = bool(accepted), float(score)
            yield row
