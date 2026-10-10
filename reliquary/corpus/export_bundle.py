"""Publish an episode dataset last, after its complete counts sidecar."""

import json
import os
from pathlib import Path
import tempfile


def require_new_episode_output(out: str | Path) -> None:
    out = Path(out)
    if any(path.exists() or path.is_symlink() for path in (out, Path(f"{out}.counts.json"))):
        raise FileExistsError("episode exports require a new output and counts path")


def publish_episode_export(temporary: str | Path, out: str | Path, counts: dict) -> None:
    """Require fresh paths; the dataset marks completion of the two-file export.

    A process killed between publications can leave counts without a dataset.
    Use a fresh output path after interruption; existing results are never replaced.
    """
    temporary, out = Path(temporary), Path(out)
    sidecar = Path(f"{out}.counts.json")
    require_new_episode_output(out)
    if temporary.is_symlink():
        raise ValueError("the staged export must be a regular file")
    if temporary.parent.resolve() != out.parent.resolve():
        raise ValueError("the staged export must be beside its output")

    counts_temporary = None
    published_counts = False
    published_data = False
    directory = os.open(out.parent, os.O_RDONLY)
    try:
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent,
                                         prefix=f".{out.name}-counts-", delete=False) as file:
            counts_temporary = Path(file.name)
            json.dump(counts, file, sort_keys=True, indent=1)
            file.flush()
            os.fsync(file.fileno())
        # Hard links publish without replacing a result created by another export.
        os.link(counts_temporary, sidecar)
        published_counts = True
        os.fsync(directory)
        os.link(temporary, out)
        published_data = True
        os.fsync(directory)
    except BaseException:
        try:
            if published_data and out.samefile(temporary):
                out.unlink()
                os.fsync(directory)
            if published_counts and sidecar.samefile(counts_temporary):
                sidecar.unlink()
                os.fsync(directory)
        except OSError:
            # Keep counts if rollback itself fails; never remove counts before data.
            pass
        raise
    finally:
        os.close(directory)
        if counts_temporary is not None:
            try:
                counts_temporary.unlink(missing_ok=True)
            except OSError:
                pass
    try:
        temporary.unlink()
    except OSError:
        pass
