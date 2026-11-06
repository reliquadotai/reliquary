"""Stream an owner-private checkpoint member without following aliases."""

import hashlib
import os
import stat

from .affine import _private_file
from .affine_scoring import require


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _member(path):
    _private_file(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.getuid()
            and not before.st_mode & 0o077,
            "ordinary owner-private checkpoint file required",
        )
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
        after = os.fstat(stream.fileno())
    require(
        _stamp(before) == _stamp(after) == _stamp(path.stat(follow_symlinks=False)),
        "local checkpoint changed while hashing",
    )
    return {"sha256": value.hexdigest(), "bytes": after.st_size}
