from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import fcntl


@contextmanager
def file_lock(lock_path: str | Path):
    """An exclusive lock to avoid multi-instance runs on the same host."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
