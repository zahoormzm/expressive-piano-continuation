"""A shared process lock for local training, validation, and CUDA generation."""

import contextlib
import json
import os
import time
from pathlib import Path

from filelock import FileLock, Timeout

from music_project.io import write_json


class ResourceBusyError(RuntimeError):
    pass


def gpu_status(root=None):
    root = Path.cwd() if root is None else Path(root)
    directory = root / "outputs"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(directory / "gpu.lock", timeout=0):
            return {"busy": False, "activity": None}
    except Timeout:
        try:
            owner = json.loads((directory / "gpu-session.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            owner = {}
        return {"busy": True, "activity": owner.get("activity") or "another model job"}


@contextlib.contextmanager
def gpu_slot(activity, enabled=True, root=None):
    if not enabled:
        yield
        return
    root = Path.cwd() if root is None else Path(root)
    directory = root / "outputs"
    directory.mkdir(parents=True, exist_ok=True)
    # Health probes hold the same lock very briefly; allow one second so a
    # probe cannot make an otherwise free GPU appear unavailable to a job.
    lock = FileLock(directory / "gpu.lock", timeout=1)
    try:
        lock.acquire()
    except Timeout as exc:
        owner = gpu_status(root)
        raise ResourceBusyError(
            f"GPU is busy with {owner['activity']}; wait for that job to finish."
        ) from exc
    try:
        write_json(
            directory / "gpu-session.json",
            {
                "activity": activity,
                "pid": os.getpid(),
                "started_unix": time.time(),
            },
        )
        yield
    finally:
        try:
            write_json(directory / "gpu-session.json", {"activity": None, "pid": None})
        finally:
            lock.release()
