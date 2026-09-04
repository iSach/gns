"""A heartbeat lock, so two trainers cannot share a run directory.

The obvious guard -- refuse if the checkpoint was written recently -- is wrong:
a cancelled job leaves a checkpoint that looks seconds old for as long as it
takes to reschedule, and the replacement refuses to start. That cost eleven runs
six hours once.

Instead the live trainer keeps ``RUNNING.json`` warm, touching it every time it
logs, and removes it on the way out, including on SIGTERM. A reader treats the
lock as free once the heartbeat goes stale, so a killed job releases it within
one heartbeat rather than one checkpoint interval.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import time
from pathlib import Path

LOCK_NAME = "RUNNING.json"
# Generous next to the logging interval, short next to a reschedule.
STALE_AFTER_SECONDS = 90.0


def lock_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / LOCK_NAME


def held(run_dir: str | Path, stale_after: float = STALE_AFTER_SECONDS) -> bool:
    """True if another trainer is alive in this run directory."""
    path = lock_path(run_dir)
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age < stale_after


def describe(run_dir: str | Path) -> dict:
    """What the lock says about its holder; empty if there is none."""
    try:
        return json.loads(lock_path(run_dir).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class RunLock:
    """Context manager that holds the lock for as long as training runs."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = lock_path(run_dir)
        self._previous_handlers: dict[int, object] = {}

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "job": os.environ.get("SLURM_JOB_ID", ""),
                    "started": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        # Slurm sends SIGTERM on scancel and at the time limit; release then, so
        # the requeued job does not have to wait for the heartbeat to go stale.
        for number in (signal.SIGTERM, signal.SIGINT):
            self._previous_handlers[number] = signal.getsignal(number)
            signal.signal(number, self._on_signal)
        return self

    def beat(self) -> None:
        """Mark the lock as still held. Cheap enough to call every log line."""
        try:
            os.utime(self.path, None)
        except FileNotFoundError:
            pass

    def release(self) -> None:
        self.path.unlink(missing_ok=True)

    def _on_signal(self, number, frame):
        self.release()
        previous = self._previous_handlers.get(number, signal.SIG_DFL)
        if callable(previous):
            previous(number, frame)
        else:
            signal.signal(number, signal.SIG_DFL)
            os.kill(os.getpid(), number)

    def __exit__(self, *exception) -> None:
        for number, previous in self._previous_handlers.items():
            signal.signal(number, previous)  # type: ignore[arg-type]
        self.release()
