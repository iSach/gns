"""The heartbeat lock that keeps two trainers out of one run directory."""

import json
import os
import time

from gns.training.run_lock import RunLock, describe, held, lock_path


def test_lock_is_free_when_absent(tmp_path):
    assert not held(tmp_path)
    assert describe(tmp_path) == {}


def test_lock_is_held_while_the_trainer_lives(tmp_path):
    with RunLock(tmp_path) as lock:
        assert held(tmp_path)
        assert describe(tmp_path)["pid"] == os.getpid()
        lock.beat()
        assert held(tmp_path)
    assert not held(tmp_path)
    assert not lock_path(tmp_path).exists()


def test_a_stale_lock_reads_as_free(tmp_path):
    """A killed job must not block its own replacement.

    This is the bug the lock exists to fix: guarding on the checkpoint's mtime
    meant a cancelled job held the directory for as long as a reschedule took.
    """
    path = lock_path(tmp_path)
    path.write_text(json.dumps({"pid": 1, "host": "gone"}))
    old = time.time() - 600
    os.utime(path, (old, old))
    assert not held(tmp_path)


def test_release_is_idempotent(tmp_path):
    lock = RunLock(tmp_path)
    lock.__enter__()
    lock.release()
    lock.release()
    assert not held(tmp_path)
