from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_HELD_DEPTHS = threading.local()


def _process_lock(key: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def exclusive_file_lock(
    path: str | Path, *, timeout_seconds: float = 30.0
) -> Iterator[None]:
    """Hold a re-entrant one-byte lock shared by all compliant writers."""

    lock_path = Path(path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.path.normcase(str(lock_path))
    deadline = time.monotonic() + timeout_seconds
    local_lock = _process_lock(key)
    if not local_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise TimeoutError(f"timed out acquiring writer lock: {lock_path}")
    try:
        depths = getattr(_HELD_DEPTHS, "values", None)
        if depths is None:
            depths = {}
            _HELD_DEPTHS.values = depths
        if depths.get(key, 0):
            depths[key] += 1
            try:
                yield
            finally:
                depths[key] -= 1
            return

        handle = lock_path.open("a+b")
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        acquired = True
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"timed out acquiring writer lock: {lock_path}"
                            )
                        time.sleep(0.01)
            else:
                import fcntl

                while True:
                    try:
                        fcntl.flock(
                            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                        acquired = True
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"timed out acquiring writer lock: {lock_path}"
                            )
                        time.sleep(0.01)
            depths[key] = 1
            try:
                yield
            finally:
                del depths[key]
        finally:
            if acquired:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
    finally:
        local_lock.release()
