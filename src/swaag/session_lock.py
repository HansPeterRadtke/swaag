"""Cross-process single-writer ownership for active session turns."""
from __future__ import annotations

import fcntl
import threading
from contextlib import contextmanager
from functools import wraps


_ownership = threading.local()


@contextmanager
def session_execution_lock(history, session_id, *, blocking=True):
    path = history.active_run_path(session_id).with_name('execution.lock')
    key = str(path.resolve())
    held = getattr(_ownership, 'held', None)
    if held is None:
        held = _ownership.held = set()
    if key in held:
        yield True
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        acquired = False
        try:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
                acquired = True
                held.add(key)
            except BlockingIOError:
                pass
            yield acquired
        finally:
            if acquired:
                held.remove(key)
                fcntl.flock(stream, fcntl.LOCK_UN)


def exclusive_session_turn(method):
    @wraps(method)
    def wrapped(self, state, *args, **kwargs):
        with session_execution_lock(self.history, state.session_id):
            return method(self, state, *args, **kwargs)
    return wrapped
