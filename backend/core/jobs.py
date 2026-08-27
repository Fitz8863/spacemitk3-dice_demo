"""Generic async job lifecycle for running a component/game function.

A job owns a worker thread, log/phase bookkeeping, cancel plumbing, and the
``queued → running → success | error`` status. The actual work is injected as a
``run_fn(on_log, is_cancelled)`` callable so the same lifecycle serves any
component or game pipeline.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

from core.errors import JobCancelledError, JobTimeoutError

# run_fn receives on_log(line) for streaming logs and is_cancelled() to check
# for a cancel request, and returns the result on success.
RunFn = Callable[[Callable[[str], None], Callable[[], bool]], Any]
# phase_of maps a log line to an optional phase label (None keeps the current).
PhaseOf = Callable[[str], str | None]


def now_ms() -> int:
    return int(time.time() * 1000)


class ComponentJob:
    def __init__(
        self,
        run_fn: RunFn,
        phase_of: PhaseOf | None = None,
        name: str = "dice-job",
    ) -> None:
        self.id = uuid.uuid4().hex
        self.status = "queued"
        self.phase = "queued"
        self.error = ""
        self.result: Any = None
        self.logs: list[str] = []
        self.started_at = now_ms()
        self.finished_at: int | None = None
        self.lock = threading.Lock()
        self._cancelled = False
        self._run_fn = run_fn
        self._phase_of = phase_of or (lambda line: None)
        self.thread = threading.Thread(target=self._run, name=f"{name}-{self.id[:8]}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def add_log(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        print(f"[job:{self.id[:8]}] {line}", flush=True)
        with self.lock:
            self.logs.append(line[-500:])
            self.logs = self.logs[-40:]
            phase = self._phase_of(line)
            if phase:
                self.phase = phase

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _run(self) -> None:
        with self.lock:
            self.status = "running"
            self.phase = "starting"
        try:
            result = self._run_fn(self.add_log, self.is_cancelled)
            self._succeed(result)
        except Exception as exc:  # keep the HTTP server alive on runtime errors
            self._fail(str(exc))

    def _succeed(self, result: Any) -> None:
        with self.lock:
            self.result = result
            self.status = "success"
            self.phase = "complete"
            self.finished_at = now_ms()

    def _fail(self, message: str) -> None:
        with self.lock:
            self.error = message
            self.status = "error"
            self.phase = "error"
            self.finished_at = now_ms()
            self.logs.append(message)
            self.logs = self.logs[-40:]

    def cancel(self) -> None:
        self._cancelled = True
        self._fail("Job cancelled by user")

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.id,
                "status": self.status,
                "phase": self.phase,
                "error": self.error,
                "result": self.result,
                "logs": list(self.logs),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }
