"""Generic async job lifecycle with structured provider events.

A job exposes two deliberately separate channels:

* ``logs`` are human-readable diagnostics and are never used as the business
  result protocol;
* ``events`` are structured provider messages used by SSE/API consumers.

The job also maintains a monotonically increasing ``revision`` for status,
phase, and structured-event changes. Diagnostic logs remain available in job
snapshots but do not create a network push for every printed line.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable


Event = dict[str, Any]
RunFn = Callable[[Callable[[str], None], Callable[[], bool], Callable[[Event], None]], Any]
PhaseOf = Callable[[str], str | None]


_TERMINAL_STATUSES = {"success", "error"}


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
        self.events: list[Event] = []
        self.event_sequence = 0
        self.revision = 0
        self.started_at = now_ms()
        self.finished_at: int | None = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self._cancelled = False
        self._terminal = False
        # New providers explicitly announce the end of their lifecycle with
        # a ``complete`` event/phase.  Keeping this bit separate from
        # ``phase`` prevents an adjudicated result from being mistaken for a
        # released/finished job while it is still holding the live video.
        self._complete_requested = False
        self._run_fn = run_fn
        self._phase_of = phase_of or (lambda line: None)
        self.thread = threading.Thread(target=self._run, name=f"{name}-{self.id[:8]}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def add_log(self, line: str) -> None:
        line = str(line).rstrip()
        if not line:
            return
        print(f"[job:{self.id[:8]}] {line}", flush=True)
        with self.condition:
            self.logs.append(line[-500:])
            self.logs = self.logs[-40:]
            phase = self._phase_of(line)
            if phase and not self._terminal and phase != self.phase:
                # Legacy adapters can still move the public phase forward, but
                # arbitrary diagnostic lines do not become a push protocol.
                self.phase = phase
                self.revision += 1
                self.condition.notify_all()

    def _append_event_locked(self, event: Event) -> None:
        """Append an event while ``self.condition`` is already held."""
        event = dict(event)
        event.setdefault("timestamp_ms", now_ms())
        self.event_sequence += 1
        event.setdefault("sequence", self.event_sequence)
        self.events.append(event)
        self.events = self.events[-40:]
        phase = event.get("phase")
        if isinstance(phase, str) and phase and not self._terminal:
            self.phase = phase
        if (
            event.get("event") == "complete"
            or event.get("phase") == "complete"
        ) and not self._terminal:
            self._complete_requested = True
        self.revision += 1

    def add_event(self, event: Event) -> None:
        if not isinstance(event, dict):
            return
        with self.condition:
            if self._terminal:
                return
            self._append_event_locked(event)
            self.condition.notify_all()

    def is_cancelled(self) -> bool:
        with self.lock:
            return self._cancelled

    def _run(self) -> None:
        with self.condition:
            if self._terminal:
                return
            self.status = "running"
            self.phase = "starting"
            self.revision += 1
            self.condition.notify_all()
        try:
            result = self._run_fn(self.add_log, self.is_cancelled, self.add_event)
            self._succeed(result)
        except Exception as exc:  # keep the HTTP server alive on runtime errors
            self._fail(str(exc))

    def _succeed(self, result: Any) -> None:
        with self.condition:
            # Cancellation or an earlier failure wins the race with a worker
            # that returns a little later after its subprocess is terminated.
            if self._terminal:
                return
            if isinstance(result, dict) and result.get("verified"):
                has_result_event = any(event.get("event") == "result" for event in self.events)
                if not has_result_event:
                    # Legacy providers may still return a verified result after
                    # parsing their old CLI envelope. Promote it to the new
                    # structured job protocol so clients never need to parse
                    # stdout logs.
                    self._append_event_locked({"event": "result", **result})
            if isinstance(result, dict) and result.get("diagnosed") and result.get("retry_required"):
                diagnosis = result.get("diagnosis")
                message = diagnosis.get("message") if isinstance(diagnosis, dict) else None
                self.result = result
                self.error = str(message or "视觉裁决未完成，请重新开始")
                self.status = "error"
                self.phase = "error"
                self.finished_at = now_ms()
                self._terminal = True
                self.logs.append(self.error[-500:])
                self.logs = self.logs[-40:]
                self.revision += 1
                self.condition.notify_all()
                return
            adjudicated = bool(isinstance(result, dict) and result.get("adjudicated"))
            # ``adjudicated`` is a business result, not a lifecycle signal.
            # A new provider may return it before the holding phase has
            # elapsed; remain running until it emits ``complete``.  Legacy
            # verified providers have no holding contract and retain their
            # historical implicit success behavior.
            if adjudicated and not self._complete_requested:
                self.status = "running"
                self.phase = "holding"
                self.finished_at = None
            else:
                self.result = result
                self.status = "success"
                self.phase = "complete"
                self.finished_at = now_ms()
                self._terminal = True
            self.revision += 1
            self.condition.notify_all()

    def _fail_locked(self, message: str) -> None:
        if self._terminal:
            return
        self.error = str(message)
        self.status = "error"
        self.phase = "error"
        self.finished_at = now_ms()
        self._terminal = True
        self.logs.append(self.error[-500:])
        self.logs = self.logs[-40:]
        self.revision += 1
        self.condition.notify_all()

    def _fail(self, message: str) -> None:
        with self.condition:
            self._fail_locked(message)

    def cancel(self) -> None:
        with self.condition:
            if self._terminal:
                return
            self._cancelled = True
            # Cancellation and the terminal transition must be atomic; a
            # worker returning concurrently cannot win the race and overwrite
            # the cancelled state with success.
            self._fail_locked("Job cancelled by user")

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "phase": self.phase,
            "error": self.error,
            "cancelled": self._cancelled,
            "result": self.result,
            "logs": list(self.logs),
            "events": list(self.events),
            "event_sequence": self.event_sequence,
            "revision": self.revision,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._snapshot_locked()

    def wait_for_update(self, after_revision: int, timeout: float) -> dict[str, Any]:
        """Wait for a status, phase, or structured-event change."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.revision > after_revision or self.status in _TERMINAL_STATUSES,
                timeout=max(0.0, timeout),
            )
            return self._snapshot_locked()
