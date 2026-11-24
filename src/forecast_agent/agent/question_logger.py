"""Per-question detailed logging for the forecasting agent.

Each call to ``ForecastAgent.run()`` is one "question".  ``QuestionLogger``
records, for that question, everything we can observe:

* the question text and its generated id
* every model call to the provider (provider, model, response, retry status,
  input/output tokens, cost, per-attempt timing, errors)
* every tool call (tool name, arguments, output, timing, errors)
* every pipeline stage start/completed/failed (with timing and model response)
* the final result (forecast, refusal reason, totals)

Events are written both to a JSON Lines file (one object per line, easy to read
back with ``pandas``/``json.loads``) and to the console so you can watch a run
live.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional


class QuestionLogger:
    """Captures a full, detailed trace of one question run."""

    def __init__(self, question_id: str, question_text: str, log_file: Optional[str] = None):
        self.question_id = question_id
        self.question_text = question_text
        self.log_file = (Path(log_file) if log_file else None) or self._default_log_file()
        self.events: list[dict[str, Any]] = []
        self.start_time = time.perf_counter()
        self._stage_start: Optional[dict[str, Any]] = None
        self._write_header()

    # -- file handling -------------------------------------------------------
    @staticmethod
    def _default_log_file() -> Path:
        out = Path("logs")
        out.mkdir(exist_ok=True)
        return out / f"question_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}.jsonl"

    def _write_header(self) -> None:
        header = {
            "question_id": self.question_id,
            "question": self.question_text,
            "started_at": datetime.now(UTC).isoformat(),
            "started_unix_ms": int(self.start_time * 1000),
        }
        self.events.append(header)
        if self.log_file is not None:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(header, default=str) + "\n")

    # -- logging helpers -----------------------------------------------------
    def _emit(self, event: dict[str, Any]) -> None:
        event["question_id"] = self.question_id
        event["elapsed_s"] = round(time.perf_counter() - self.start_time, 3)
        event.setdefault("ts", datetime.now(UTC).isoformat())
        self.events.append(event)
        if self.log_file is not None:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")

    def question(self, reply: Optional[str] = None) -> None:
        self._emit({"event": "question", "question": self.question_text, "reply": reply or ""})

    def model_call(
        self,
        *,
        provider: str,
        model: str,
        status: str,
        response: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        timing_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        self._emit({
            "event": "model_call",
            "provider": provider,
            "model": model,
            "status": status,
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "timing_ms": timing_ms,
            "error": error,
        })

    def tool_call(
        self,
        *,
        tool: str,
        args: dict,
        output: dict,
        timing_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        self._emit({
            "event": "tool_call",
            "tool": tool,
            "args": args,
            "output": output,
            "timing_ms": timing_ms,
            "error": error,
        })

    def stage(
        self,
        *,
        name: str,
        status: str,
        model_response: Optional[str] = None,
        timing_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        self._emit({
            "event": "stage",
            "stage": name,
            "status": status,
            "model_response": model_response,
            "timing_ms": timing_ms,
            "error": error,
        })

    def summary(
        self,
        *,
        forecast: Optional[dict],
        refusal_reason: Optional[str],
        token_usage: dict,
        latency_s: float,
    ) -> None:
        self._emit({
            "event": "summary",
            "forecast": forecast,
            "refusal_reason": refusal_reason,
            "token_usage": token_usage,
            "latency_s": latency_s,
        })

    def finish(self) -> dict:
        """Return the accumulated log as a dict plus the log file path."""
        return {"question_id": self.question_id, "events": self.events, "log_file": str(self.log_file)}
