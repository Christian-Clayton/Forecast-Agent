from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4


@dataclass(frozen=True)
class TraceEvent:
    step: str
    status: Literal["started", "completed", "failed"]
    occurred_at: str
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class TraceRecorder:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid4().hex
        self.events: list[TraceEvent] = []

    def record(
        self,
        step: str,
        status: Literal["started", "completed", "failed"],
        *,
        request_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                step=step,
                status=status,
                occurred_at=datetime.now(UTC).isoformat(),
                request_id=request_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        )

    def export(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "events": [asdict(event) for event in self.events],
        }
