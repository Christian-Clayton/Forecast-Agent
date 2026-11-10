from forecast_agent.agent.tracing import TraceRecorder


def test_trace_recorder_stores_metadata_without_prompt_contents() -> None:
    trace = TraceRecorder(trace_id="trace-1")

    trace.record(
        "intake",
        "completed",
        request_id="request-1",
        input_tokens=20,
        output_tokens=10,
        cost_usd=0.01,
    )
    exported = trace.export()

    assert exported["trace_id"] == "trace-1"
    assert exported["events"][0]["step"] == "intake"
    assert exported["events"][0]["request_id"] == "request-1"
    assert "content" not in exported["events"][0]
