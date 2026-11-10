from forecast_agent.mcp_servers.quant_server import create_quant_server
from tests.mcp_servers.client_helpers import result_payload, run_client


def test_quant_server_exposes_schema_validated_tools() -> None:
    server = create_quant_server()

    async def inspect_and_call(session):
        tools = await session.list_tools()
        names = {tool.name for tool in tools.tools}
        baseline_tool = next(
            tool for tool in tools.tools if tool.name == "baseline_forecast"
        )
        result = await session.call_tool(
            "baseline_forecast",
            {
                "series": [1.0, 3.0, 2.0, 4.0],
                "horizon": 1,
                "method": "naive",
            },
        )
        invalid = await session.call_tool(
            "baseline_forecast",
            {
                "series": [1.0, 3.0, 2.0, 4.0],
                "horizon": 1,
                "method": "unknown",
            },
        )
        return names, baseline_tool.inputSchema, result, invalid

    names, schema, result, invalid = run_client(server, inspect_and_call)
    payload = result_payload(result)

    assert {
        "baseline_forecast",
        "fit_ets",
        "fit_arima",
        "walk_forward_score",
        "fit_frequency_severity",
        "aggregate_loss_mc",
        "buhlmann_credibility",
        "chain_ladder",
        "bornhuetter_ferguson",
        "evt_pot_fit",
        "ensemble_aggregate",
    } <= names
    assert {"series", "horizon", "method"} <= set(schema["required"])
    assert result.structuredContent["tool_call_id"]
    assert payload["forecast"] == [4.0]
    assert invalid.isError


def test_quant_server_tools_are_callable_from_mcp_client() -> None:
    server = create_quant_server()
    series = [2.0, 3.0, 2.5, 4.0, 3.5, 4.5]
    triangle = [[100.0, 150.0, 180.0], [120.0, 170.0, None], [130.0, None, None]]
    calls = [
        ("fit_ets", {"series": series, "horizon": 1}),
        ("fit_arima", {"series": series, "horizon": 1}),
        (
            "walk_forward_score",
            {
                "series": series,
                "model": "naive",
                "min_train_size": 4,
                "horizon": 1,
            },
        ),
        (
            "fit_frequency_severity",
            {
                "event_table": [
                    {"exposure": 1, "count": 2, "severity": [10.0, 20.0]},
                    {"exposure": 1, "count": 1, "severity": [15.0]},
                ],
                "exposure_column": "exposure",
                "frequency_column": "count",
                "severity_column": "severity",
            },
        ),
        (
            "aggregate_loss_mc",
            {
                "frequency_parameters": {
                    "distribution": "poisson",
                    "rate": 1.0,
                    "dispersion": 0.0,
                },
                "severity_parameters": {
                    "distribution": "lognormal",
                    "shape": 0.5,
                    "scale": 2.0,
                },
                "horizon": 1.0,
                "n_sims": 20,
                "seed": 7,
            },
        ),
        ("buhlmann_credibility", {"experience": {"a": [1.0, 2.0], "b": [3.0, 4.0]}}),
        ("chain_ladder", {"triangle": triangle}),
        (
            "bornhuetter_ferguson",
            {
                "triangle": triangle,
                "premiums": [200.0, 200.0, 200.0],
                "expected_loss_ratios": [0.8, 0.8, 0.8],
            },
        ),
        (
            "evt_pot_fit",
            {
                "series": [float(value) for value in range(1, 21)],
                "threshold": 10.0,
                "return_periods": [5.0],
            },
        ),
        ("ensemble_aggregate", {"probabilities": [0.2, 0.5, 0.8]}),
    ]

    async def call_all_tools(session):
        results = []
        for name, arguments in calls:
            results.append((name, await session.call_tool(name, arguments)))
        return results

    results = run_client(server, call_all_tools)

    for name, result in results:
        assert not result.isError, f"{name}: {result.content[0].text}"
        assert result.structuredContent["tool_call_id"]
        assert "result" in result.structuredContent
