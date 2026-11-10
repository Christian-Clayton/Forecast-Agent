from dataclasses import asdict
from typing import Literal

import pandas as pd
from mcp.server.fastmcp import Context, FastMCP

from forecast_agent.quant import (
    baselines,
    credibility,
    ensemble_math,
    evt,
    frequency_severity,
    reserving,
    timeseries,
)


def _tool_result(ctx: Context, result: object) -> dict[str, object]:
    return {"tool_call_id": ctx.request_id, "result": result}


def create_quant_server() -> FastMCP:
    server = FastMCP(
        name="forecast-quant",
        instructions="Deterministic quantitative forecasting tools.",
        json_response=True,
    )

    @server.tool()
    def baseline_forecast(
        ctx: Context,
        series: list[float],
        horizon: int,
        method: Literal["naive", "seasonal_naive"],
        season_length: int | None = None,
    ) -> dict[str, object]:
        return _tool_result(
            ctx,
            asdict(
                baselines.baseline_forecast(
                    series,
                    horizon=horizon,
                    method=method,
                    season_length=season_length,
                )
            ),
        )

    @server.tool()
    def fit_ets(
        ctx: Context,
        series: list[float],
        horizon: int,
        seasonal_periods: int | None = None,
    ) -> dict[str, object]:
        return _tool_result(
            ctx,
            asdict(
                timeseries.fit_ets(
                    series,
                    horizon=horizon,
                    seasonal_periods=seasonal_periods,
                )
            ),
        )

    @server.tool()
    def fit_arima(
        ctx: Context,
        series: list[float],
        horizon: int,
        order: list[int] | None = None,
    ) -> dict[str, object]:
        selected_order = tuple(order) if order is not None else (1, 1, 1)
        if len(selected_order) != 3:
            raise ValueError("ARIMA order must contain three integers")
        return _tool_result(
            ctx,
            asdict(
                timeseries.fit_arima(
                    series,
                    horizon=horizon,
                    order=selected_order,
                )
            ),
        )

    @server.tool()
    def walk_forward_score(
        ctx: Context,
        series: list[float],
        model: Literal["naive", "seasonal_naive", "ets", "arima"],
        min_train_size: int,
        horizon: int,
        step: int = 1,
        season_length: int | None = None,
    ) -> dict[str, object]:
        return _tool_result(
            ctx,
            asdict(
                timeseries.walk_forward_score(
                    series,
                    model=model,
                    min_train_size=min_train_size,
                    horizon=horizon,
                    step=step,
                    season_length=season_length,
                )
            ),
        )

    @server.tool()
    def fit_frequency_severity(
        ctx: Context,
        event_table: list[dict[str, object]],
        exposure_column: str,
        frequency_column: str,
        severity_column: str,
        frequency_distribution: Literal["poisson", "negative_binomial"] = "poisson",
        severity_distribution: Literal["gamma", "lognormal"] = "lognormal",
    ) -> dict[str, object]:
        result = frequency_severity.fit_frequency_severity(
            pd.DataFrame(event_table),
            exposure_column=exposure_column,
            frequency_column=frequency_column,
            severity_column=severity_column,
            frequency_distribution=frequency_distribution,
            severity_distribution=severity_distribution,
        )
        return _tool_result(ctx, asdict(result))

    @server.tool()
    def aggregate_loss_mc(
        ctx: Context,
        frequency_parameters: dict[str, object],
        severity_parameters: dict[str, object],
        horizon: float,
        n_sims: int,
        seed: int,
    ) -> dict[str, object]:
        frequency = frequency_severity.FrequencyParameters(**frequency_parameters)
        severity = frequency_severity.SeverityParameters(**severity_parameters)
        return _tool_result(
            ctx,
            asdict(
                frequency_severity.aggregate_loss_mc(
                    frequency,
                    severity,
                    horizon=horizon,
                    n_sims=n_sims,
                    seed=seed,
                )
            ),
        )

    @server.tool()
    def buhlmann_credibility(
        ctx: Context,
        experience: dict[str, list[float]],
    ) -> dict[str, object]:
        return _tool_result(ctx, asdict(credibility.buhlmann_credibility(experience)))

    @server.tool()
    def chain_ladder(
        ctx: Context, triangle: list[list[float | None]]
    ) -> dict[str, object]:
        return _tool_result(ctx, asdict(reserving.chain_ladder(triangle)))

    @server.tool()
    def bornhuetter_ferguson(
        ctx: Context,
        triangle: list[list[float | None]],
        premiums: list[float],
        expected_loss_ratios: list[float],
    ) -> dict[str, object]:
        return _tool_result(
            ctx,
            asdict(
                reserving.bornhuetter_ferguson(
                    triangle,
                    premiums=premiums,
                    expected_loss_ratios=expected_loss_ratios,
                )
            ),
        )

    @server.tool()
    def evt_pot_fit(
        ctx: Context,
        series: list[float],
        threshold: float,
        return_periods: list[float],
    ) -> dict[str, object]:
        result = asdict(
            evt.evt_pot_fit(
                series,
                threshold=threshold,
                return_periods=return_periods,
            )
        )
        result["return_levels"] = [
            {"return_period": period, "level": level}
            for period, level in result["return_levels"].items()
        ]
        return _tool_result(ctx, result)

    @server.tool()
    def ensemble_aggregate(
        ctx: Context, probabilities: list[float]
    ) -> dict[str, object]:
        return _tool_result(
            ctx, asdict(ensemble_math.ensemble_aggregate(probabilities))
        )

    return server


def main() -> None:
    create_quant_server().run()


if __name__ == "__main__":
    main()
