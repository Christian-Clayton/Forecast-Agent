from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import Context, FastMCP

from forecast_agent.data.real_data import (
    create_real_world_series,
    merge_real_world_series,
)
from forecast_agent.data.store import DataStore
from forecast_agent.data.synth.generator import (
    SyntheticDataset,
    generate_synthetic_data,
)


def _serialize_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()
            elif isinstance(value, np.generic):
                record[key] = value.item()
            elif isinstance(value, list):
                record[key] = [
                    item.item() if isinstance(item, np.generic) else item
                    for item in value
                ]
    return records


def _tool_result(ctx: Context | None, result: object) -> dict[str, object]:
    try:
        tool_call_id = ctx.request_id if ctx is not None else None
        return {"tool_call_id": tool_call_id, "result": result}
    except Exception:
        return {"result": result}


def create_data_server(
    dataset: SyntheticDataset | DataStore | None = None,
) -> FastMCP:
    if isinstance(dataset, DataStore):
        store = dataset
    else:
        # Generate synthetic data + merge in real-world economic/health data series
        base_dataset = generate_synthetic_data() if dataset is None else dataset
        real_series = create_real_world_series()
        merged_series = merge_real_world_series(base_dataset.series, real_series)
        store = DataStore(
            SyntheticDataset(
                parameters=base_dataset.parameters,
                series=merged_series,
                event_tables=base_dataset.event_tables,
                truth=base_dataset.truth,
            )
        )
    server = FastMCP(
        name="forecast-data",
        instructions="Synthetic telecom data with strict as_of retrieval gates.",
        json_response=True,
    )

    @server.tool()
    def list_series(ctx: Context) -> dict[str, object]:
        return _tool_result(
            ctx,
            [
                {
                    "series_id": item.series_id,
                    "observations": item.observations,
                    "start": item.start.isoformat(),
                    "end": item.end.isoformat(),
                }
                for item in store.list_series()
            ],
        )

    @server.tool()
    def get_series(
        ctx: Context,
        series_id: str,
        start: datetime,
        end: datetime,
        as_of: datetime,
    ) -> dict[str, object]:
        return _tool_result(ctx, _serialize_records(store.get_series(series_id, start, end, as_of)))

    @server.tool()
    def get_event_table(
        ctx: Context,
        series_id: str,
        start: datetime,
        end: datetime,
        as_of: datetime,
    ) -> dict[str, object]:
        return _tool_result(ctx, _serialize_records(store.get_event_table(series_id, start, end, as_of)))

    @server.tool()
    def get_reference_class(
        ctx: Context,
        description: str,
        as_of: datetime,
    ) -> dict[str, object]:
        reference = store.get_reference_class(description, as_of)
        return _tool_result(
            ctx,
            {
                "description": reference.description,
                "base_rate": reference.base_rate,
                "source": reference.source,
                "as_of": reference.as_of.isoformat(),
            },
        )

    @server.tool()
    def get_stock_prices(
        ctx: Context,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, object]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            if df.empty:
                return _tool_result(ctx, {"symbol": symbol, "prices": [], "error": "No data found"})
            records = []
            for idx, row in df.iterrows():
                records.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                })
            return _tool_result(ctx, {"symbol": symbol, "prices": records, "count": len(records)})
        except Exception as e:
            return _tool_result(ctx, {"symbol": symbol, "prices": [], "error": str(e)})

    return server


def main() -> None:
    create_data_server().run()


if __name__ == "__main__":
    main()
