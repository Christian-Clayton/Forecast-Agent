from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import Context, FastMCP

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


def _tool_result(ctx: Context, result: object) -> dict[str, object]:
    return {"tool_call_id": ctx.request_id, "result": result}


def create_data_server(
    dataset: SyntheticDataset | DataStore | None = None,
) -> FastMCP:
    if isinstance(dataset, DataStore):
        store = dataset
    else:
        store = DataStore(dataset if dataset is not None else generate_synthetic_data())
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
        return _tool_result(
            ctx, _serialize_records(store.get_series(series_id, start, end, as_of))
        )

    @server.tool()
    def get_event_table(
        ctx: Context,
        series_id: str,
        start: datetime,
        end: datetime,
        as_of: datetime,
    ) -> dict[str, object]:
        return _tool_result(
            ctx,
            _serialize_records(store.get_event_table(series_id, start, end, as_of)),
        )

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

    return server


def main() -> None:
    create_data_server().run()


if __name__ == "__main__":
    main()
