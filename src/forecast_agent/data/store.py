import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from forecast_agent.data.synth.generator import SyntheticDataset

REFERENCE_CLASS = "daily telecom incident count"


def _copy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for column in result.select_dtypes(include="object").columns:
        result[column] = [deepcopy(value) for value in result[column]]
    return result


@dataclass(frozen=True)
class SeriesSummary:
    series_id: str
    observations: int
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ReferenceClass:
    description: str
    base_rate: float
    source: str
    as_of: datetime


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


class DataStore:
    def __init__(self, dataset: SyntheticDataset) -> None:
        self._dataset = dataset
        self._series = {
            series_id: _copy_frame(frame) for series_id, frame in dataset.series.items()
        }
        self._event_tables = {
            series_id: _copy_frame(frame)
            for series_id, frame in dataset.event_tables.items()
        }

    def list_series(self) -> list[SeriesSummary]:
        return [
            SeriesSummary(
                series_id=series_id,
                observations=len(frame),
                start=frame["date"].iloc[0].to_pydatetime(),
                end=frame["date"].iloc[-1].to_pydatetime(),
            )
            for series_id, frame in self._series.items()
        ]

    def get_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
        as_of: datetime,
    ) -> pd.DataFrame:
        return self._get_frame(self._series, series_id, start, end, as_of)

    def get_event_table(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
        as_of: datetime,
    ) -> pd.DataFrame:
        return self._get_frame(self._event_tables, series_id, start, end, as_of)

    def get_reference_class(
        self,
        description: str,
        as_of: datetime,
    ) -> ReferenceClass:
        cutoff = _utc_datetime(as_of)
        if description.strip().casefold() != REFERENCE_CLASS:
            raise ValueError(f"unsupported synthetic reference class: {description}")

        eligible = [
            frame.loc[frame["date"] <= cutoff] for frame in self._series.values()
        ]
        total_exposure = math.fsum(float(frame["exposure"].sum()) for frame in eligible)
        if total_exposure == 0:
            raise ValueError("no reference-class observations exist as of this date")
        total_count = math.fsum(float(frame["count"].sum()) for frame in eligible)
        return ReferenceClass(
            description=REFERENCE_CLASS,
            base_rate=total_count / total_exposure,
            source=f"synthetic:{self._dataset.parameters.seed}",
            as_of=cutoff,
        )

    @staticmethod
    def _get_frame(
        frames: dict[str, pd.DataFrame],
        series_id: str,
        start: datetime,
        end: datetime,
        as_of: datetime,
    ) -> pd.DataFrame:
        start_utc = _utc_datetime(start)
        end_utc = _utc_datetime(end)
        cutoff = _utc_datetime(as_of)
        if start_utc > end_utc:
            raise ValueError("start must be on or before end")
        if end_utc > cutoff:
            raise ValueError("end must be on or before as_of")
        if series_id not in frames:
            raise KeyError(f"unknown series: {series_id}")

        frame = frames[series_id]
        mask = (
            (frame["date"] >= start_utc)
            & (frame["date"] <= end_utc)
            & (frame["date"] <= cutoff)
        )
        return _copy_frame(frame.loc[mask].reset_index(drop=True))
