from datetime import UTC, datetime, timedelta

import pytest

from forecast_agent.data.store import DataStore
from forecast_agent.data.synth.generator import generate_synthetic_data


def make_store() -> DataStore:
    dataset = generate_synthetic_data(
        seed=7,
        n_groups=4,
        days=80,
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
    )
    return DataStore(dataset)


def test_series_and_event_queries_refuse_end_after_as_of() -> None:
    store = make_store()
    start = datetime(2024, 1, 1, tzinfo=UTC)
    as_of = start + timedelta(days=20)
    end = as_of + timedelta(days=1)

    with pytest.raises(ValueError, match="as_of"):
        store.get_series("cell-00", start, end, as_of)
    with pytest.raises(ValueError, match="as_of"):
        store.get_event_table("cell-00", start, end, as_of)


def test_successful_data_queries_never_return_rows_after_cutoff() -> None:
    store = make_store()
    start = datetime(2024, 1, 1, tzinfo=UTC)
    as_of = start + timedelta(days=20)

    series = store.get_series("cell-00", start, as_of, as_of)
    events = store.get_event_table("cell-00", start, as_of, as_of)

    assert series["date"].max().to_pydatetime() <= as_of
    assert events["date"].max().to_pydatetime() <= as_of


def test_reference_class_rate_is_computed_only_through_as_of() -> None:
    store = make_store()
    as_of = datetime(2024, 1, 20, tzinfo=UTC)

    reference = store.get_reference_class("daily telecom incident count", as_of)

    assert reference.base_rate >= 0
    assert reference.as_of == as_of
    assert reference.source == "synthetic:7"


def test_store_rejects_naive_timestamps() -> None:
    store = make_store()
    naive = datetime(2024, 1, 1)

    with pytest.raises(ValueError, match="timezone-aware"):
        store.get_series("cell-00", naive, naive, naive)


def test_store_is_isolated_from_mutations_to_generated_event_cells() -> None:
    dataset = generate_synthetic_data(
        seed=7,
        n_groups=4,
        days=80,
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
    )
    store = DataStore(dataset)
    dataset.event_tables["cell-00"].at[0, "severity"].append(999.0)

    events = store.get_event_table(
        "cell-00",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
    )

    assert 999.0 not in events.at[0, "severity"]
