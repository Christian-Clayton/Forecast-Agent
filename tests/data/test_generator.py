import pandas as pd

from forecast_agent.data.synth.generator import generate_synthetic_data


def test_synthetic_dataset_is_reproducible_from_seed() -> None:
    first = generate_synthetic_data(seed=41, n_groups=4, days=80)
    second = generate_synthetic_data(seed=41, n_groups=4, days=80)

    assert first.parameters == second.parameters
    assert first.truth == second.truth
    for group_id in first.series:
        pd.testing.assert_frame_equal(first.series[group_id], second.series[group_id])
        pd.testing.assert_frame_equal(
            first.event_tables[group_id], second.event_tables[group_id]
        )


def test_synthetic_dataset_plants_different_histories_and_fat_groups() -> None:
    dataset = generate_synthetic_data(seed=42, n_groups=6, days=120)
    history_lengths = [len(frame) for frame in dataset.series.values()]

    assert len(set(history_lengths)) == len(history_lengths)
    assert sum(label == "fat" for label in dataset.truth["domain_labels"].values()) == 2
    assert dataset.truth["severity"]["tail_probability"] > 0
    assert dataset.truth["development_ultimates"]
