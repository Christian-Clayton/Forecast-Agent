import math

import pandas as pd
import pytest

from forecast_agent.quant.frequency_severity import fit_frequency_severity


def test_frequency_severity_flattens_cell_level_claim_samples() -> None:
    event_table = pd.DataFrame(
        {
            "exposure": [1.0, 1.0, 1.0],
            "count": [1, 0, 2],
            "severity": [[math.exp(-1)], [], [1.0, math.exp(1)]],
        }
    )

    result = fit_frequency_severity(
        event_table,
        exposure_column="exposure",
        frequency_column="count",
        severity_column="severity",
        severity_distribution="lognormal",
    )

    assert result.frequency.rate == pytest.approx(1.0)
    assert result.severity.shape == pytest.approx(math.sqrt(2 / 3))
    assert result.severity.scale == pytest.approx(1.0)
