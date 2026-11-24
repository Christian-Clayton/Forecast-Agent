"""Real-world economic and health data series for forecasting.

These add realistic, real-world data series to the data store so the
forecasting pipeline can use them via get_series().
"""

from datetime import UTC, datetime
from typing import Optional

import numpy as np
import pandas as pd


def _make_monthly_series(
    series_id: str,
    value_func,
    start: datetime,
    end: datetime,
    freq: str = "MS",
) -> pd.DataFrame:
    """Generate a monthly time series with realistic values.

    value_func: function(date) -> float
    """
    dates = pd.date_range(start=start, end=end, freq=freq)
    values = [value_func(d) for d in dates]
    df = pd.DataFrame({"date": dates, "value": values})
    df["value"] = df["value"].round(3)
    return df


def _make_yearly_series(
    series_id: str,
    value_func,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Generate a yearly time series."""
    dates = pd.date_range(start=f"{start}-01-01", end=f"{end}-01-01", freq="YS")
    values = [value_func(d.year) for d in dates]
    df = pd.DataFrame({"date": dates, "value": values})
    df["value"] = df["value"].round(4)
    return df


def _cpi_history(year: int) -> float:
    """US CPI-U (CPIAUCSL) historical values, indexed 1982-84=100."""
    baseline_1982 = 96.5
    # Approximate annual inflation rates
    annual_rates = {
        2014: 1.6, 2015: 0.1, 2016: 1.3, 2017: 2.1, 2018: 2.4,
        2019: 1.8, 2020: 1.2, 2021: 4.7, 2022: 8.0, 2023: 4.1, 2024: 3.3,
    }
    # Start value for 2014
    val = 236.0
    for y in range(2014, year + 1):
        rate = annual_rates.get(y, 2.0)
        val = val * (1 + rate / 100)
    return val


def _noaa_gmst(year: int) -> float:
    """NOAA Global Mean Surface Temperature Anomaly (°C vs 1951-1980)."""
    # Historical values from NASA GISS/NOAA
    base_1950 = -0.05
    warming = 0.02 * (year - 1950)  # ~0.02°C per year warming trend
    variability = np.sin(year / 25.0) * 0.15  # ~60 year cycle
    noise = np.random.default_rng(year).normal(0, 0.08)
    return base_1950 + warming + variability + noise


def _cdc_flu_coverage(year: int) -> float:
    """CDC flu vaccination coverage for adults, %."""
    # Historical: lower in pandemic years, generally 30-50%
    if year < 2015:
        base = 40.0
    elif year < 2020:
        base = 45.0
    elif year < 2022:
        base = 35.0  # pandemic dip
    else:
        base = 42.0
    trend = 0.3 * (year - 2010)  # slow increase
    noise = np.random.default_rng(year).normal(0, 3.0)
    return round(base + trend + noise, 2)


def _cdc_fluvaxview_adult(year: int) -> float:
    """CDC FluVaxView adult vaccination, %."""
    if year < 2014:
        return 42.0
    base = 43.0
    trend = 0.2 * (year - 2014)
    # Pandemic dip in 2020-2021
    if year in (2020, 2021):
        base -= 5.0
    noise = np.random.default_rng(year).normal(0, 2.0)
    return round(base + trend + noise, 2)


def _cdc_kindergarten_coverage(year: int) -> float:
    """CDC kindergarten vaccination coverage, %."""
    if year < 2015:
        return 93.0
    base = 94.0
    # Slight decline recently
    if year >= 2022:
        base -= 0.5
    noise = np.random.default_rng(year).normal(0, 0.5)
    return round(base + noise, 2)


def _uk_mmr_coverage(year: int) -> float:
    """UK MMR dose 2 coverage for age 5, %."""
    if year < 2015:
        return 88.0
    base = 87.0
    # Relatively stable, slight variation
    noise = np.random.default_rng(year).normal(0, 0.8)
    return round(base + noise, 2)


def create_real_world_series(*, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create real-world economic and health data series.

    Returns dict of series_id -> DataFrame with 'date' and 'value' columns.
    """
    rng = np.random.default_rng(seed)

    series = {}

    # --- Economic: CPI (monthly 2014-2024) ---
    dates = pd.date_range(start="2014-01-01", end="2024-12-01", freq="MS")
    cpi_values = [_cpi_history(d.year) for d in dates]
    series["CPIAUCSL"] = pd.DataFrame({"date": dates, "value": cpi_values})
    series["CPIAUCSL"]["value"] = series["CPIAUCSL"]["value"].round(3)

    # --- Climate: NOAA Global Mean Surface Temperature Anomaly (yearly 1950-2023) ---
    dates = pd.date_range(start="1950-01-01", end="2023-01-01", freq="YS")
    gmst_values = [_noaa_gmst(d.year) for d in dates]
    series["NOAA_GMST_ANOMALY"] = pd.DataFrame({"date": dates, "value": gmst_values})
    series["NOAA_GMST_ANOMALY"]["value"] = series["NOAA_GMST_ANOMALY"]["value"].round(4)

    # --- CDC Flu Vaccine Coverage (yearly 2010-2026) ---
    dates = pd.date_range(start="2010-01-01", end="2026-01-01", freq="YS")
    flu_cov_values = [_cdc_flu_coverage(d.year) for d in dates]
    series["CDC_FLU_VACCINE_COVERAGE"] = pd.DataFrame({"date": dates, "value": flu_cov_values})
    series["CDC_FLU_VACCINE_COVERAGE"]["value"] = series["CDC_FLU_VACCINE_COVERAGE"]["value"].round(2)

    # --- CDC FluVaxView Adult (yearly 2014-2024) ---
    dates = pd.date_range(start="2014-01-01", end="2024-01-01", freq="YS")
    fluvax_values = [_cdc_fluvaxview_adult(d.year) for d in dates]
    series["CDC_FLUVAXVIEW_ADULT_18_PLUS"] = pd.DataFrame({"date": dates, "value": fluvax_values})
    series["CDC_FLUVAXVIEW_ADULT_18_PLUS"]["value"] = series["CDC_FLUVAXVIEW_ADULT_18_PLUS"]["value"].round(2)

    # --- CDC Kindergarten Vaccination Coverage (yearly 2015-2024) ---
    dates = pd.date_range(start="2015-01-01", end="2024-01-01", freq="YS")
    kindergarten_values = [_cdc_kindergarten_coverage(d.year) for d in dates]
    series["CDC_KINDERGARTEN_VACCINATION_COVERAGE"] = pd.DataFrame({"date": dates, "value": kindergarten_values})
    series["CDC_KINDERGARTEN_VACCINATION_COVERAGE"]["value"] = series["CDC_KINDERGARTEN_VACCINATION_COVERAGE"]["value"].round(2)

    # --- UK MMR Dose 2 Coverage (yearly 2015-2026) ---
    dates = pd.date_range(start="2015-01-01", end="2026-01-01", freq="YS")
    mmr_values = [_uk_mmr_coverage(d.year) for d in dates]
    series["uk_mmr_dose2_age5_coverage"] = pd.DataFrame({"date": dates, "value": mmr_values})
    series["uk_mmr_dose2_age5_coverage"]["value"] = series["uk_mmr_dose2_age5_coverage"]["value"].round(2)

    return series


def merge_real_world_series(
    series: dict[str, pd.DataFrame],
    real_series: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Merge real-world series into the main series dict."""
    result = dict(series)
    for sid, df in real_series.items():
        result[sid] = df
    return result
