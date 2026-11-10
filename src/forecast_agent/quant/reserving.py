import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReservingResult:
    factors: list[float]
    ultimates: list[float]
    ibnr: list[float]


def _normalize_triangle(
    triangle: Sequence[Sequence[float | None]],
) -> list[list[float | None]]:
    if len(triangle) < 2:
        raise ValueError("triangle must contain at least two origin periods")
    width = max(map(len, triangle))
    if width < 2 or any(len(row) > width for row in triangle):
        raise ValueError("triangle must contain at least two development periods")

    normalized: list[list[float | None]] = []
    for row in triangle:
        values = list(row) + [None] * (width - len(row))
        missing_seen = False
        for value in values:
            if value is None:
                missing_seen = True
            else:
                if missing_seen:
                    raise ValueError("observations must form a contiguous prefix")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("cumulative values must be finite and positive")
        if values[0] is None:
            raise ValueError("each origin period needs an observed value")
        normalized.append(values)

    if any(value is None for value in normalized[0]):
        raise ValueError("the first origin period must be fully developed")
    return normalized


def _development_factors(
    triangle: Sequence[Sequence[float | None]],
) -> tuple[list[list[float | None]], list[float]]:
    normalized = _normalize_triangle(triangle)
    width = len(normalized[0])
    factors = []
    for development_age in range(width - 1):
        pairs = [
            (row[development_age], row[development_age + 1])
            for row in normalized
            if row[development_age] is not None
            and row[development_age + 1] is not None
        ]
        if not pairs:
            raise ValueError(f"no observations for development age {development_age}")
        denominator = math.fsum(previous for previous, _ in pairs)
        numerator = math.fsum(current for _, current in pairs)
        factors.append(numerator / denominator)
    return normalized, factors


def chain_ladder(
    triangle: Sequence[Sequence[float | None]],
) -> ReservingResult:
    normalized, factors = _development_factors(triangle)
    ultimates = []
    ibnr = []
    for row in normalized:
        last_age = max(index for index, value in enumerate(row) if value is not None)
        latest = row[last_age]
        if latest is None:
            raise ValueError("each origin period needs an observed value")
        ultimate = latest * math.prod(factors[last_age:])
        ultimates.append(ultimate)
        ibnr.append(ultimate - latest)
    return ReservingResult(factors=factors, ultimates=ultimates, ibnr=ibnr)


def bornhuetter_ferguson(
    triangle: Sequence[Sequence[float | None]],
    premiums: Sequence[float],
    expected_loss_ratios: Sequence[float],
) -> ReservingResult:
    """Estimate IBNR from expected ultimate loss and unreported development."""
    normalized, factors = _development_factors(triangle)
    if len(premiums) != len(normalized) or len(expected_loss_ratios) != len(
        normalized
    ):
        raise ValueError("premiums and loss ratios must match origin periods")

    ultimates = []
    ibnr = []
    for row, premium, loss_ratio in zip(
        normalized, premiums, expected_loss_ratios, strict=True
    ):
        if (
            not math.isfinite(premium)
            or premium < 0
            or not math.isfinite(loss_ratio)
            or loss_ratio < 0
        ):
            raise ValueError("premiums and loss ratios must be finite and nonnegative")
        last_age = max(index for index, value in enumerate(row) if value is not None)
        latest = row[last_age]
        if latest is None:
            raise ValueError("each origin period needs an observed value")
        cumulative_factor = math.prod(factors[last_age:])
        expected_ultimate = premium * loss_ratio
        unreported_fraction = 1 - 1 / cumulative_factor
        expected_ibnr = expected_ultimate * unreported_fraction
        ibnr.append(expected_ibnr)
        ultimates.append(latest + expected_ibnr)
    return ReservingResult(factors=factors, ultimates=ultimates, ibnr=ibnr)
