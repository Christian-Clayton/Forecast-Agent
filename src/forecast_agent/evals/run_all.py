"""Run all Phase 6 evaluation harnesses.

One command regenerates all results and commits them to evals/results/.
"""

import argparse
import json
from pathlib import Path

from forecast_agent.evals.backtest import BacktestRunner
from forecast_agent.evals.citation_validator_eval import (
    CitationValidatorEvaluator,
)
from forecast_agent.evals.classifier_eval import ClassifierEvaluator
from forecast_agent.evals.credibility_eval import CredibilityEvaluator
from forecast_agent.evals.leakage_eval import LeakageEvaluator
from forecast_agent.evals.regression_tracker import RegressionTracker
from forecast_agent.evals.report_calibration import CalibrationReportBuilder

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "evals" / "results"


def run_backtest() -> dict:
    """Run the walk-forward backtest harness."""
    runner = BacktestRunner(
        series_id="cell-00",
        start="2024-01-01",
        end="2024-12-31",
    )
    result = runner.run()
    return {
        "series_id": "cell-00",
        "mae": result.mae,
        "rmse": result.rmse,
        "baseline_mae": result.baseline_mae,
        "baseline_rmse": result.baseline_rmse,
        "candidate_selected": result.candidate_selected,
    }


def run_classifier_eval() -> dict:
    """Run the classifier evaluation harness."""
    evaluator = ClassifierEvaluator(series_id="cell-00")
    result = evaluator.run()
    return {
        "series_id": "cell-00",
        "predicted_domain": result.predicted_domain,
        "true_domain": result.true_domain,
        "accuracy": result.accuracy,
    }


def run_credibility_eval() -> dict:
    """Run the credibility evaluation harness."""
    groups = {
        "cell-00": [1, 2, 3, 4, 5],
        "cell-01": [2, 4, 6, 8],
        "cell-02": [0, 1],
    }
    evaluator = CredibilityEvaluator(groups=groups)
    result = evaluator.run()
    return {
        "groups": len(groups),
        "blended_mean": result.blended_mean,
        "raw_mean": result.raw_mean,
        "credibility_values": result.credibility,
        "k": result.k,
        "improved": result.improved,
    }


def run_leakage_eval() -> dict:
    """Run the leakage evaluation harness."""
    evaluator = LeakageEvaluator(store_id="synthetic")
    result = evaluator.run()
    return {
        "total_attempts": result.total_attempts,
        "blocked_attempts": result.blocked_attempts,
        "blocked_rate": result.blocked_rate,
    }


def run_citation_validator_eval() -> dict:
    """Run the citation validator evaluation harness."""
    evaluator = CitationValidatorEvaluator(ledger_path=":memory:")
    result = evaluator.run()
    return {
        "validated_claims": result.validated_claims,
        "uncited_claims": result.uncited_claims,
    }


def run_calibration() -> dict:
    """Build the calibration report."""
    builder = CalibrationReportBuilder()
    builder.add_forecast(brier=0.25, log_score=0.3, score_is_valid=True)
    builder.add_forecast(brier=0.15, log_score=0.2, score_is_valid=True)
    report = builder.build()
    return {
        "mean_brier": report.mean_brier,
        "mean_log_score": report.mean_log_score,
        "sharpness": report.sharpness,
    }


def run_regression_tracker() -> dict:
    """Run the regression tracker harness."""
    tracker = RegressionTracker(questions=["q1", "q2", "q3"])
    result = tracker.run()
    return {
        "diff": [
            {
                "metric": d.metric,
                "previous_value": d.previous_value,
                "current_value": d.current_value,
                "delta": d.delta,
                "passed": d.passed,
            }
            for d in result.diff
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6 evals")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results (default: evals/results/)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    results = {
        "backtest": run_backtest(),
        "classifier_eval": run_classifier_eval(),
        "credibility_eval": run_credibility_eval(),
        "leakage_eval": run_leakage_eval(),
        "citation_validator_eval": run_citation_validator_eval(),
        "calibration": run_calibration(),
        "regression_tracker": run_regression_tracker(),
    }

    output_file = args.output / "phase6_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results written to {output_file}")
    for name, result in results.items():
        print(f"  {name}: {json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()
