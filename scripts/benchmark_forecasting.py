"""Benchmark the complete numerical pipeline without an LLM or external data.

Run from the repository root; see docs/statistical-improvements.md.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "data_forecaster" / "backend")
)

from forecasting.backtesting import evaluate_candidates, evaluate_final_candidate
from forecasting.engine import BASELINES, ForecastEngine, backtest_config
from forecasting.fixtures import ALL_FIXTURES
from forecasting.selection_policy import CandidateEvidence, select_model_deterministic


def benchmark(name: str, length: int, horizon: int, origins: int) -> dict:
    started = time.monotonic()
    data = ALL_FIXTURES[name](n=length)
    engine = ForecastEngine("MS", {"backtesting": {"max_origins": origins}})
    config = backtest_config(len(data), horizon, "MS", engine.options)
    evaluations = evaluate_candidates(data, engine.candidates, config)
    evidence = [
        CandidateEvidence(name=key, backtest=value, is_baseline=key in BASELINES)
        for key, value in evaluations.items()
    ]
    excluded = []
    while True:
        selected = select_model_deterministic(
            evidence, user_loss_preference="mae", exclude_models=excluded
        ).selected_model
        if not selected:
            raise RuntimeError(f"No validated production candidate for {name}.")
        try:
            fitted = engine.candidates[selected].fit(data, horizon)
            break
        except (ValueError, RuntimeError):
            excluded.append(selected)
    final, _ = evaluate_final_candidate(
        selected, data, engine.candidates[selected], config
    )
    return {
        "fixture": name,
        "observations": len(data),
        "requested_horizon": horizon,
        "selected": selected,
        "loss": "mae",
        "validation_design": evaluations[selected].validation_design,
        "selection_metrics": evaluations[selected].pooled_metrics.model_dump(),
        "final_test_metrics": final.model_dump(),
        "baseline_skill": evaluations[selected].skill_scores,
        "interval_label": fitted.interval_label,
        "production_failures": excluded,
        "candidates": {
            key: {
                "rankable": value.is_rankable,
                "successful_origins": value.n_origins,
                "failed_origins": value.n_failed_origins,
                "mae": value.pooled_metrics.mae,
                "warnings": value.warnings,
            }
            for key, value in evaluations.items()
        },
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", default="additive_seasonal,random_walk,structural_break"
    )
    parser.add_argument("--length", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--origins", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/forecast-benchmark.json")
    )
    args = parser.parse_args()
    supported = {
        "additive_seasonal",
        "multiplicative_seasonal",
        "random_walk",
        "structural_break",
        "stationary_ar",
        "trend",
        "constant",
        "zeros",
    }
    fixtures = args.fixtures.split(",")
    if not set(fixtures) <= supported or args.length < 48:
        parser.error("Choose supported regular fixtures and at least 48 observations.")
    results = [
        benchmark(name, args.length, args.horizon, args.origins) for name in fixtures
    ]
    output = {
        "purpose": "Synthetic integration benchmark; not a claim about production accuracy.",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "statsmodels", "pmdarima")
        },
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for result in results:
        print(
            f"{result['fixture']}: {result['selected']}; selection MAE={result['selection_metrics']['mae']:.4f}; final MAE={result['final_test_metrics']['mae']}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
