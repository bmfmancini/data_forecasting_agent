"""Characterization tests for the pipeline orchestration contract."""

from __future__ import annotations

import ast
from pathlib import Path


def test_pipeline_progress_events_stay_in_expected_order() -> None:
    """The user-visible progress sequence is an explicit orchestration contract."""
    source = Path("data_forecaster/backend/services/pipeline_service.py").read_text()
    tree = ast.parse(source)

    class ProgressVisitor(ast.NodeVisitor):
        """Collect literal progress messages in source order."""

        def __init__(self) -> None:
            self.steps: list[str] = []

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"_progress", "progress"}
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                self.steps.append(node.args[1].value)
            self.generic_visit(node)

    def progress_steps(function_name: str) -> list[str]:
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        visitor = ProgressVisitor()
        visitor.visit(function)
        return visitor.steps

    assert progress_steps("_run_statistical_stages") == [
        "Validating data\u2026",
        "Data validation complete",
        "Running statistical analysis\u2026",
        "Statistical analysis complete",
    ]
    assert "Selecting forecasting model\u2026" in progress_steps("_select_model")
    assert progress_steps("_run_forecast_stages") == [
        "Running forecast\u2026",
        "Forecast complete",
    ]
    assert progress_steps("_run_statistical_review") == [
        "Statistical review\u2026",
        "Statistical review complete",
    ]
    assert progress_steps("_run_report_stage") == [
        "Generating report\u2026",
        "Report complete",
    ]
    assert progress_steps("_build_visualizations") == ["Generating visualizations\u2026"]
    assert progress_steps("run_pipeline")[-1] == "Analysis complete"
