"""Characterization tests for the pipeline orchestration contract."""

from __future__ import annotations

import ast
from pathlib import Path


def test_pipeline_progress_events_stay_in_expected_order() -> None:
    """The user-visible progress sequence is an explicit orchestration contract."""
    source = Path("data_forecaster/backend/services/pipeline_service.py").read_text()
    tree = ast.parse(source)
    progress_steps: list[str] = []

    class ProgressVisitor(ast.NodeVisitor):
        """Collect literal progress messages in source order."""

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "_progress"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                progress_steps.append(node.args[1].value)
            self.generic_visit(node)

    ProgressVisitor().visit(tree)

    assert progress_steps[:4] == [
        "Validating data\u2026",
        "Data validation complete",
        "Running statistical analysis\u2026",
        "Statistical analysis complete",
    ]
    assert "Selecting forecasting model\u2026" in progress_steps
    assert progress_steps[-1] == "Analysis complete"
