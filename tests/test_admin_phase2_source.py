"""Source-level tests for Phase 2 route error handling."""

from __future__ import annotations

import ast
from pathlib import Path

_ADMIN_ROUTES = Path("data_forecaster/frontend/blueprints/admin/routes.py")
_MAIN_ROUTES = Path("data_forecaster/frontend/blueprints/main/routes.py")
_BACKEND_MAIN = Path("data_forecaster/backend/main.py")
_CHAT_SERVICE = Path("data_forecaster/backend/services/chat_service.py")
_FILE_SERVICE = Path("data_forecaster/backend/services/file_service.py")
_JOB_SERVICE = Path("data_forecaster/backend/services/job_service.py")
_PIPELINE_SERVICE = Path("data_forecaster/backend/services/pipeline_service.py")
_VISUALIZATION = Path("data_forecaster/backend/utils/visualization.py")


def _admin_tree() -> ast.Module:
    """Parse the admin routes module."""
    return ast.parse(_ADMIN_ROUTES.read_text())


def _broad_exception_handlers(tree: ast.Module) -> list[tuple[str, int]]:
    """Return ``(function_name, line)`` for broad ``except Exception`` handlers."""
    handlers: list[tuple[str, int]] = []
    function_stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        """Collect broad exception handlers with their enclosing function."""

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            caught = node.type
            if isinstance(caught, ast.Name) and caught.id == "Exception":
                handlers.append((function_stack[-1] if function_stack else "", node.lineno))
            self.generic_visit(node)

    Visitor().visit(tree)
    return handlers


def test_admin_routes_do_not_catch_bare_exception() -> None:
    """Admin route handlers should catch boundary-specific exceptions."""
    assert _broad_exception_handlers(_admin_tree()) == []


def test_remaining_broad_catches_are_documented_boundaries() -> None:
    """Remaining broad catches should stay limited to isolation boundaries."""
    main_tree = ast.parse(_MAIN_ROUTES.read_text())
    backend_tree = ast.parse(_BACKEND_MAIN.read_text())
    job_tree = ast.parse(_JOB_SERVICE.read_text())
    backend_handlers = _broad_exception_handlers(backend_tree)
    job_handlers = _broad_exception_handlers(job_tree)

    assert _broad_exception_handlers(main_tree) == []
    assert [handler[0] for handler in backend_handlers] == ["chat_explorer", "chat_explorer"]
    assert [handler[0] for handler in job_handlers] == ["_run_job"]


def test_services_do_not_add_undocumented_broad_exception_handlers() -> None:
    """Service modules should use specific exception categories by default."""
    for source_path in (_CHAT_SERVICE, _FILE_SERVICE, _PIPELINE_SERVICE, _VISUALIZATION):
        tree = ast.parse(source_path.read_text())
        assert _broad_exception_handlers(tree) == [], source_path


def test_job_service_defers_heavy_pipeline_and_rag_imports_at_module_load() -> None:
    """Scheduler imports should stay independent of the ML/RAG execution stack."""
    tree = ast.parse(_JOB_SERVICE.read_text())
    imported_names = {
        f"{node.module}.{alias.name}"
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if node.module
    }

    assert "services.pipeline_service.run_pipeline" not in imported_names
    assert "services.rag_service.get_rag_kb" not in imported_names


def test_admin_settings_save_frontend_config_after_backend_success() -> None:
    """Frontend settings must not commit before backend job settings succeed."""
    tree = _admin_tree()
    settings_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "settings"
    )

    update_line: int | None = None
    save_line: int | None = None
    for node in ast.walk(settings_fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update_job_settings"
        ):
            update_line = node.lineno
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_save_app_config_values"
        ):
            save_line = node.lineno

    assert update_line is not None
    assert save_line is not None
    assert update_line < save_line
