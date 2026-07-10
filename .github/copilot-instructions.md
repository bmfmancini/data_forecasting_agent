# GitHub Copilot Instructions — Time Series Data Forecaster Agent

## Priority Guidelines

When generating code for this repository:

1. **Version Compatibility**: Always detect and respect the exact versions of languages, frameworks, and libraries used in this project (see [Technology Versions](#technology-versions) below).
2. **External Standards**: Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) as the authoritative external standard for all Python code. Apply its rules actively — do not wait to be reminded (see [Google Python Style Guide Compliance](#google-python-style-guide-compliance)).
3. **Context Files**: Prioritize patterns and standards defined in the `.github/instructions/` directory — especially `forecasting.instructions.md`, `python.instructions.md`, and `sonarqube_mcp.instructions.md`.
4. **Codebase Patterns**: When context files and external standards don't provide specific guidance, scan the codebase for established patterns (see [Codebase Scanning Instructions](#codebase-scanning-instructions)).
5. **Architectural Consistency**: Maintain the **mixed** architectural style — a FastAPI backend (layered services + agents) and a Flask frontend (application-factory + blueprints) — and respect the established boundaries between `backend/`, `frontend/`, and `core/` modules.
6. **Code Quality**: Prioritize **maintainability, performance, security, accessibility, and testability** in all generated code.

> **Important**: When the codebase conflicts with the Google Python Style Guide, prefer the Google standard unless doing so would break existing functionality. Note the deviation in a comment so it can be addressed over time. The user has explicitly requested that external best practices be enforced because they are not always followed otherwise.

---

## Project Overview

A multi-agent time series forecasting system.
- **Backend**: FastAPI + LangChain agents + statistical forecasting models (ARIMA, SARIMA, Holt-Winters, EWMA)
- **Frontend**: Flask web application (replaced Streamlit in a prior refactor)
- **Deployment**: Docker Compose — backend on port `8000`, frontend on port `5000`

---

## Repository Layout

```
data_forecaster/
├── backend/          # FastAPI service
│   ├── agents/       # LangChain-based AI agents (validation, statistical, model-selection, forecasting, report)
│   ├── auth/         # API key DB, Argon2id helpers, FastAPI auth dependencies
│   ├── core/         # config.py, llm_factory.py, logging_config.py
│   ├── forecasting/  # arima_model.py, sarima_model.py, holt_winters.py, ewma_model.py
│   ├── prompts/      # LLM prompt strings per agent (ChatPromptTemplate)
│   ├── rag/          # ChromaDB knowledge base + txt/md docs
│   ├── scripts/      # reset_api_key.py admin CLI
│   ├── services/     # chat_service, file_service, job_service, pipeline_service, rag_service
│   ├── utils/        # data_cleaning, data_parser, preflight, schemas, statistical, token_tracking, visualization
│   ├── main.py       # FastAPI app + route definitions (thin handlers)
│   ├── schemas.py    # Pydantic request/response models
│   └── exceptions.py # Custom domain exceptions
├── frontend/         # Flask application
│   ├── blueprints/
│   │   ├── main/     # Core pages (chat, overview, quality, stats, model, forecast, report)
│   │   │             # and AJAX endpoints (upload, preflight, analyze, job-status, chat …)
│   │   ├── auth/     # /auth/login, /auth/logout, /auth/change-password
│   │   ├── admin/    # User management, API credential management (admin role only)
│   │   └── decorators.py  # password_change_required decorator
│   ├── db/           # SQLite helpers (db.py, crypto.py) and schema.sql
│   ├── services/     # BackendAPIClient (api_client.py), PDF export (pdf_service.py)
│   ├── static/       # CSS (main.css) and JS (app.js, charts.js, chat.js, polling.js)
│   ├── templates/    # Jinja2 templates
│   ├── app.py        # Application factory (create_app)
│   ├── config.py     # DevelopmentConfig / ProductionConfig / TestingConfig
│   ├── extensions.py # csrf, login_manager singletons
│   ├── models.py     # Flask-Login User model
│   ├── manage.py     # CLI commands (user-create, credentials-set, generate-key)
│   ├── mypy.ini      # strict mypy config (python_version = 3.12)
│   └── wsgi.py       # Gunicorn entry point (wsgi:application)
├── data/             # Sample CSVs
├── docker/           # Dockerfile.backend, Dockerfile.flask, Dockerfile.nginx, docker-compose*.yml
└── tests/            # pytest test suite
```

---

## Technology Versions

Before generating code, detect and respect these exact versions. Never use language features, APIs, or library features beyond the detected versions.

### Language Versions

| Component | Version | Source |
|---|---|---|
| Backend Python | **3.11** | `data_forecaster/docker/Dockerfile.backend` (`FROM python:3.11-slim`), `pyproject.toml` (`requires-python = ">=3.11"`, `python_version = "3.11"`) |
| Frontend Python (Docker) | **3.12** | `data_forecaster/docker/Dockerfile.flask` (`FROM python:3.12-slim`), `frontend/mypy.ini` (`python_version = 3.12`) |

- Use `from __future__ import annotations` at the top of every Python module (already present in all existing files). This postpones the evaluation of type annotations, allowing for forward references. The backend's Python 3.11 natively supports PEP 604 union syntax (`str | None`).
- Never use 3.12+-only syntax features (e.g. `type` statement, `@override` decorator) in backend code.

### Backend Framework & Library Versions

Versions are pinned in `data_forecaster/backend/uv.txt` (Docker) and `data_forecaster/backend/requirements.txt` (local). The pinned set is authoritative:

| Library | Pinned Version | Usage |
|---|---|---|
| fastapi | `0.111.0` | API framework |
| uvicorn[standard] | `0.29.0` | ASGI server |
| langchain | `1.3.9` | LLM orchestration |
| langchain-community | `0.4.2` | Community integrations |
| langchain-ollama | `1.1.0` | Ollama LLM provider |
| langchain-google-genai | `4.2.5` | Gemini LLM provider |
| chromadb | `0.5.3` | Vector database |
| sentence-transformers | `5.5.1` | Embedding model (`all-MiniLM-L6-v2`) |
| statsmodels | `0.14.2` | Statistical models (Holt-Winters, STL, ACF/PACF) |
| pandas | `2.2.2` | DataFrame operations |
| pyarrow | `15.0.2` | Parquet I/O for file storage |
| numpy | `1.26.4` | Numerical operations |
| scipy | `1.13.1` | Statistical tests |
| plotly | `6.8.0` | Chart generation |
| matplotlib | `3.10.9` | ACF/PACF PNG generation (`Agg` backend) |
| pmdarima | `2.0.4` | `auto_arima` for ARIMA/SARIMA |
| python-multipart | `0.0.31` | File upload parsing |
| openpyxl | `3.1.3` | XLSX parsing |
| python-dotenv | `1.2.2` | `.env` loading |
| argon2-cffi | `23.1.0` | API key hashing (Argon2id) |

**Compatibility shim**: `pmdarima` 2.0.x uses `sklearn`'s `force_all_finite` which was removed in scikit-learn 1.6. The codebase patches `sklearn.utils.validation.check_array` at import time in `arima_model.py` and `sarima_model.py` — preserve this shim when modifying those files.

### Frontend Framework & Library Versions

Versions are pinned in `data_forecaster/frontend/requirements.txt`:

| Library | Version Constraint | Usage |
|---|---|---|
| flask | `>=3.0.3` | Web framework |
| flask-login | `>=0.6.3` | Session-based auth |
| flask-wtf | `>=1.2.1` | CSRF protection + WTForms |
| flask-session | `>=0.8.0` | Server-side sessions (filesystem) |
| werkzeug | `>=3.0.3` | Password hashing (`generate_password_hash`, `check_password_hash`) |
| cryptography | `>=42.0.5` | Fernet encryption for stored API credentials |
| requests | `>=2.32.3` | HTTP client for backend API calls |
| gunicorn | `>=22.0.0` | WSGI server |
| fpdf2 | `>=2.8.1` | PDF report export |
| markdown | `>=3.6` | Markdown rendering |
| bleach | `>=6.1.0` | HTML sanitization |
| pandas | `>=2.2.0` | Demo data handling |

### Frontend JavaScript Libraries

The frontend uses vendored libraries in `frontend/static/libs/` (no npm/build step). JavaScript files in `static/js/` are plain ES5-compatible IIFE modules:
- `app.js` — file upload, column selection, LLM health check
- `charts.js` — Plotly chart rendering (`Charts.renderAll`, `Charts.renderPie`, `Charts.renderDynamic`)
- `chat.js` — chat interface (`/api/chat` polling, DOMPurify + marked rendering)
- `polling.js` — job status polling (`/api/jobs/status`, 1.5s interval)

All AJAX requests include the CSRF token read from `<meta name="csrf-token">`.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI 0.111, Uvicorn 0.29 |
| AI / LLM | LangChain 1.3, Google Gemini (`langchain-google-genai`) or Ollama (`langchain-ollama`) |
| Forecasting | pmdarima 2.0.4, statsmodels 0.14.2 |
| Vector DB | ChromaDB 0.5.3 + sentence-transformers (`all-MiniLM-L6-v2`) |
| Frontend | Flask 3, Flask-Login, Flask-WTF, Flask-Session |
| Frontend auth | Username/password with `werkzeug.security` hashed passwords |
| Frontend DB | SQLite via `db/db.py` helpers (no ORM) |
| Backend auth | API key with Argon2id hashes (`argon2-cffi`) |
| PDF export | fpdf2 |
| Deployment | Docker Compose, Gunicorn |
| Python version | 3.11 (backend), 3.12 (frontend Docker image) |
| Type checking | mypy strict (frontend `mypy.ini`), mypy with `ignore_missing_imports` (backend `pyproject.toml`) |

---

## Google Python Style Guide Compliance

All Python code in this project must follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). The rules below are the key provisions — apply them actively on every change, even when the surrounding code doesn't yet comply.

### Language Rules

- **Lint**: Run `pylint` on your code. Suppress warnings with line-level `# pylint: disable=symbolic-name` comments (not the deprecated `disable-msg` form). Add an explanation when the reason isn't clear from the symbolic name. Prefer `del unused_arg  # Unused.` over `_` or `unused_` prefixes for unused arguments.
- **Imports**: Use `import x` for packages/modules, `from x import y` for submodules, `from x import y as z` only to resolve conflicts or shorten long names, `import y as z` only for standard abbreviations (e.g. `import numpy as np`). No relative imports — always use the full package path. Named imports (`from x import y`) are allowed and encouraged where they improve readability — e.g. importing specific classes, functions, or constants from a module. Exceptions: avoid importing individual symbols from modules that re-export a very large surface area (prefer `import x` in those cases).
- **Exceptions**: Use built-in exception classes where sensible (e.g. `ValueError` for precondition violations). Custom exceptions must inherit from an existing exception class and end in `Error` (e.g. `ForecastingAgentError`). Never use `assert` for input validation in production code (it's fine in tests). Never use catch-all `except:` or bare `except Exception:` unless re-raising or creating an isolation point. Minimise code inside `try`/`except` blocks. Use `finally` for cleanup.
- **Mutable Global State**: Avoid mutable global state. Module-level constants are permitted and encouraged (`UPPER_SNAKE_CASE` for public, `_UPPER_SNAKE_CASE` for internal). If mutable global state is unavoidable, prefix with `_` and document the design reason in a comment.
- **Nested Functions/Classes**: Fine when closing over a local variable. Don't nest a function just to hide it — prefix with `_` at module level instead so it remains testable.
- **Comprehensions & Generator Expressions**: Allowed for simple cases. No multiple `for` clauses or filter expressions in a single comprehension — use a nested loop instead.
- **Default Iterators & Operators**: Use `for key in adict:` not `for key in adict.keys():`. Use `if obj in alist:`. Don't mutate a container while iterating.
- **Generators**: Use `Yields:` not `Returns:` in generator docstrings. Wrap generators managing expensive resources with a context manager (PEP 0533).
- **Lambda Functions**: One-liners only. If the body exceeds 60-80 chars, define a regular nested function. Prefer `operator` module functions (e.g. `operator.mul`) over lambdas for common operations.
- **Conditional Expressions**: Simple cases only, each portion on one line. Use a full `if` statement when things get complicated.
- **Default Argument Values**: Never use mutable objects (`[]`, `{}`, `set()`) as default values. Use `None` and create inside the function. Empty tuples `()` are OK (immutable).
- **Properties**: Use `@property` only when necessary (computation, access control, lazy evaluation). Don't use a property for trivial get/set of an internal attribute — make the attribute public instead.
- **True/False Evaluations**: Use implicit false (`if not users:` not `if len(users) == 0:`). Always use `if foo is None:` / `is not None` for `None` checks. Never compare a boolean to `False` with `==`. For sequences, use `if seq:` / `if not seq:`. For numpy arrays, use `.size` not implicit bool.
- **Decorators**: Use judiciously. Avoid `staticmethod` (write a module-level function instead). Limit `classmethod` to named constructors or class-specific routines. Decorators should not have external dependencies (they run at import time).
- **Threading**: Do not rely on atomicity of built-in types. Use `queue.Queue` for inter-thread communication. Use `threading` locks and condition variables.
- **Power Features**: Avoid metaclasses, bytecode access, dynamic inheritance, `__del__` customisation, import hacks, and excessive reflection. Standard library modules using these internally are fine.
- **`from __future__` imports**: Encouraged. This project already uses `from __future__ import annotations` in every module — preserve it.
- **Type Annotations**: Strongly encouraged on all public APIs. Annotate code prone to type errors or that is hard to understand. Use `X | None` (not `Optional[X]`). Prefer `collections.abc` abstract types (`Sequence`, `Mapping`) over concrete types (`list`, `dict`) in function signatures.

### Style Rules

- **Semicolons**: Never terminate lines with semicolons or put two statements on one line.
- **Line length**: Google specifies 80 characters. **This project uses 88 characters** (Black formatter default) — follow the project's 88-char limit. Exceptions: long import statements, URLs/pathnames in comments, pylint disable comments.
- **Parentheses**: Use sparingly. Don't use around conditions in `if`/`while`/`return` unless for line continuation or tuple indication.
- **Indentation**: 4 spaces, never tabs. Align wrapped elements vertically or use a hanging 4-space indent. No 2-space hanging indents.
- **Trailing commas**: Recommended in multi-line sequences when the closing bracket is on a separate line (guides Black/Pyink formatting).
- **Blank lines**: Two blank lines between top-level definitions (functions/classes). One blank line between methods. No blank line after a `def` line.
- **Whitespace**: No whitespace inside parentheses/brackets/braces. No whitespace before commas/colons. No whitespace before open paren of argument list/index. No trailing whitespace. Spaces around binary operators (`=`, `==`, `<`, `and`, etc.). No spaces around `=` in keyword arguments or default parameters without type annotations — **but do use spaces when a type annotation is present** (`def f(a: int = 0):` not `def f(a: int=0):`). Don't vertically align tokens on consecutive lines.
- **Shebang**: Only on the main executable file: `#!/usr/bin/env python3`.
- **Strings**: Use f-strings, `%`, or `.format()` — never concatenate with `+` for formatting (single `a + b` join is OK). Avoid `+=` for string accumulation in loops — use `''.join(list)` or `io.StringIO`. Be consistent with quote character within a file. Use `"""` for docstrings and multi-line strings.
- **Logging**: Pass pattern strings with `%`-placeholders as the first argument, not f-strings: `logger.info('Value is: %s', value)` not `logger.info(f'Value is: {value}')`. This lets logging implementations collect the unexpanded pattern as a queryable field.
- **Error messages**: Must precisely match the actual error condition, clearly identify interpolated pieces, and allow simple grepping. Use `f'Not a probability: {p=}'` style.
- **Files & sockets**: Close explicitly. Prefer `with` statements. Use `contextlib.closing()` for objects without `with` support.
- **TODO comments**: Format: `# TODO: bug-reference - Description.` Avoid TODOs referring to individuals.
- **Imports formatting**: One import per line (exception: `typing` and `collections.abc`). Group: future → stdlib → third-party → local. Sort lexicographically within each group. Optional blank line between groups.
- **Statements**: One statement per line. `if foo: bar(foo)` is OK only with no `else`.
- **Naming**: `module_name`, `package_name`, `ClassName`, `method_name`, `ExceptionName`, `function_name`, `GLOBAL_CONSTANT_NAME`, `global_var_name`, `instance_var_name`, `function_parameter_name`, `local_var_name`. Avoid single-character names (except `i`/`j`/`k` for counters, `e` for exceptions, `f` for file handles). Avoid dashes in module names. Avoid type suffixes (`id_to_name_dict`). Don't use `__double_leading_and_trailing_underscore__`.
- **Main**: Put executable logic in a `main()` function guarded by `if __name__ == '__main__':`.
- **Function length**: Prefer small, focused functions. If a function exceeds ~40 lines, consider breaking it up. Use `# pylint: disable=too-many-locals` only when splitting would harm structure.
- **Type annotations**: Use `X | None` not `Optional[X]`. Prefer `collections.abc.Sequence` over `list` in signatures. Specify generic type parameters (`Sequence[int]` not bare `Sequence`). Use `TypeVar` with descriptive names when constrained or externally visible. Type aliases should be CapWorded (`_Private` if module-local).

### Docstrings (Google Style)

- Always use triple double-quote `"""` format.
- Summary line (one physical line, ≤80 chars) terminated by a period/question/exclamation mark, followed by a blank line, then the body.
- **Modules**: Start with a docstring describing contents and usage. Include a typical usage example for complex modules.
- **Functions/Methods**: Docstring mandatory for public API, nontrivial, or non-obvious functions. Sections: `Args:` (each parameter by name with type if not annotated), `Returns:` (or `Yields:` for generators — describe semantics and type), `Raises:` (only exceptions relevant to the interface, not those from API violations).
- **Classes**: Docstring with one-line summary describing what an instance represents (not "Class that describes..."). Include an `Attributes:` section for public attributes.
- **Overridden methods**: No docstring needed if decorated with `@override` (from `typing_extensions`), unless behaviour materially changes.
- **Block/inline comments**: Start at least 2 spaces from code, `#` followed by at least one space. Comment tricky code before it starts. Never describe what the code does — assume the reader knows Python.
- **Punctuation/spelling/grammar**: Comments should be as readable as narrative text with proper capitalisation and punctuation.

---

## Codebase Scanning Instructions

When context files and the Google Python Style Guide don't provide specific guidance:

1. Identify similar files to the one being modified or created (e.g. other agents in `backend/agents/`, other routes in `frontend/blueprints/`).
2. Analyze patterns for:
   - Naming conventions (snake_case functions, PascalCase classes, UPPER_SNAKE_CASE constants)
   - Code organization (module-level docstring → imports → constants → functions/classes)
   - Error handling (try/except with `logger.warning`, fallback values, custom exceptions from `backend.exceptions`)
   - Logging approaches (`get_logger(__name__)` in backend, `logging.getLogger(__name__)` in frontend)
   - Documentation style (Google-style docstrings with `Args:`, `Returns:`, `Raises:`)
   - Testing patterns (pytest fixtures, `sys.path` manipulation for imports, `monkeypatch` for env vars)
3. Follow the most consistent patterns found in the codebase.
4. When conflicting patterns exist, prioritize patterns in newer files or files with higher test coverage.
5. Never introduce patterns not found in the existing codebase or the Google Python Style Guide.

---

## Coding Conventions

### General (all Python files)
- **Python 3.11** compatible syntax throughout (backend); 3.12 for frontend Docker image.
- Start every module with `from __future__ import annotations` to postpone evaluation of type annotations.
- **PEP 8** with 4-space indentation and max line length **88** characters.
- **Type hints** on all function parameters and return values — use `str | None`, `list[str]`, `dict[str, Any]` (not `Optional[str]`, `List[str]`).
- **Import order**: standard library → third-party → local; alphabetised within each group; no wildcard imports.
- **Docstrings**: Google style on all public functions, methods, and classes — include `Args:`, `Returns:`, and `Raises:` sections.
- **Logging**: use `core.logging_config.get_logger` (backend) or `logging.getLogger(__name__)` (frontend) — never `print()`.
- **Custom exceptions**: raise from `backend.exceptions` for domain validation errors (e.g. `LLMConfigError`, `ForecastingAgentError`).
- **Constants**: module-level constants in UPPER_SNAKE_CASE (e.g. `MAX_JOBS`, `UPLOAD_TIMEOUT`, `_BLEACH_ALLOWED_TAGS`).
- **Private helpers**: prefix with underscore (e.g. `_infer_seasonal_period`, `_calculate_metrics`, `_safe_error_detail`).

### Backend (`backend/`)
- Follow the forecasting instruction file at `.github/instructions/forecasting.instructions.md` for any changes to `backend/forecasting/`.
- Keep agent logic inside `backend/agents/`; prompt strings inside `backend/prompts/`.
- Data schemas (Pydantic models) live in `backend/schemas.py` — use `BaseModel` with `Field(default_factory=...)` for mutable defaults.
- Route handlers in `main.py` must stay **thin** — delegate business logic to `services/`.
- Services layer (`services/`) contains business logic: `file_service`, `job_service`, `pipeline_service`, `chat_service`, `rag_service`.
- Use `core.llm_factory.get_llm(temperature=0)` to obtain LLM instances — never instantiate `ChatOllama` or `ChatGoogleGenerativeAI` directly outside `llm_factory.py`.
- Prompts are `ChatPromptTemplate.from_messages([...])` with `system` + `human` tuples, wrapped with `apply_token_budget()`.
- Token tracking: use `utils.token_tracking.extract_token_usage()` and `estimate_input_text()` for every LLM call.
- Thread-safe singletons: use module-level locks with double-check pattern (see `rag_service.get_rag_kb`, `file_service._file_store_lock`).
- API auth: `require_api_key` and `require_admin_api_key` dependencies from `auth/dependency.py`.
- API keys are hashed with Argon2id via `auth/argon2_helpers.py` — never store plaintext keys.
- File storage is disk-backed parquet (`file_service.py`) with an in-memory metadata index — never hold DataFrames in memory long-term.
- Job queue is an in-memory `asyncio.Queue` with a background worker (`job_service.py`).

### Backend Forecasting Models (`backend/forecasting/`)
- Each model function (`fit_arima`, `fit_holt_winters`, `fit_sarima`, `fit_ewma`) accepts `(series: pd.Series, forecast_horizon: int)` and returns a `dict` with keys: `forecast`, `lower_ci`, `upper_ci`, `rmse`, `mae`, `mape`.
- Always call `series.dropna().astype(float)` at the start of every model function.
- Use 80/20 train-test split for metrics: `split = max(int(len(series) * 0.8), len(series) - forecast_horizon)`.
- Wrap fitting and metric calculation in `try/except`; on exception, `logger.warning(...)` and set metrics to `0.0` (or `float('nan')` per the forecasting instructions).
- `fit_sarima` falls back to non-seasonal ARIMA when `len(series) < 2 * seasonal_period`.
- `_infer_seasonal_period` inspects `series.index.freq`: monthly→12, quarterly→4, weekly→52, daily→7, default 12.
- Preserve the pmdarima/sklearn compatibility shim at the top of `arima_model.py` and `sarima_model.py`.

### Frontend (`frontend/`)
- Use the **application factory** pattern — do not import `app` directly; use `current_app`.
- Keep routes inside blueprints; do not add routes to `app.py`.
- Blueprint structure: `__init__.py` creates the `Blueprint` object, then imports `routes` at the bottom (e.g. `from blueprints.main import routes  # noqa: E402, F401`).
- All form classes use **Flask-WTF** (`FlaskForm`) with CSRF enabled — never disable CSRF (except `TestingConfig`).
- Form fields use WTForms validators: `DataRequired`, `Length`, `EqualTo`, `Optional`, `URL`, `NumberRange`.
- Database access goes through `db.db.query_db` / `db.db.execute_db`; no raw `sqlite3` calls outside `db/`.
- `query_db(sql, args, one=False)` returns `list[dict]` or `dict | None`; `execute_db(sql, args)` returns `lastrowid`.
- Sensitive values (API credentials) are encrypted via `db/crypto.py` (Fernet) before storage.
- Admin-only routes must use the `admin_required` decorator from `blueprints/admin/routes.py`.
- `password_change_required` decorator from `blueprints/decorators.py` redirects to password change when `must_change_password` is set.
- JavaScript assets (`static/js/`) communicate with Flask AJAX endpoints, which proxy requests to the backend.
- Sanitize any HTML rendered from backend responses with `bleach` before passing to templates — use the `_BLEACH_ALLOWED_TAGS` / `_BLEACH_ALLOWED_ATTRS` pattern.
- Chat rendering uses `marked.parse()` + `DOMPurify.sanitize()` on the client side.
- `BackendAPIClient` (`services/api_client.py`) is the single HTTP client for backend calls — use `get_api_client()` factory.
- Config classes: `BaseConfig` → `DevelopmentConfig` / `ProductionConfig` / `TestingConfig`, selected via `FLASK_ENV`.
- `ProductionConfig.__init__` raises `RuntimeError` if `SECRET_KEY` is not set — preserve this guard.
- Frontend type hints: use `str | None` syntax (enabled by `from __future__ import annotations`).
- Frontend mypy: `strict = True`, `ignore_missing_imports = True` — all code must pass strict mypy.
- `# type: ignore[import-untyped]` on Flask-WTF and Flask-Session imports (they lack type stubs).
- `# type: ignore[misc]` on `FlaskForm` subclass declarations.

### Frontend JavaScript
- Use IIFE pattern: `(function () { "use strict; ... })();`
- ES5-compatible syntax (`var`, `function` declarations) — no ES6+ arrow functions, `let`/`const`, or template literals.
- Read CSRF token from `<meta name="csrf-token">` for all AJAX POSTs.
- Use `fetch()` for AJAX calls (not jQuery).
- Chart rendering via Plotly (`Charts.renderAll`, `Charts.renderPie`, `Charts.renderDynamic`).
- Dark theme layout overrides in `charts.js` (`DARK_LAYOUT_OVERRIDES`).

### Testing
- Tests live in `tests/` (root) and `data_forecaster/tests/` and are run with `pytest`.
- All public functions must have corresponding unit tests.
- Mock external HTTP calls and LLM calls; do not make real network requests in tests.
- Test fixtures use `pytest` fixtures with `autouse` for DB resets (see `test_api_key_roles.py`).
- Use `monkeypatch` for environment variable patching — patch both `os.environ` and cached module attributes (e.g. `monkeypatch.setattr(settings, "API_KEY_ENABLED", True)`).
- `sys.path.insert(0, ...)` at the top of test files to make backend/frontend modules importable.
- Use `SimpleNamespace` for mocking LLM response objects (see `test_token_tracking.py`).
- FastAPI tests use `TestClient` from `fastapi.testclient`.

---

## Code Quality Standards

### Maintainability
- Write self-documenting code with clear naming (snake_case functions, PascalCase classes).
- Follow established patterns for consistency — keep functions focused on single responsibilities.
- Limit function complexity to match existing patterns (e.g. `run_pipeline` uses `_progress` helper and `# pylint: disable=too-many-locals` when needed).
- Module-level docstrings describe the module's purpose and relationship to other modules.

### Performance
- Use pandas vectorized operations instead of iterating through DataFrames.
- Disk-backed file storage (parquet) with lazy loading — never hold large DataFrames in process memory.
- In-memory stores have eviction caps (`MAX_INMEMORY_FILES=50`, `MAX_INMEMORY_JOBS=100`) — preserve these bounds.
- Thread-safe singletons with double-check locking for expensive resources (RAG KB, file index).
- `matplotlib.use("Agg")` for non-interactive chart generation.
- Plotly figures are serialised to dict (`_fig_to_dict`) for JSON transport, not rendered server-side.

### Security
- Validate all input data before processing (file extensions, MIME types, column detection).
- API keys hashed with Argon2id (`argon2-cffi`) — never store plaintext.
- Frontend credentials encrypted with Fernet (`cryptography.fernet`) before DB storage.
- `FLASK_ENCRYPTION_KEY` must be set — `db/crypto.py` raises `RuntimeError` if missing.
- HTML sanitization with `bleach` using explicit allowed tags/attrs lists.
- Chat visualization configs validated against whitelists (`_ALLOWED_PLOTLY_TYPES`, `_ALLOWED_TRACE_KEYS`, `_MAX_DATA_POINTS=10000`) to prevent XSS/injection.
- `_is_safe_scalar` rejects strings containing `<script`, `javascript:`, or `data:` patterns.
- Auth dependencies return generic 401s that don't reveal whether username or key was invalid.
- `ProductionConfig` enforces non-default `SECRET_KEY` and API credentials.
- Parameterized SQL queries throughout (`?` placeholders) — no string interpolation in SQL.
- `secrets.token_urlsafe(32)` for API key generation.

### Accessibility
- Frontend templates use Bootstrap classes (`alert`, `btn`, `form-control`).
- Form fields have associated labels via WTForms.
- Progress indicators have ARIA-friendly text updates (`progress-text`, `progress-bar`).

### Testability
- Thin route handlers delegate to service layer — services are independently testable.
- `get_llm()` factory enables LLM mocking in tests.
- `TestClient` for FastAPI integration tests.
- Fixtures provide isolated temp directories (`tmp_path`) for SQLite/ChromaDB/file storage.

---

## Documentation Requirements

- Follow the **Standard** documentation level: Google-style docstrings on all public functions, methods, and classes.
- Include `Args:`, `Returns:`, and `Raises:` sections in docstrings.
- Module-level docstrings describe the module's purpose (e.g. `"""Job queue service for the Data Forecaster backend."""`).
- For complex functions, include usage examples in the docstring (see `scripts/reset_api_key.py`).
- Class docstrings list fields/attributes (see form classes in `blueprints/admin/forms.py`).
- Inline comments use `# ── Section Name ─────────────────────...` pattern for visual section breaks (match existing style).

---

## Testing Approach

### Unit Testing
- Match the structure of existing tests in `tests/` and `data_forecaster/tests/`.
- Use `pytest` fixtures for test data (e.g. `sample_series`, `series_with_missing`).
- Class-based test grouping: `class TestExtractTokenUsage:` with `test_*` methods.
- Use `SimpleNamespace` for mocking objects with attributes.
- Assert on dict equality for structured return values.

### Integration Testing
- FastAPI integration via `TestClient` (see `test_api_key_roles.py`).
- `autouse` fixtures reset the API key DB to a temp directory for each test.
- Patch both env vars and cached module attributes when config is read at import time.

---

## Python Guidelines

- **Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)** as the primary external standard — see [Google Python Style Guide Compliance](#google-python-style-guide-compliance) for the full rule set.
- Detect and adhere to Python 3.11 (backend) / 3.12 (frontend Docker).
- Follow the same import organization: `from __future__ import annotations` first, then stdlib, third-party, local — one import per line, sorted lexicographically (Google §3.13).
- Match type hinting approaches: `str | None`, `list[str]`, `dict[str, Any]`, `tuple[float, float, float]` — prefer `collections.abc` abstract types in signatures (Google §3.19.12).
- Apply the same error handling patterns: `try/except` with `logger.warning`, fallback return values — minimise code inside `try` blocks, never use bare `except:` (Google §2.4).
- Follow the same module organization: docstring → imports → constants → functions/classes.
- Use `# noqa: E402` for imports that must come after side-effect imports (e.g. blueprint route imports).
- Use `# type: ignore[...]` with specific error codes, not bare `# type: ignore` (Google §3.19.7).
- Use `# pylint: disable=symbolic-name` for pylint suppressions with an explanation (Google §2.1).
- Logging: pass `%`-pattern strings, not f-strings: `logger.info('Value: %s', val)` (Google §3.10.1).
- No mutable default arguments: `def f(x: list = [])` is forbidden — use `None` (Google §2.12).
- No semicolons, no backslash line continuation, one statement per line (Google §3.1, §3.2, §3.14).
- Functions should be small and focused — reconsider if exceeding ~40 lines (Google §3.18).
- Naming: `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants, no single-char names except counters/exceptions (Google §3.16).

---

## Version Control Guidelines

- Follow **Semantic Versioning** as applied in `pyproject.toml` (`version = "1.0.0"`).
- The project version is `1.0.0` — update in `pyproject.toml` when releasing.

---

## Environment Variables

| Variable | Used by | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | backend | Gemini API key |
| `GEMINI_MODEL` | backend | Model name (default `gemini-1.5-flash`) |
| `GEMINI_TEMPERATURE` | backend | LLM temperature (default `0.1`) |
| `USE_OLLAMA` | backend | Switch to Ollama instead of Gemini |
| `USE_OLLAMA_CLOUD` | backend | Use Ollama Cloud (requires `OLLAMA_API_KEY`) |
| `OLLAMA_BASE_URL` | backend | Ollama server URL (default `http://host.docker.internal:11434`) |
| `OLLAMA_MODEL` | backend | Ollama model name (default `llama3`) |
| `OLLAMA_API_KEY` | backend | Ollama Cloud API key |
| `MAX_UPLOAD_MB` | both | Max file upload size (backend default 100, frontend default 200) |
| `CHROMA_PERSIST_DIR` | backend | ChromaDB storage path (default `./chroma_db`) |
| `FILE_STORAGE_DIR` | backend | Parquet file storage directory (default `./file_store`) |
| `API_KEY_DB_PATH` | backend | API key SQLite DB directory (default `./data`) |
| `API_KEY_ENABLED` | backend | Enable API key auth (default `false`) |
| `ADMIN_API_KEY` | backend | Secret for bootstrap endpoint protection |
| `FRONTEND_API_USERNAME` | both | Pre-shared service account username (default `frontend`) |
| `FRONTEND_API_KEY` | both | Pre-shared service account key (default `frontend`) |
| `CORS_ALLOWED_ORIGINS` | backend | Comma-separated allowed CORS origins |
| `MAX_INMEMORY_FILES` | backend | Max uploaded files before eviction (default 50) |
| `MAX_INMEMORY_JOBS` | backend | Max jobs before eviction (default 100) |
| `FLASK_ENV` | frontend | `development` / `production` / `testing` |
| `SECRET_KEY` | frontend | Flask session secret (required in production) |
| `FLASK_ENCRYPTION_KEY` | frontend | Fernet key for encrypting stored API credentials |
| `BACKEND_URL` | frontend | URL of the FastAPI backend (default `http://localhost:8000`) |
| `API_VERIFY_SSL` | frontend | Verify backend TLS cert (default `false`) |
| `FRONTEND_ADMIN_USERNAME` | frontend | Default admin username (default `admin`) |
| `FRONTEND_ADMIN_PASSWORD` | frontend | Default admin password (default `admin`) |
| `DEMO_DATA_PATH` | frontend | Path to demo CSV file |

---

## Key Architectural Notes

- The Flask frontend is a **thin client**: it delegates all forecasting work to the FastAPI backend via `services/api_client.py` (`BackendAPIClient`).
- The frontend stores a `job_id` in the Flask session and polls `/ajax/job-status/<job_id>` via `static/js/polling.js` (1.5s interval).
- User accounts and roles are stored in SQLite (`instance/forecaster.db`). The DB is auto-seeded on first startup via `init_db()`.
- The admin blueprint (`/admin/`) allows admin users to create accounts, reset passwords, and configure the backend URL and API credentials.
- The backend uses a **6-agent pipeline** (`pipeline_service.run_pipeline`): data validation → statistical analysis → model selection → forecasting → statistical review → report generation.
- Each agent combines **Python-computed metrics** with **LLM reasoning** — the LLM never computes numbers, only interprets pre-computed results.
- RAG knowledge base (`rag/knowledge_base.py`) uses ChromaDB with `all-MiniLM-L6-v2` embeddings, chunked at 400 chars with 80 char overlap.
- The backend auto-creates a `frontend` service account from `FRONTEND_API_USERNAME`/`FRONTEND_API_KEY` env vars on first startup.
- The previous Streamlit frontend has been fully replaced by this Flask application. Do not reference Streamlit anywhere in the codebase.

---

## General Best Practices

- Follow naming conventions exactly as they appear in existing code.
- Match code organization patterns from similar files.
- Apply error handling consistent with existing patterns (`try/except` + `logger.warning` + fallback values).
- Follow the same approach to testing as seen in the codebase (pytest fixtures, `monkeypatch`, `TestClient`).
- Match logging patterns (`get_logger(__name__)` backend, `logging.getLogger(__name__)` frontend).
- Use the same approach to configuration (`os.getenv` with defaults in `core/config.py` and `frontend/config.py`).

---

## Project-Specific Guidance

- Scan the codebase thoroughly before generating any code.
- Respect existing architectural boundaries without exception (backend services vs agents vs utils; frontend blueprints vs services vs db).
- Match the style and patterns of surrounding code **while actively applying the Google Python Style Guide** — if existing code violates the Google standard, follow the Google standard for new/modified code and note the deviation.
- When in doubt, prioritize the Google Python Style Guide over existing codebase patterns for Python style decisions (naming, formatting, docstrings, type hints, error handling). Prioritize codebase patterns for architectural decisions (service boundaries, blueprint structure, agent pipeline).
- Never reference Streamlit — it has been fully replaced by Flask.
- Preserve the pmdarima/sklearn compatibility shim in forecasting modules.
- Always use `from __future__ import annotations` as the first import.
- Run SonarQube analysis on modified files at the end of a task (per `.github/instructions/sonarqube_mcp.instructions.md`).
