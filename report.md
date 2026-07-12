# Enterprise Code Review Report

**Repository:** `data_forecasting_agent`  
**Review date:** 2026-07-12  
**Scope:** Python backend and frontend, JavaScript, templates, tests, CI, Docker,
configuration, and project documentation. Vendored/minified JavaScript was treated as
third-party code and was not style-reviewed.

## Executive summary

The application has a credible architecture and several good security foundations:
parameterized SQL, Argon2id API-key hashes, Werkzeug password hashes, CSRF protection,
server-side sessions, ownership checks, upload-size limits, output sanitization, generic
authentication failures, and secret examples rather than committed live `.env` files.
The code is not, however, at an enterprise release standard yet.

The main concerns are fail-open/default credentials, a production Compose default that
neutralizes the application's `SECRET_KEY` validation, informational rather than
enforcing quality/security gates, large modules and orchestration functions, widespread
broad exception handling, import-time side effects that force scattered imports, and
incomplete API/test documentation. The most important recommendation is to make secure
configuration and automated enforcement prerequisites for production deployment before
undertaking broad structural work.

This report deliberately does **not** recommend wholesale rewrites or abstraction for
its own sake. Large code should be split only where there is a clear boundary, a security
benefit, or an independently testable responsibility.

## Review baseline and validation

The review applied:

- `.github/copilot-instructions.md` as the primary repository standard, including the
  Google Python Style Guide, 88-character limit, Google docstrings, absolute imports,
  top-level imports, explicit exception handling, and layered service boundaries.
- `.github/copilot/copilot-instructions.md` and scoped instructions in
  `.github/instructions/`, including forecasting behavior and testing requirements.
- Enterprise security and operations expectations: secure-by-default configuration,
  least privilege, deterministic builds, auditable failures, non-root containers,
  enforced CI, and documented recovery/rotation procedures.

Static AST inspection covered 97 Python modules and found module docstrings in 75 and
docstrings on 414 of 478 public definitions. The codebase contains about 21,000 lines of
Python. The largest files include `backend/report/builder.py` (1,493 lines), frontend
admin and main route modules (1,160 and 1,149 lines), and `backend/main.py` (990 lines).

`python3 -m pytest -q` could not collect the suite in the current environment because
required packages were absent (`fastapi`, `chromadb`) and the installed NumPy/compiled
Matplotlib combination was incompatible. This is an environment result, not evidence
that product tests fail. It does expose that the repository lacks a lightweight,
reproducible review/test bootstrap and that collection imports the full forecasting/RAG
dependency graph.

## Ranked findings: least breaking to most breaking

The order below is the requested implementation-risk order. Severity describes business
risk; breaking risk describes the likelihood that remediation changes behavior or
interfaces.

### 1. Normalize comments and module documentation

**Breaking risk:** Very low  
**Severity:** Low  
**Area:** Documentation, comment placement, consistency

Twenty-two Python modules lack module docstrings, including several agent, forecasting,
parser, statistical, and package modules. Sixty-four public definitions lack docstrings.
Many existing docstrings are good, but several describe implementation rather than the
public contract, omit meaningful `Raises:` information, or use Sphinx roles inconsistently
with otherwise Google-style prose.

Decorative comments such as the pipeline's long Unicode stage separators explain visible
control flow rather than design intent (`backend/services/pipeline_service.py:100` and
following). Configuration comments are sometimes much longer than the setting they
explain (`backend/core/config.py:67-82`). Conversely, important design decisions—why auth
can run in open mode, why TLS verification is disabled in the single-host deployment,
and the trust boundary for administrator-configured backend URLs—are not documented as
formal security decisions.

**Recommendation:** Add concise module/public API docstrings; remove narration that
duplicates code; retain comments that explain statistical constraints, compatibility
shims, security assumptions, or non-obvious invariants. Adopt `# TODO: issue-reference -
Description.` for actionable debt. Add architecture decision records for open mode,
self-signed TLS, service credentials, and model compatibility shims.

### 2. Enforce formatting, import order, lint, typing, and dependency checks in CI

**Breaking risk:** Very low initially; low once existing debt becomes blocking  
**Severity:** High (governance)

CI currently marks Black, pylint, mypy, pip-audit, and Grype as non-blocking
(`.github/workflows/ci.yml:37-50`, `57-70`). Pylint is additionally followed by `|| true`.
Consequently, the repository's strict written standards are not enforced. CI tests only
`tests/` at the repository root (`ci.yml:52-55`) and omits the substantial
`data_forecaster/tests/` suite. The Python matrix contains only 3.11 even though the
frontend production image and frontend mypy configuration use 3.12.

There is no configured Ruff/isort equivalent, coverage threshold, secret scan, SAST,
license policy, or generated dependency lock for the frontend. `requirements.txt` uses
minimum constraints while the instructions call for pinned, reproducible dependencies.

**Recommendation:** First record a lint/type/security baseline without blocking. Then
make changed-file formatting and import order blocking, add both test directories, test
backend on 3.11 and frontend on 3.12, and ratchet lint/type/coverage violations down.
Make high/critical dependency and image findings blocking under an explicit exception
policy. Pin GitHub Actions to commit SHAs for stronger supply-chain control.

### 3. Move imports to module tops and eliminate import-order workarounds

**Breaking risk:** Low  
**Severity:** Medium

Confirmed production import scattering includes:

- `frontend/app.py:22-28`: `load_dotenv()` executes before local imports.
- `frontend/app.py:77`: `manage` is imported inside `create_app`.
- `frontend/blueprints/admin/routes.py:399`: API client import inside a route.
- `backend/auth/api_key_db.py:121`: Argon2 verifier import inside a function.
- `backend/scripts/reset_api_key.py:33-35`: imports follow runtime `sys.path` mutation.
- Blueprint `__init__.py` modules import routes only after constructing a global blueprint.
- `backend/forecasting/arima_model.py` and `sarima_model.py` import pmdarima after applying
  a compatibility monkey patch. `backend/utils/visualization.py` similarly configures
  Matplotlib before later imports.

The forecasting and Matplotlib cases have legitimate ordering constraints, but still
violate the stated top-of-file rule and make import behavior fragile. They should be
encapsulated in explicit compatibility/bootstrap modules rather than normalized blindly.

**Recommendation:** Centralize environment loading in executable entry points before
application modules are imported; make config objects read the prepared environment.
Move ordinary imports to the top. Isolate unavoidable pre-import compatibility setup in
small, documented modules with tests. Replace `sys.path` mutation with package execution
and absolute imports. Avoid changing the pmdarima shim until compatibility tests exist.

### 4. Align code with the documented package and logging conventions

**Breaking risk:** Low to medium  
**Severity:** Medium

The instructions require absolute package imports and backend logging through the project
logger. Current imports (`from agents...`, `from services...`, `from auth...`) depend on
service directories being inserted into `sys.path` rather than a stable installable
package. `pyproject.toml` explicitly declares no packages. This creates different import
identities and makes tests depend on manual path manipulation. Several `__init__.py`
files also omit the required future import and module documentation.

**Recommendation:** Choose and document one import root, preferably
`data_forecaster.backend...` / `data_forecaster.frontend...`, package it correctly, and
run entry points with `python -m`. Migrate incrementally by subsystem because changing
module identity can break monkeypatches and singleton state. Resolve the instruction
conflict that mentions both `core.logging_config` and a nonexistent
`utils.logging_config`.

### 5. Remove low-value duplication and centralize narrow utilities

**Breaking risk:** Low to medium  
**Severity:** Medium

ARIMA and SARIMA contain duplicate `_calculate_metrics` implementations
(`backend/forecasting/arima_model.py:28`, `sarima_model.py:28`); baseline modeling contains
a third metric implementation. Report renderers, PDF rendering, and chat visualization
each implement their own sanitization/conversion rules. API user row-to-dict conversion
is repeated throughout `backend/auth/api_key_db.py`.

**Recommendation:** Extract only behavior that has the same contract. A shared forecast
metric helper and API-user mapper are justified. Do not build a generic “utilities” layer
for unrelated sanitizers: HTML, Markdown-table, PDF-character, and Plotly sanitization
protect different trust boundaries and should remain explicit.

### 6. Replace broad exception handling with boundary-specific failures

**Breaking risk:** Medium  
**Severity:** Medium to high

Frontend main/admin routes contain more than 30 `except Exception` handlers. Some are
reasonable isolation boundaries around backend calls, but many collapse programming,
database, parsing, and network errors into “backend unavailable,” hindering incident
diagnosis and potentially committing partial operations. `run_pipeline` catches broad
exceptions during transformations (`pipeline_service.py:141-148`) and uses `e` rather
than the repository's preferred `exc`. The global FastAPI exception handler is an
appropriate outer boundary, but it should not substitute for domain-specific handling.

**Recommendation:** Catch `requests.RequestException`, SQLite errors, validation errors,
and domain exceptions separately. Keep broad catches only at worker/request isolation
boundaries, always log correlation-safe context, and test fallback behavior. Audit
multi-step admin operations for transactions so frontend state cannot be updated when a
backend update fails.

### 7. Split oversized functions along existing responsibilities

**Breaking risk:** Medium  
**Severity:** High (maintainability and testability)

`backend/services/pipeline_service.run_pipeline` is 439 lines and explicitly suppresses
too-many-locals/branches/statements. `run_statistical_agent` is 286 lines, several other
agent functions exceed 120 lines, and report builder helpers reach 184 lines. The route
files exceed 1,100 lines each, although individual handlers are generally smaller.

These are not merely style problems: pipeline transformations mutate the same series and
result objects across stages, error policy varies by stage, and most behavior can only be
tested through a large dependency graph. `backend/report/builder.py` combines health
scoring, recommendations, risks, assumptions, dashboard construction, and model mapping.

**Recommendation:** Extract named stage functions with typed inputs/results and preserve
the current coordinator sequence. Divide route modules by capability only if blueprint
URLs and endpoint names remain stable. Split report construction into existing concepts
(health, recommendations, risks, dashboard), not arbitrary line-count units. Add
characterization tests first. Do not introduce a generic agent framework or workflow DSL;
the six explicit stages are easier to audit.

### 8. Improve test isolation, coverage, and reproducibility

**Breaking risk:** Medium  
**Severity:** High

Tests manipulate `sys.path`, and importing ownership/job code pulls in pmdarima,
Matplotlib, ChromaDB, and the complete agent pipeline. This made a storage-ownership test
uncollectable when unrelated ML dependencies were unavailable. Public forecasting
functions do not all have the unit tests mandated by scoped instructions. There are no
clear tests for production configuration rejection, SSRF/backend URL policy, session
security, upload decompression/resource limits, rate limiting, or deployment defaults.

**Recommendation:** Inject the pipeline callable into the job worker or import it behind
an application composition boundary; keep domain/storage tests independent of ML/RAG.
Create fast unit, service integration, statistical regression, and end-to-end test tiers.
Use deterministic fixtures and tolerances for model output. Add coverage reporting by
critical subsystem rather than pursuing a single vanity percentage.

### 9. Strengthen input/resource controls and browser security policy

**Breaking risk:** Medium  
**Severity:** High

Uploads are byte-limited, streamed, extension/MIME checked, and stored under generated
identifiers—good foundations. Remaining gaps include no demonstrated XLSX decompression
or parser resource budget, row/column/cell limits, request rate limiting, authentication
attempt throttling, or forecast-compute quotas beyond concurrent job counts. Backend
security headers omit a Content Security Policy, Referrer-Policy, and Permissions-Policy.
The frontend constructs HTML strings with `innerHTML`; values are escaped in the reviewed
preflight path, but this pattern is fragile and a future omission becomes DOM XSS.

**Recommendation:** Add parser-level limits and time budgets, per-user request/upload/job
quotas, login/API authentication throttling, and explicit retention limits. Prefer DOM
node construction for ordinary UI; reserve sanitized HTML for Markdown. Add a CSP in
report-only mode, inventory required script/style directives, then enforce it. Add the
remaining headers at the external Nginx boundary.

### 10. Restrict administrator-configured outbound backend URLs

**Breaking risk:** Medium to high  
**Severity:** High

The frontend stores a configurable backend base URL and sends API credentials to it on
every request (`frontend/services/api_client.py:45-61`, e.g. `94-100`). An administrator
can intentionally configure it, but a compromised admin session can turn the frontend
into an SSRF client and exfiltrate the backend API key to an attacker-controlled host.
The review found no scheme, host, port, DNS rebinding, redirect, or private-address policy.
The `requests` defaults also follow redirects, potentially forwarding sensitive headers.

**Recommendation:** Require HTTPS in production, validate against an operator-managed
allowlist, reject userinfo and unexpected ports, resolve and validate destination
addresses, and disable redirects for credentialed requests. Separate connectivity tests
from credentialed tests. Treat expanding the allowlist as deployment configuration, not
a web-admin operation. Test redirects and DNS/IP edge cases.

### 11. Make authentication and secret configuration fail closed

**Breaking risk:** High  
**Severity:** Critical

The backend defaults to the public `frontend`/`frontend` service credential
(`backend/core/config.py:78-84`) and starts in open mode when it cannot establish users
(`backend/main.py:111-143`). That makes a configuration or database/bootstrap failure an
authorization bypass, not merely a startup failure. The documentation acknowledges the
default but enterprise deployments should never rely on operators noticing a warning.

Production frontend validation checks only that `SECRET_KEY` is non-empty
(`frontend/config.py:88-99`); it does not reject the known development value or enforce
entropy. It also does not validate required service/admin credentials, HTTPS backend URL,
TLS verification, writable secure session paths, or environment selection.

**Recommendation:** Introduce an explicit `ALLOW_INSECURE_DEVELOPMENT` mode confined to
local development. In production, fail startup if auth is disabled, users/credentials are
missing, default values are present, TLS verification is disabled for a remote backend,
or secrets do not meet length/entropy requirements. Use secret files or a secrets manager
rather than environment defaults. Provide a migration/runbook before removing existing
defaults.

### 12. Correct deployment defaults and harden containers

**Breaking risk:** High  
**Severity:** Critical

`docker-compose.yml:47` and `docker-compose.distributed.yml:35` set
`SECRET_KEY=${SECRET_KEY:-change-me-in-production}` while forcing `FLASK_ENV=production`.
Because the value is non-empty, `ProductionConfig` accepts it, so every deployment that
omits the variable shares a known session-signing secret. The single-host frontend also
sets `API_VERIFY_SSL=false` (`docker-compose.yml:46`). Self-signed internal TLS plus
disabled verification provides encryption without server authentication and gives a
false sense of trust; authenticated plain HTTP on a private container network or a
properly trusted internal CA is clearer.

The Dockerfiles do not establish an application user, so workloads appear to run as root.
Compose lacks read-only roots, dropped capabilities, `no-new-privileges`, resource limits,
and explicit secret mounts. Base images are tag-pinned rather than digest-pinned. Nginx
generates persistent self-signed keys automatically, an onboarding convenience that is
not an enterprise certificate lifecycle.

**Recommendation:** Remove secret fallback expressions so Compose fails when required
values are absent; use Compose required-variable syntax or secret mounts. Choose verified
TLS with an internal CA or documented private-network HTTP. Run all services as non-root,
use read-only roots with named writable mounts/tmpfs, drop capabilities, enable
`no-new-privileges`, set CPU/memory/PID limits, pin release images by digest, and move
certificate provisioning/rotation outside the container entrypoint. Validate backup,
restore, migration, and key-rotation procedures before production rollout.

## Cross-cutting observations

### Architecture and unnecessary complexity

The layered split—FastAPI backend, Flask frontend, services, agents, forecasting, reports,
and prompts—is understandable and should be retained. The highest complexity is local,
not a reason to replace the architecture. Avoid introducing repositories for every table,
a universal agent base class, event buses, or a workflow engine unless concrete scale or
operational requirements demand them.

The best simplifications are direct:

- Make `run_pipeline` a short coordinator of six typed stage functions.
- Keep route handlers thin and move only reusable business operations into services.
- Replace mutable module configuration (`set_api_key_enabled`) with explicit application
  state/configuration and a clear startup state machine.
- Separate lightweight job/storage modules from optional ML and RAG imports.
- Use one canonical representation for API users and forecast metric results.

### Documentation gaps

README and deployment/API documents provide a useful start, but enterprise operation
also needs:

- A supported-version and dependency update policy.
- Threat model and data classification/retention policy.
- Authentication bootstrap, rotation, revocation, and break-glass runbooks.
- Backup/restore and SQLite migration/recovery procedures.
- Observability guidance: structured logs, correlation IDs, metrics, alert thresholds,
  audit event retention, and redaction rules.
- Statistical model limitations, reproducibility expectations, and model/LLM audit trail.
- API compatibility/versioning and deprecation policy.
- Release checklist and rollback procedure.

### Positive findings to preserve

- SQL calls reviewed use bound parameters rather than string interpolation.
- API keys are generated with `secrets`, stored as Argon2id hashes, and returned only at
  creation/rotation.
- Authentication errors are intentionally generic and API key hashes are excluded from
  response mappings.
- CSRF and Flask-Login protections are broadly applied; admin routes use a centralized
  role decorator.
- File and job ownership checks exist for non-admin API users.
- Markdown output is sanitized, and preflight values interpolated into HTML are escaped.
- Uploads are streamed to temporary storage and checked against configured limits.
- The backend global error handler avoids returning exception details to clients.
- Security scanning workflows and dependency review exist, even though enforcement needs
  improvement.

## Phased remediation plan

### Phase 0 — Release guardrails and decisions (1–2 weeks)

1. Declare that current Compose defaults are development-only; block production releases
   until items 11 and 12 have accepted owners and dates.
2. Remove/override the known production `SECRET_KEY` fallback immediately and rotate the
   key anywhere it may have been used. Rotation logs out all existing sessions.
3. Inventory deployments using default backend credentials, open mode, or disabled TLS
   verification; rotate credentials and document exceptions.
4. Capture a clean CI baseline covering both test trees, Python 3.11/3.12, Black, pylint,
   mypy, pip-audit, and image scanning.
5. Write short security decisions for backend URL trust, TLS topology, and development
   open mode.

**Exit criteria:** No production environment uses a known signing/API key; CI executes
both suites; all insecure deployment exceptions are explicit and time-bound.

### Phase 1 — Low-risk standards sweep (1–2 weeks)

Address findings 1–5:

1. [x] Add missing module documentation and remove redundant comments.
   - Completed 2026-07-12: added module docstrings to every Python module that
     lacked one; AST validation now reports zero missing module docstrings.
   - Remaining follow-up: continue public API docstring cleanup during touched-code
     work rather than bulk editing every public definition in one pass.
2. [x] Apply deterministic formatting/import sorting in reviewable subsystem-sized commits.
   - Completed 2026-07-12: CI now runs Black and isort as blocking checks.
3. [x] Move ordinary imports to module tops; isolate pmdarima/Matplotlib bootstrap ordering.
   - Completed 2026-07-12: moved ordinary local imports in the frontend factory,
     admin/main routes, frontend CLI/API/database modules, backend main module,
     API-key verifier, reset script, and EWMA model.
   - Completed 2026-07-12: isolated pmdarima and Matplotlib ordering requirements
     into `forecasting.pmdarima_compat` and `utils.matplotlib_backend` with focused
     compatibility tests.
   - Intentional remaining exceptions: blueprint package route-registration imports
     remain after blueprint construction, and the interactive backend bootstrap script
     keeps optional backend imports inside the reset helper so it can run on a host
     before backend dependencies are installed.
4. [x] Add changed-file CI gates, then ratchet the baseline.
   - Completed 2026-07-12: CI now blocks on Black, isort, both test trees, pip-audit,
     and Grype high findings; pylint and mypy remain informational while their
     existing baseline is ratcheted down.
5. [x] Consolidate only identical metric/mapping logic and add unit tests.
   - Completed 2026-07-12: extracted shared ARIMA/SARIMA holdout metric calculation
     into `forecasting.metrics` and added focused regression tests.

**Exit criteria:** Complete as of 2026-07-12. New/changed code passes syntax,
documentation, focused compatibility, and formatting checks; no unexplained scattered
production imports remain; compatibility shims have focused tests.

### Phase 2 — Error handling and test seams (2–4 weeks)

Address findings 6 and 8:

1. [x] Define domain exception categories and map them consistently at HTTP/worker boundaries.
   - Completed 2026-07-12: added backend domain exception categories
     (`DataValidationError`, `PipelineExecutionError`, `StorageAccessError`) and
     converted the short-series pipeline stop to a domain validation error while
     preserving `ValueError` compatibility.
   - Remaining follow-up: continue mapping these categories at each HTTP/worker
     boundary as broad handlers are narrowed.
2. [x] Replace broad catches inside routes/services and add transaction tests for multi-step
   admin actions.
   - Partially completed 2026-07-12: narrowed the Box-Cox remediation handler to
     expected value/type failures.
   - Partially completed 2026-07-12: removed broad catches from frontend admin routes,
     narrowed admin backend-call failures to request/JSON/encryption/database errors,
     and added a fast guard that frontend admin settings are persisted only after
     backend job settings are accepted.
   - Partially completed 2026-07-12: narrowed broad catches in frontend main
     backend-proxy routes and backend upload/preflight/LLM-health request handlers;
     added a source guard that allows remaining route-level broad catches only at
     documented isolation boundaries.
   - Completed 2026-07-12: narrowed pipeline, chat-service, file-service, visualization,
     job indexing, and frontend report-persistence handlers to expected exception
     families. Remaining broad catches are limited by source tests to the backend chat
     request boundary and the job worker boundary.
3. [x] Decouple storage/job tests from ML/RAG imports through composition/injection.
   - Completed 2026-07-12: job scheduling no longer imports the forecasting pipeline or
     RAG service at module load; those optional dependencies are loaded only when a job
     executes or completed results are indexed. Added a source guard to prevent
     accidental reintroduction of eager ML/RAG imports.
4. [x] Create fast unit and integration tiers with deterministic dependency installation.
   - Completed 2026-07-12 for the first fast unit slice: added focused tests that
     avoid optional ML/RAG service imports and updated CI to execute both test trees
     under Python 3.11 and 3.12.
   - Remaining follow-up: formalize markers or separate CI jobs for unit vs.
     integration coverage.
5. [x] Add security regression tests for production config, ownership, auth failure, and
   sensitive error redaction.
   - Completed 2026-07-12 for production config; ownership and auth-failure tests
     already existed before this pass.
   - Completed 2026-07-12: added explicit connection-error redaction tests so API
     keys, bearer values, and URL credentials are not reflected to the browser.

**Exit criteria:** Complete as of 2026-07-12. Core tests for this slice collect without
optional ML/RAG services; broad catches exist only at documented isolation boundaries;
quality gates are blocking for touched code.

### Phase 3 — Targeted maintainability changes (3–6 weeks)

Address finding 7 and the packaging part of finding 4:

1. [x] Characterize current pipeline outputs and progress events.
   - Completed 2026-07-12: added a progress-event characterization test for the
     current pipeline orchestration contract.
2. [ ] Extract pipeline stages one at a time with typed contracts; keep orchestration explicit.
   - Not completed in this pass; characterization is now in place so extraction can
     proceed in safer follow-up commits.
3. [ ] Split report-builder responsibilities along existing report concepts.
   - Not completed in this pass; requires behavioral characterization before moving
     rendering/report assembly code.
4. [ ] Split frontend route modules only where capability ownership is clear.
   - Not completed in this pass; ordinary import cleanup was done without changing
     route ownership boundaries.
5. [ ] Establish canonical packages/import roots and migrate entry points and tests
   incrementally.
   - Partially completed 2026-07-12: reset script no longer performs unconditional
     `sys.path` mutation before imports.
   - Remaining follow-up: define the canonical package root and migrate tests/entry
     points away from ad hoc path insertion.

**Exit criteria:** No central coordinator mixes statistical computation, rendering,
fallback policy, and persistence; package imports no longer require `sys.path` mutation;
behavioral characterization tests remain green.

### Phase 4 — Security boundary enforcement (2–4 weeks)

Address findings 9–11:

1. Add rate limits, parser budgets, user quotas, and retention enforcement.
2. Introduce and enforce CSP/security headers after report-only validation.
3. Validate/allowlist backend destinations and disable credentialed redirects.
4. Implement production config validation and explicit local-only insecure mode.
5. Move credentials to managed secret delivery and add rotation/revocation audit events.

**Exit criteria:** Production startup is fail-closed; outbound credential destinations are
constrained; authentication and expensive endpoints have tested abuse controls.

### Phase 5 — Deployment hardening and operational readiness (3–6 weeks)

Address finding 12 and operational documentation:

1. Build non-root, least-privilege containers with read-only roots and resource limits.
2. Replace automatic production certificates with managed issuance and rotation.
3. Pin release artifacts, generate SBOMs, sign images, and enforce vulnerability policy.
4. Test backup/restore, database migration, disaster recovery, secret rotation, and
   rollback.
5. Add structured audit logs, correlation IDs, metrics, dashboards, and alerting.

**Exit criteria:** A release can pass a documented security/operations checklist; restore
and rollback exercises succeed; production images and configuration are reproducible.

## Suggested issue ordering

Create small, independently reviewable issues in this order:

1. Remove known Compose `SECRET_KEY` defaults and rotate affected secrets.
2. Run both test directories in CI and add frontend Python 3.12 coverage.
3. Baseline and progressively enforce formatter/import/lint/type checks.
4. Document insecure development mode and enforce production configuration validation.
5. Restrict credentialed backend URLs and redirects.
6. Isolate optional ML/RAG imports from storage and job tests.
7. Replace broad exceptions subsystem by subsystem.
8. Add parser/rate/quota controls and browser security policy.
9. Extract pipeline stages with characterization tests.
10. Normalize package imports and remove `sys.path` manipulation.
11. Split report and route modules only at stable responsibility boundaries.
12. Harden containers and complete operational runbooks.

This ordering delivers immediate security value, establishes enforcement before a large
cleanup, and postpones the most behavior-sensitive structural changes until tests can
detect regressions.
