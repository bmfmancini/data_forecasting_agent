## 1. Forecast Setup and Submission

- [x] 1.1 Add an optional, accessible report-title input with a 200-character limit and default-name guidance to Step 3 of `forecast_setup.html`, populated from setup state.
- [x] 1.2 Extend the setup-state endpoint and browser wizard code to save, restore, and submit `report_title` with the existing forecast configuration.
- [x] 1.3 Add a shared Flask title-normalization helper that trims custom values, applies `Forecast Report — <dataset stem>` for blank values, rejects over-length input before job creation, and returns field-specific errors.
- [x] 1.4 Use the normalized title as the existing `report_name` in the analyze payload and add backend request-schema length validation as defense in depth.

## 2. Durable Report Metadata

- [x] 2.1 Extend `ReportMetadata` with backwards-compatible title and prepared-by fields while retaining `generated_at` as the canonical UTC forecast creation timestamp.
- [x] 2.2 Propagate the durable job's resolved `report_name` and authenticated `application_username` from the worker through `run_pipeline`, the report stage, report agent, and executive report builder.
- [x] 2.3 Populate structured report identity during deterministic report construction and include title, prepared-by, and forecast creation date in the backend Markdown and HTML metadata renderers.
- [x] 2.4 Add or update backend unit tests for request validation, per-job identity propagation, metadata construction, renderer escaping, UTC timestamps, and old serialized metadata compatibility.

## 3. Saved Report Persistence

- [x] 3.1 Change `save_report` to accept a resolved title while retaining the dataset-based helper as a compatibility fallback for callers that do not supply one.
- [x] 3.2 Update automatic polling finalization and explicit job finalization to read the durable job's `report_name` and persist it, preventing stale Flask setup state from affecting concurrent jobs.
- [x] 3.3 Ensure decoded/listed saved reports expose the stored title and saved timestamp needed by presentation fallbacks without changing report ownership or idempotency checks.
- [x] 3.4 Add persistence and route tests for custom/default titles, concurrent job finalization, idempotent re-finalization, rename behavior, and legacy records.

## 4. Web and PDF Presentation

- [x] 4.1 Build a shared report-identity presentation helper that prefers a saved report's current title, formats structured `generated_at` in UTC, and applies documented legacy fallbacks for missing title, author, or date.
- [x] 4.2 Replace the hard-coded web report hero title with the resolved report title and display `Prepared by` plus the forecast creation date using escaped template output.
- [x] 4.3 Extend PDF generation to print the same title, prepared-by username, and creation date in its first-page title block and derive a bounded filesystem-safe download name from the display title.
- [x] 4.4 Add web-rendering and PDF tests covering matching identity values, Unicode/markup-safe titles, renamed reports, creation-date stability, and missing legacy metadata.

## 5. Verification

- [x] 5.1 Run the focused frontend and backend report, job, route, rendering, and PDF test suites and resolve regressions.
- [x] 5.2 Run the full automated test suite and static checks configured by the project, documenting any unrelated pre-existing failures.
- [x] 5.3 Manually verify Forecast Setup with custom, blank, whitespace-only, and over-length titles, then confirm identity consistency in the job queue, saved-report list, web report, renamed report, and exported PDF.
