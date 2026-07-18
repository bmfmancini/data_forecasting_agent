## Context

The Forecast Setup wizard currently persists forecast horizon, model choice, and user context in the Flask session. At submission time, the frontend creates a dataset-based `report_name`, sends it to the backend, and the backend stores it on the durable forecast job for queue display. Completed reports are saved in the frontend database, but `save_report` independently creates another dataset-based title. The web report header is hard-coded, PDF titles/filenames are derived from the source filename, and structured `ReportMetadata` contains a UTC `generated_at` timestamp but no title or author.

The backend job already stores `report_name`, `application_username`, and job timestamps. The frontend report table already stores `title` and `created_at`. This change should use those existing paths rather than introduce a second report identity store or a database migration.

## Goals / Non-Goals

**Goals:**

- Let a user optionally name a report before submitting a forecast.
- Resolve one canonical report title and carry it through the job queue, structured report, saved report, web view, and PDF export.
- Attribute new reports to the authenticated application username without trusting a browser-supplied author value.
- Use the structured report's UTC `generated_at` value as the forecast/report creation timestamp and display it clearly.
- Keep previously saved reports viewable and exportable.
- Preserve the existing saved-report rename behavior without changing author or creation metadata.

**Non-Goals:**

- Allowing users to edit `Prepared by` or the creation timestamp.
- Adding free-form report descriptions, recipients, approval workflows, or report versioning.
- Backfilling immutable historical author data that was not stored when an old report was created.
- Changing authentication, timezone preferences, or the report retention limit.

## Decisions

### 1. Resolve and validate the report title at the trusted frontend boundary

Step 3 of Forecast Setup will include an optional `report_title` text input with a 200-character limit. Browser setup-state persistence will retain the value across wizard navigation and page reloads. The Flask submission handler will trim leading/trailing whitespace and apply the canonical default `Forecast Report — <dataset stem>` (or `Forecast Report — data`) when the result is empty. It will reject a non-empty title longer than 200 characters with a field-specific HTTP 400 response.

The resolved value, rather than the raw optional input, will be sent as the existing backend `report_name`. Backend request validation will also enforce the 200-character bound for defense in depth.

This reuses the existing job contract and gives every accepted job a non-empty title. Keeping title resolution in Flask also avoids different defaults between JavaScript, the job queue, and saved-report persistence. An alternative was to resolve the default independently at each rendering surface; that was rejected because the current code already demonstrates how those names can diverge.

### 2. Treat authenticated job metadata as the source for author attribution

The browser will not submit a `prepared_by` field. Flask will continue attaching `current_user.username` as `application_username`, and the backend worker will pass the persisted job username into the report pipeline. `ReportMetadata` will gain a backwards-compatible `prepared_by` field, populated from that trusted application identity for new reports.

This keeps authorship tied to the account that submitted the forecast. Allowing an editable author field was rejected because the requirement defines the author as the username and client-controlled attribution would be misleading.

### 3. Extend structured report metadata and reuse `generated_at`

`ReportMetadata` will include `title` and `prepared_by`; its existing `generated_at` ISO-8601 UTC timestamp remains the canonical creation timestamp. The worker will pass the resolved job title and username through `run_pipeline`, the report stage, and `ExecutiveReportBuilder`. Renderers will label `generated_at` as the forecast creation date in user-facing output.

Using the existing generation timestamp avoids conflicting notions of creation time between job submission, report completion, and frontend persistence. The raw value remains UTC and presentation helpers format it as a human-readable value with an explicit UTC label. The frontend database `created_at` remains the saved-record timestamp and is only a legacy fallback when structured metadata is absent.

### 4. Make saved-report title persistence explicit and rename-aware

`save_report` will accept the resolved report title instead of always recalculating it from the source filename. Both automatic and manual job finalization paths will read `report_name` from the durable job response and pass it into persistence. Existing idempotency by `job_id` remains unchanged.

For saved reports, the `forecast_reports.title` column is authoritative for the displayed/exported title so the existing rename action continues to take effect. The structured metadata title is the creation-time title and supplies the value for an unsaved/current result. Renaming does not modify `prepared_by` or `generated_at`.

Adding another frontend database column for author or creation date was considered but rejected: those fields naturally belong to the already persisted `executive_report_json`, and `forecast_reports.created_at` already supports legacy fallback.

### 5. Use a shared presentation identity for web and PDF output

The report presentation helper will resolve a view model containing title, prepared-by username, and formatted creation date. The web report hero will replace its hard-coded title with this resolved identity. PDF generation will accept the same metadata, print it in a title block, use the resolved title for the document title, and derive a filesystem-safe download name without changing the stored display title.

Markdown and backend HTML metadata renderers will include the same fields so durable report artifacts remain self-describing. All values continue through existing escaping/sanitization paths.

### 6. Apply explicit fallbacks for historical records

New metadata model fields will have safe defaults so old serialized executive reports remain readable. When viewing an older saved report, presentation will prefer the saved `forecast_reports.title`, use the saved owner/current authenticated username as the author fallback when available, and use the saved-record `created_at` when `generated_at` is unavailable. If no reliable fallback exists, the field displays `Unknown` rather than inventing data.

This avoids a risky data rewrite while ensuring old reports do not fail template or PDF rendering.

## Risks / Trade-offs

- [A report can be renamed after its Markdown metadata was generated, leaving the embedded appendix title stale] → Treat the saved database title as authoritative in web/PDF presentation and test rename-followed-by-export behavior; creation-time structured metadata remains historical.
- [Usernames can later be renamed] → Persist the username value on the job/report at creation time so authorship reflects who prepared that artifact then.
- [Multiple timestamps already exist] → Document and label `generated_at` as the forecast creation timestamp; retain job and saved-record timestamps only for operational history and legacy fallback.
- [Long or unusual Unicode titles can produce unsafe filenames] → Preserve Unicode in displayed titles but generate a bounded, sanitized download basename with a deterministic fallback.
- [Existing work is in progress around job/report persistence] → Implement additively against the current `report_name`, `application_username`, and idempotent finalization paths, preserving their ownership checks.

## Migration Plan

1. Deploy additive model/request changes and pipeline propagation with defaults so old callers and stored JSON remain valid.
2. Deploy the frontend setup input, title resolution, persistence wiring, and shared web/PDF presentation.
3. Run schema initialization as usual; no frontend or backend database schema migration is required.
4. Verify a newly created custom-title report, a blank-title report, a renamed saved report, and a pre-change saved report.
5. Roll back by reverting application code; existing database rows and serialized results remain readable because no destructive schema/data migration is introduced.

## Open Questions

None. The design uses UTC for a deterministic creation timestamp; user-specific timezone display can be proposed separately if needed.
