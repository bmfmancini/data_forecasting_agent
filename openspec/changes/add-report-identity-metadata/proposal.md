## Why

Forecast reports currently lack a clear business identity at the point of creation: users cannot name a report before running it, and the finished report does not prominently identify its author or creation date. Adding these fields makes reports easier to recognize, share, and audit without requiring users to rename them afterward.

## What Changes

- Add an optional report title field to the Forecast Setup configuration step, with the existing dataset-based report name used when the field is blank.
- Carry the resolved title through forecast submission, job storage, report generation, saved-report persistence, the job queue, and report exports so one title is used consistently.
- Set `Prepared by` from the authenticated application username; it is system-controlled and not editable in Forecast Setup.
- Record the report/forecast creation timestamp when the report artifact is generated.
- Display the title, prepared-by username, and creation date together in the web report and PDF output, while preserving compatibility for reports created before these metadata fields exist.
- Validate and normalize user-entered titles, including whitespace-only values and the existing 200-character report-name limit.

## Capabilities

### New Capabilities

- `report-identity-metadata`: Defines report title selection and fallback behavior, authenticated author attribution, creation timestamps, persistence, and presentation across report surfaces.

### Modified Capabilities

None.

## Impact

- Forecast Setup template and browser-side setup state/submission payload.
- Frontend analysis request construction, completed-job finalization, report persistence, report rendering, and PDF export.
- Backend request/job contracts and the report-generation pipeline/model/renderers.
- Saved report, current report, user job queue, and exported PDF presentation.
- Automated tests for setup state, API validation, job metadata propagation, persistence, legacy fallback behavior, web rendering, and PDF output.
- No new third-party dependency or breaking API removal is expected.
