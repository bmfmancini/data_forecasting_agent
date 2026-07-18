## ADDED Requirements

### Requirement: Optional report title in Forecast Setup
The system SHALL provide an optional report-title field in the configuration step of Forecast Setup and SHALL retain its value with the other setup state until the forecast is submitted or the setup state is cleared.

#### Scenario: User enters a custom report title
- **WHEN** the user enters `Q4 report` in the report-title field and submits the forecast
- **THEN** the system submits the forecast with `Q4 report` as its resolved report title

#### Scenario: User navigates within the setup wizard
- **WHEN** the user enters a report title and moves backward and forward between setup steps or reloads the setup page
- **THEN** the report-title field retains the saved setup-state value

### Requirement: Canonical default report title
The system SHALL trim leading and trailing whitespace from the optional title and SHALL use `Forecast Report — <dataset stem>` as the canonical default when the trimmed value is empty, using `data` when no usable dataset stem exists.

#### Scenario: Title is omitted
- **WHEN** the user submits a forecast without entering a report title for `sales.csv`
- **THEN** the resolved report title is `Forecast Report — sales`

#### Scenario: Title contains only whitespace
- **WHEN** the user submits a title containing only whitespace
- **THEN** the system treats the title as omitted and applies the canonical default

### Requirement: Report title validation
The system SHALL accept resolved report titles from 1 through 200 characters and SHALL reject over-length custom titles before a forecast job is created.

#### Scenario: Custom title exceeds the limit
- **WHEN** the user submits a non-empty custom report title longer than 200 characters
- **THEN** the analysis endpoint returns a field-specific validation error and creates no forecast job

#### Scenario: Title includes display-safe Unicode
- **WHEN** the user submits a title of at most 200 characters containing Unicode text
- **THEN** the system preserves that title for report display while safely deriving any export filename

### Requirement: Authenticated prepared-by attribution
The system SHALL populate `Prepared by` for each newly generated report from the authenticated application username associated with the submitted forecast job, and SHALL NOT accept an editable author value from the browser.

#### Scenario: Authenticated user creates a report
- **WHEN** authenticated user `alice` submits and completes a forecast
- **THEN** the generated report records `alice` as `Prepared by`

#### Scenario: Browser payload attempts to supply an author
- **WHEN** a browser request includes an unrecognized or forged author value
- **THEN** the system ignores that value and uses the authenticated application username

### Requirement: Forecast creation timestamp
The system SHALL record the report generation time as an ISO-8601 UTC timestamp and SHALL present it as the forecast creation date with an explicit UTC indication.

#### Scenario: Report generation completes
- **WHEN** the report artifact is generated successfully
- **THEN** its structured metadata contains a non-empty UTC `generated_at` timestamp used as the displayed forecast creation date

#### Scenario: Report is viewed later
- **WHEN** a user opens or exports a saved report after its creation day
- **THEN** the displayed creation date remains the original report-generation timestamp rather than the current view or export time

### Requirement: Consistent report identity across lifecycle surfaces
The system SHALL carry the resolved report title, prepared-by username, and creation timestamp through durable job execution, report generation, saved-report persistence, retrieval, and presentation. The resolved title SHALL be used consistently in the user's job queue, saved-report list, web report, and PDF title block.

#### Scenario: Custom-title job completes
- **WHEN** a forecast submitted as `Q4 report` completes and is finalized
- **THEN** the job queue, saved-report record, web report, and PDF show `Q4 report` for that report

#### Scenario: Concurrent forecasts use different titles
- **WHEN** one user submits multiple forecasts with different titles before earlier jobs finish
- **THEN** each completed report uses the identity stored with its own durable job rather than setup state from another submission

#### Scenario: Saved report is renamed
- **WHEN** the owner renames a saved report after creation
- **THEN** the saved-report list, subsequent web views, and subsequent PDF exports use the renamed title while prepared-by and creation-date values remain unchanged

### Requirement: Report identity presentation
The system SHALL display the report title, `Prepared by` username, and forecast creation date together in the web report header and PDF title block, and SHALL escape or sanitize those values for their output context.

#### Scenario: Web report is displayed
- **WHEN** the user opens a completed report in the web application
- **THEN** the header visibly presents the resolved title, prepared-by username, and creation date without exposing executable markup from the title

#### Scenario: PDF report is exported
- **WHEN** the user downloads a report as PDF
- **THEN** the first-page title block presents the same title, prepared-by username, and creation date as the web report

### Requirement: Legacy report compatibility
The system SHALL continue to render and export reports saved before title, prepared-by, or structured creation metadata was introduced by applying non-destructive presentation fallbacks.

#### Scenario: Legacy saved report has no new structured identity fields
- **WHEN** the owner opens a legacy saved report
- **THEN** the system uses its saved report title, uses reliable owner and saved-date fallbacks when available, and completes rendering without a schema or template error

#### Scenario: Reliable legacy value is unavailable
- **WHEN** a legacy report lacks both a structured metadata value and a reliable persisted fallback
- **THEN** the presentation identifies the value as `Unknown` rather than failing or inventing attribution
