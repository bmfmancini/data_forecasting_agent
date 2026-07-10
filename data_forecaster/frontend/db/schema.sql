CREATE TABLE IF NOT EXISTS roles (
    id   INTEGER PRIMARY KEY,
    name TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS users (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    username             TEXT    NOT NULL UNIQUE,
    password_hash        TEXT    NOT NULL,
    role_id              INTEGER NOT NULL DEFAULT 2,
    active               INTEGER NOT NULL DEFAULT 1,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (role_id) REFERENCES roles (id)
);

CREATE TABLE IF NOT EXISTS app_config (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS forecast_reports (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id               INTEGER NOT NULL,
    title                 TEXT    NOT NULL,
    source_filename       TEXT    NOT NULL,
    model_used            TEXT,
    forecast_horizon      INTEGER,
    report_markdown       TEXT    NOT NULL,
    executive_report_json TEXT,
    visual_assets_json    TEXT    NOT NULL,
    llm_fallback          INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS forecast_reports_user_created_idx
ON forecast_reports(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS api_credentials (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    label              TEXT    NOT NULL UNIQUE,
    encrypted_username TEXT,
    encrypted_password TEXT,
    base_url           TEXT    NOT NULL,
    timeout            INTEGER NOT NULL DEFAULT 30,
    verify_ssl         INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT    NOT NULL DEFAULT (datetime('now'))
);
