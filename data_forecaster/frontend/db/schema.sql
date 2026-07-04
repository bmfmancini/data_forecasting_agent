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

CREATE TABLE IF NOT EXISTS api_credentials (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    label              TEXT    NOT NULL UNIQUE,
    encrypted_username TEXT,
    encrypted_password TEXT,
    base_url           TEXT    NOT NULL,
    timeout            INTEGER NOT NULL DEFAULT 30,
    created_at         TEXT    NOT NULL DEFAULT (datetime('now'))
);
