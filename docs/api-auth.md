# API Authentication

## How it works

The FastAPI backend requires an API key for all protected endpoints. Keys are hashed with Argon2id and stored in a SQLite database (`api_keys.db`). Plaintext keys are never stored — they're only shown once at creation or rotation time.

Every request to a protected endpoint needs two headers:

```
X-API-Username: frontend
X-API-Key: <your key>
```

On success, the backend updates `last_used` and `last_used_ip` on the user record. On failure, you get a generic `401 Unauthorized` — the error doesn't tell you whether the username or the key was wrong, so you can't probe for valid usernames.

## Default credentials

The stack ships with a default service account:

- **Username:** `frontend`
- **Key:** `frontend`

Both the backend and frontend auto-seed these on first boot, so everything works out of the box without any manual configuration.

> **This key is publicly known.** Anyone reading the source code or this doc knows it. Rotate it before exposing the stack to anything beyond your local machine. The backend logs a `SECURITY` warning on startup when the default key is in use.

## Rotating the default key

1. Log into the admin panel at `https://localhost` (`admin` / `admin`).
2. Go to **Admin → API Keys**. You'll see the `frontend` user with a ⚠ badge (it's flagged as a bootstrap account).
3. Click 🔄 (rotate) on the `frontend` user. Copy the new key — it's only shown once.
4. Go to **Admin → API Config** and update the **API Key** field with the new key. Save.
5. The ⚠ badge stays until you either rotate the key or delete the bootstrap user and create a new one.

Alternatively, you can create a brand new API user, update the API Config with those credentials, and delete the `frontend` user entirely.

## Creating additional API users

1. Go to **Admin → API Keys** → **+ Create API User**.
2. Enter a username and description.
3. The plaintext key is displayed once — copy it immediately.
4. Use those credentials in any client that talks to the backend API.

## Disabling / enabling users

From **Admin → API Keys**, you can toggle the `enabled` flag on any user. Disabled users can't authenticate. This is useful for revoking access without deleting the account.

## How the frontend stores credentials

The Flask frontend stores the API username and key (encrypted with Fernet) in its own SQLite database (`instance/forecaster.db`). On every request to the backend, the `BackendAPIClient` decrypts them and sends the headers. The plaintext key only exists in memory for the duration of a single request.

The admin panel lets you update these credentials under **Admin → API Config**. Changes take effect immediately — no restart needed.

## Disabling auth entirely (dev only)

Set `API_KEY_ENABLED=false` in the backend `.env` to make the auth dependency a no-op. All protected endpoints become open. This is useful for local development or testing — **never do this in production**.

## If a key is compromised

1. **Admin → API Keys** → 🔄 (rotate) on the affected user.
2. Copy the new key.
3. **Admin → API Config** → update the stored key.
4. The old key is dead immediately. Any client still using it gets `401`.
