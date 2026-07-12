# API Authentication

## How it works

The FastAPI backend requires an API key for all protected endpoints. Keys are hashed with Argon2id and stored in the backend SQLite database (`backend.db`). Plaintext keys are never stored — they're only shown once at creation or rotation time.

Every request to a protected endpoint needs two headers:

```
X-API-Username: frontend
X-API-Key: <your key>
```

On success, the backend updates `last_used` and `last_used_ip` on the user record. On failure, you get a generic `401 Unauthorized` — the error doesn't tell you whether the username or the key was wrong, so you can't probe for valid usernames.

## First setup

The frontend does not read backend API credentials from its `.env` file.
After first login, configure the active backend connection in
**Admin → API Config**.

For a new deployment:

1. Log into the frontend admin panel with `admin` / `admin`.
2. Change the password when prompted.
3. Create or rotate a backend API user under **Admin → API Keys**.
4. Copy the plaintext key shown once.
5. Enter the backend URL, SSL verification setting, username, and key under
   **Admin → API Config**.

The frontend stores the API username and key in its own SQLite database. The
key is encrypted at rest and is never displayed back to the browser.

## Rotating a key

1. Go to **Admin → API Keys**.
2. Click 🔄 (rotate) on the affected user.
3. Copy the new key — it's only shown once.
4. Go to **Admin → API Config** and update the **API Key** field with the new key. Save.

Alternatively, create a brand new API user, update API Config with those
credentials, and delete or disable the old user.

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
