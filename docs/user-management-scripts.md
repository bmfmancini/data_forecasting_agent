# User Management Scripts

This project has two separate user systems, and each one has its own
management script:

- Frontend users log in to the Flask web application.
- Backend API users authenticate requests to the FastAPI backend.

The scripts are intentionally named `user_managment.py` to match the current
repository filenames.

Run these scripts from an environment that has the matching service
dependencies installed. For Docker deployments, the simplest path is usually
`docker compose exec frontend ...` or `docker compose exec backend ...`.

## When to use each script

| Need | Script |
|---|---|
| Add a person who can log in to the web UI | `data_forecaster/frontend/scripts/user_managment.py` |
| Reset a web user's password | `data_forecaster/frontend/scripts/user_managment.py` |
| Disable a web user's login immediately | `data_forecaster/frontend/scripts/user_managment.py` |
| Create a backend API credential for the frontend or another client | `data_forecaster/backend/scripts/user_managment.py` |
| Rotate a backend API key outside the admin UI | `data_forecaster/backend/scripts/user_managment.py` |
| Disable backend API access for a service/client | `data_forecaster/backend/scripts/user_managment.py` |

## Frontend User Script

The frontend script manages users in the Flask application database. These are
human users who sign in through the browser.

Run from the frontend directory:

```bash
cd data_forecaster/frontend
python scripts/user_managment.py --help
```

By default the script uses the configured frontend SQLite database. You can
override it with `--db-path` when working against a copied database or a test
fixture.

### Add a frontend user

```bash
python scripts/user_managment.py --add--user alice
```

The script prompts for a password interactively. New users are required to
change their password on next login by default.

Create an administrator:

```bash
python scripts/user_managment.py --add--user alice --admin
```

Provide a password non-interactively:

```bash
FRONTEND_USER_PASSWORD="temporary-password" \
  python scripts/user_managment.py --add--user alice
```

For bulk account creation, let the script generate a temporary password and
print it once:

```bash
python scripts/user_managment.py --add--user alice --generate-temp-password
```

Create a user without forcing a password change:

```bash
python scripts/user_managment.py --add--user alice --no-must-change-password
```

For non-interactive automation, prefer `FRONTEND_USER_PASSWORD`, stdin from a
secret manager, or `--generate-temp-password`. The frontend script does not
accept plaintext passwords as command-line arguments because those can be
exposed through shell history and process listings.

### Reset a frontend user's password

```bash
python scripts/user_managment.py --reset--password alice
```

Reset by database ID:

```bash
python scripts/user_managment.py --reset--password --id 3
```

Password reset increments the user's `session_version`, which invalidates
existing sessions.

To reset with a generated temporary password:

```bash
python scripts/user_managment.py --reset--password alice --generate-temp-password
```

### Disable a frontend user

```bash
python scripts/user_managment.py --disable--user alice
```

Disable by database ID:

```bash
python scripts/user_managment.py --disable--user --id 3
```

Disabling a user sets `active = 0` and increments `session_version`, which
forces existing sessions out.

### Delete a frontend user

```bash
python scripts/user_managment.py --delete--user alice --yes
```

Deletion requires `--yes` because deleting a frontend user also deletes that
user's reports through database cascade behavior.

## Backend API User Script

The backend script manages API users in the FastAPI backend database. These are
service/client identities used through request headers:

```text
X-API-Username: <username>
X-API-Key: <key>
```

Run from the backend directory:

```bash
cd data_forecaster/backend
python scripts/user_managment.py --help
```

By default the script uses the configured backend SQLite database. Override it
with `--db-path` when operating on a specific database file.

### Add a backend API user

```bash
python scripts/user_managment.py --add--user reporting-service \
  --description "Reporting integration"
```

Create an API administrator:

```bash
python scripts/user_managment.py --add--user api-admin --admin \
  --description "Emergency backend administration"
```

The script prints the plaintext API key once. Store it immediately; only the
Argon2id hash is saved in the database and the key cannot be recovered later.

### Reset a backend API key

```bash
python scripts/user_managment.py --reset--apikey reporting-service
```

Reset by database ID:

```bash
python scripts/user_managment.py --reset--apikey --id 3
```

Resetting an API key invalidates the previous key immediately. If the user is
the frontend service account, update Admin -> API Config or the matching
frontend environment/secret source before expecting frontend-to-backend calls
to succeed.

### Disable a backend API user

```bash
python scripts/user_managment.py --disable--user reporting-service
```

Disable by database ID:

```bash
python scripts/user_managment.py --disable--user --id 3
```

Disabled API users cannot authenticate, but their records remain available for
audit and later re-enablement through the application code/admin flows.

### Delete a backend API user

```bash
python scripts/user_managment.py --delete--user reporting-service
```

Do not delete or disable the API user currently configured in the frontend API
Config unless you have already configured and tested a replacement credential.

## Docker Notes

For a Compose deployment, the databases usually live in mounted volumes or
bind-mounted data directories. Prefer running the scripts in the relevant
container so paths and Python dependencies match the deployed service.

Frontend example:

```bash
cd data_forecaster
docker compose -f docker/docker-compose.yml exec frontend \
  python scripts/user_managment.py --reset--password admin
```

Backend example:

```bash
cd data_forecaster
docker compose -f docker/docker-compose.yml exec backend \
  python scripts/user_managment.py --reset--apikey frontend
```

If you run scripts from the host, confirm `--db-path` points at the same
database file used by the running container.

## Operational Guidance

- Use the frontend script for browser login accounts only.
- Use the backend script for API credentials only.
- Prefer disabling users before deletion when auditability matters.
- Store newly generated API keys immediately; they are shown once.
- After rotating the frontend service API key, verify Admin -> API Config can
  authenticate to the backend.
- Keep script output out of shared logs when it contains newly generated API
  keys or temporary passwords.
