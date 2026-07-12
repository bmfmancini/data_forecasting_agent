# Deployment Guide

## Architecture

Both the frontend and backend sit behind their own Nginx container that handles TLS. Only the Nginx containers are exposed externally — the app containers talk to each other on the internal Docker network.

```
                    End users
                       |
                  HTTPS :443
                       |
              +------------------+
              | nginx-frontend   |  TLS termination
              +------------------+
                       |
                  HTTP :5000
                       |
              +------------------+
              | Flask frontend   |  web UI, auth, admin
              +------------------+
                       |
            (HTTPS to backend URL)
                       |
              +------------------+
              | nginx-backend    |  TLS termination
              +------------------+
                       |
                  HTTP :8000
                       |
              +------------------+
              | FastAPI backend  |  forecasting, agents, LLM
              +------------------+
```

## Single-machine deployment

Everything runs on one host. This is the default and what most people want.

```bash
./scripts/build_containers.sh --single
```

That uses `docker/docker-compose.yml` and brings up all four containers. The frontend talks to the backend at `https://nginx-backend:8443` over the internal Docker network.

Verify it's up:

```bash
curl -k https://localhost/auth/login        # should return 200
curl -k https://localhost:8443/health       # should return {"status":"ok"}
```

## Distributed deployment

If you want the frontend on one machine and the backend on another, you run each side separately.

### On the backend machine

```bash
./scripts/build_containers.sh --distributed --role backend
```

This uses `docker/docker-compose.backend.yml` and brings up just the backend and its Nginx proxy. The backend Nginx listens on port 8443.

### On the frontend machine

```bash
./scripts/build_containers.sh --distributed --role frontend
```

This uses `docker/docker-compose.distributed.yml` as an override and drops
the backend and nginx-backend services. After the frontend starts, log in and
configure the backend URL, SSL verification setting, username, and key under
**Admin → API Config**.

## TLS certificates

### Self-signed (default)

If no certificate exists, the Nginx entrypoint generates one automatically using `openssl req -x509`. The cert is valid for 365 days and includes SANs for `SSL_DOMAIN`, `localhost`, and `127.0.0.1`.

Certificates are stored in:
- `data_forecaster/certs/frontend/server.crt` and `server.key`
- `data_forecaster/certs/backend/server.crt` and `server.key`

Existing certificates are never overwritten. If you want a fresh one, delete the files and restart.

### Bring your own

Drop your `server.crt` and `server.key` into the `certs/frontend/` or `certs/backend/` directories before starting the containers. The entrypoint will detect them and skip generation.

### Changing the domain

Set `SSL_DOMAIN` in your shell or Compose project `.env` to match your hostname:

```bash
SSL_DOMAIN=forecaster.example.com
```

This becomes the CN and primary SAN on the generated certificate.

## SSL verification

The frontend talks to the backend over HTTPS. Configure whether the frontend
verifies the backend certificate under **Admin → API Config**.

Turn verification off for the default self-signed certificate, and turn it on
when the backend uses a CA-signed certificate.

## Environment variables

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `SSL_DOMAIN` | nginx | `localhost` | CN/SAN for self-signed certs |
| `HTTP_PORT` | nginx-frontend | `80` | HTTP port (redirects to HTTPS) |
| `FRONTEND_HTTPS_PORT` | nginx-frontend | `443` | Frontend HTTPS port |
| `BACKEND_HTTPS_PORT` | nginx-backend | `8443` | Backend HTTPS port |
| `CORS_ALLOWED_ORIGINS` | backend | `http://localhost:5000,...` | Comma-separated allowed origins |
| `FRONTEND_API_USERNAME` | backend | `frontend` | Optional backend bootstrap service-account username |
| `FRONTEND_API_KEY` | backend | `frontend` | Optional backend bootstrap service-account key |
| `SECRET_KEY` | frontend | (random) | Flask session secret |
| `FLASK_ENCRYPTION_KEY` | frontend | (random) | Fernet key for encrypting stored API credentials |

The frontend stores the active backend URL, SSL verification setting,
username, and encrypted key in its SQLite database via **Admin → API Config**.

## The build script

`scripts/build_containers.sh` is a wrapper around docker-compose so you don't have to remember the file paths:

```bash
# Single-machine
./scripts/build_containers.sh --single

# Distributed — frontend host
./scripts/build_containers.sh --distributed --role frontend

# Distributed — backend host
./scripts/build_containers.sh --distributed --role backend

# Other actions
./scripts/build_containers.sh --single --down      # stop everything
./scripts/build_containers.sh --single --logs       # tail logs
./scripts/build_containers.sh --single --no-build   # up without rebuilding
```

## Troubleshooting

**`SSL_ERROR_RX_RECORD_TOO_LONG`** — The Nginx config is missing the `ssl` keyword on the `listen` directive. This was a bug in an earlier version; make sure you're using the current `docker/nginx/conf.d/*.conf.template` files.

**Frontend can't reach backend (401)** — Check that the API credentials match. If you rotated the key on the backend, update it in **Admin → API Config** on the frontend.

**Frontend can't reach backend (SSL error)** — Disable SSL verification under **Admin → API Config** when using the default self-signed backend certificate.

**Backend healthcheck fails on startup** — The backend takes ~25 seconds to initialize (ChromaDB + LLM setup). The healthcheck has a 40-second start period, but if your machine is slow, increase `start_period` in `docker-compose.yml`.

**Stale database after changing backend bootstrap credentials** — The backend DB lives in a Docker volume and a bind mount. If you change `FRONTEND_API_KEY` in `backend/.env` and the old user still exists, the backend won't recreate it. Rotate or create the backend API user from **Admin → API Keys**, then update **Admin → API Config**. For a full reset:

```bash
docker compose down -v
rm -f data_forecaster/data/backend.db
docker compose up -d --build
```
