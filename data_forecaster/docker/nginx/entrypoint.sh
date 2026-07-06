#!/bin/sh
# Nginx entrypoint: generate self-signed certificate if missing, then
# substitute environment variables into the config templates and start
# nginx.
#
# Environment variables:
#   SSL_DOMAIN          - CN/SAN for the generated certificate (default: localhost)
#   FRONTEND_HTTPS_PORT - frontend HTTPS port (default: 443)
#   BACKEND_HTTPS_PORT  - backend HTTPS port (default: 8443)
#   HTTP_PORT           - frontend HTTP redirect port; if empty the redirect
#                         block is skipped (default: 80)
#   FRONTEND_UPSTREAM   - frontend upstream host:port (default: frontend:5000)
#   BACKEND_UPSTREAM    - backend upstream host:port (default: backend:8000)
#   NGINX_ROLE          - "frontend" or "backend"; selects which template to use
set -eu

CERT_DIR=/etc/nginx/certs
CERT_FILE="${CERT_DIR}/server.crt"
KEY_FILE="${CERT_DIR}/server.key"
CONF_DIR=/etc/nginx/conf.d

SSL_DOMAIN="${SSL_DOMAIN:-localhost}"
FRONTEND_HTTPS_PORT="${FRONTEND_HTTPS_PORT:-443}"
BACKEND_HTTPS_PORT="${BACKEND_HTTPS_PORT:-8443}"
HTTP_PORT="${HTTP_PORT:-80}"
FRONTEND_UPSTREAM="${FRONTEND_UPSTREAM:-frontend:5000}"
BACKEND_UPSTREAM="${BACKEND_UPSTREAM:-backend:8000}"
NGINX_ROLE="${NGINX_ROLE:-frontend}"

mkdir -p "${CERT_DIR}"

# ── Certificate handling ────────────────────────────────────────────────
if [ -s "${CERT_FILE}" ] && [ -s "${KEY_FILE}" ]; then
    echo "[entrypoint] Using existing certificate at ${CERT_FILE}"
else
    echo "[entrypoint] No certificate found; generating self-signed cert for CN=${SSL_DOMAIN}"
    # Build a SAN extension that covers the bare domain and a wildcard so
    # the cert is valid for the service name and localhost.
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "${KEY_FILE}" \
        -out "${CERT_FILE}" \
        -subj "/CN=${SSL_DOMAIN}" \
        -addext "subjectAltName=DNS:${SSL_DOMAIN},DNS:localhost,IP:127.0.0.1" \
        >/dev/null 2>&1
    chmod 600 "${KEY_FILE}"
    chmod 644 "${CERT_FILE}"
    echo "[entrypoint] Self-signed certificate generated."
fi

# ── Config templating ───────────────────────────────────────────────────
# Clear any default config so only our templates are active.
rm -f "${CONF_DIR}/default.conf"

# Compute a port suffix for the redirect URL.  When the HTTPS port is the
# default 443, browsers omit it; otherwise include ":<port>".
if [ "${FRONTEND_HTTPS_PORT}" = "443" ]; then
    FRONTEND_HTTPS_PORT_SUFFIX=""
else
    FRONTEND_HTTPS_PORT_SUFFIX=":${FRONTEND_HTTPS_PORT}"
fi
export FRONTEND_HTTPS_PORT_SUFFIX

if [ "${NGINX_ROLE}" = "frontend" ]; then
    envsubst \
        '${FRONTEND_HTTPS_PORT} ${FRONTEND_HTTPS_PORT_SUFFIX} ${FRONTEND_UPSTREAM}' \
        < /templates/frontend.conf.template \
        > "${CONF_DIR}/frontend.conf"

    if [ -n "${HTTP_PORT}" ]; then
        envsubst \
            '${HTTP_PORT} ${FRONTEND_HTTPS_PORT_SUFFIX}' \
            < /templates/frontend-http.conf.template \
            > "${CONF_DIR}/frontend-http.conf"
    fi
elif [ "${NGINX_ROLE}" = "backend" ]; then
    envsubst \
        '${BACKEND_HTTPS_PORT} ${BACKEND_UPSTREAM}' \
        < /templates/backend.conf.template \
        > "${CONF_DIR}/backend.conf"
else
    echo "[entrypoint] ERROR: NGINX_ROLE must be 'frontend' or 'backend', got '${NGINX_ROLE}'" >&2
    exit 1
fi

echo "[entrypoint] Starting nginx (role=${NGINX_ROLE})"
exec nginx -g 'daemon off;'