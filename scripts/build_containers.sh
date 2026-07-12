#!/usr/bin/env bash
# build_containers.sh — convenience wrapper around docker-compose for the
# Data Forecaster Agent.
#
# Usage:
#   ./scripts/build_containers.sh --single [--up] [--build] [--down] [--logs]
#   ./scripts/build_containers.sh --distributed [--role frontend|backend] [--up] [--build] [--down] [--logs]
#   ./scripts/build_containers.sh --help
#
# Modes
# -----
#   --single         Single-machine deployment.  Frontend, backend, and
#                    both Nginx proxies run on one host.  Uses
#                    docker/docker-compose.yml.
#
#   --distributed    Distributed deployment.  Frontend and backend run on
#                    separate machines.  You must run this script once on
#                    each machine and select the role with --role:
#                      --role backend   → runs backend + nginx-backend only
#                      --role frontend  → runs frontend + nginx-frontend only,
#                                        pointed at REMOTE_BACKEND_URL
#
# Actions (default: --up --build)
#   --up             Create and start containers (docker-compose up -d).
#   --build          (Re)build images before starting (passed to up).
#   --down           Stop and remove containers (docker-compose down).
#   --logs           Tail logs (docker-compose logs -f).
#   --no-build       Skip image build even with --up.
#
# Environment
# -----------
#   REMOTE_BACKEND_URL   Backend URL the frontend should use in distributed
#                        mode (e.g. https://api.example.com:8443).
#   API_VERIFY_SSL       Whether the frontend verifies the backend TLS cert
#                        (false for self-signed, true for trusted certs).
#   SSL_DOMAIN           CN/SAN for auto-generated self-signed certs.
#
# See README.md → Deployment for full details.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_DIR="${REPO_ROOT}/data_forecaster/docker"
BACKEND_ENV_FILE="${REPO_ROOT}/data_forecaster/backend/.env"
FRONTEND_ENV_FILE="${REPO_ROOT}/data_forecaster/frontend/.env"
LEGACY_ENV_FILE="${REPO_ROOT}/data_forecaster/.env"

# ── Defaults ────────────────────────────────────────────────────────────
MODE=""
ROLE="frontend"
ACTION=""
DO_BUILD=1

# ── Colour helpers (optional, degrade gracefully) ────────────────────────
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    RED=$(tput setaf 1)
    BOLD=$(tput bold)
    RESET=$(tput sgr0)
else
    GREEN="" YELLOW="" RED="" BOLD="" RESET=""
fi

log()  { echo "${GREEN}${BOLD}[build_containers]${RESET} $*"; }
warn() { echo "${YELLOW}${BOLD}[build_containers]${RESET} $*" >&2; }
err()  { echo "${RED}${BOLD}[build_containers]${RESET} $*" >&2; }

usage() {
    cat <<'EOF'
build_containers.sh — deploy the Data Forecaster Agent with Docker Compose.

Usage:
  ./scripts/build_containers.sh --single [ACTIONS]
  ./scripts/build_containers.sh --distributed --role frontend|backend [ACTIONS]

Modes:
  --single         Single-machine: all services on one host.
  --distributed    Multi-machine: run with --role backend on the backend
                   host and --role frontend on the frontend host.

Distributed role (only with --distributed):
  --role frontend  Run frontend + nginx-frontend, pointed at REMOTE_BACKEND_URL.
  --role backend   Run backend + nginx-backend only.

Actions (default: --up --build):
  --up             Start containers (detached).
  --build          (Re)build images before starting.
  --no-build       Do not build images.
  --down           Stop and remove containers.
  --logs           Tail container logs.
  --help, -h       Show this help.

Environment variables (set in data_forecaster/backend/.env,
data_forecaster/frontend/.env, or exported):
  REMOTE_BACKEND_URL  Backend URL for distributed frontend mode.
  API_VERIFY_SSL       Verify backend TLS cert (false for self-signed).
  SSL_DOMAIN           CN/SAN for auto-generated self-signed certs.

Examples:
  # Single-machine, build and start
  ./scripts/build_containers.sh --single

  # Single-machine, stop
  ./scripts/build_containers.sh --single --down

  # Distributed: backend host
  ./scripts/build_containers.sh --distributed --role backend

  # Distributed: frontend host (set REMOTE_BACKEND_URL in frontend/.env first)
  ./scripts/build_containers.sh --distributed --role frontend
EOF
}

load_env_file() {
    local file="$1"

    if [ ! -f "${file}" ]; then
        return 0
    fi

    # Export values so Docker Compose can use them for ${VAR} interpolation.
    set -a
    # shellcheck disable=SC1090
    . "${file}"
    set +a
}

warn_missing_env_file() {
    local file="$1"
    local example="$2"

    if [ ! -f "${file}" ]; then
        warn "Env file not found at ${file} — using compose and application defaults."
        warn "Create it with: cp ${example} ${file}"

        if [ -f "${file} " ]; then
            warn "Found '${file} ' with a trailing space in the filename; rename it to '${file}'."
        fi
    fi
}

# ── Argument parsing ────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --single)        MODE="single"; shift ;;
        --distributed)   MODE="distributed"; shift ;;
        --role)
            if [ $# -lt 2 ]; then
                err "Missing argument for --role. Use 'frontend' or 'backend'."
                exit 1
            fi
            ROLE="$2"; shift 2
            if [ "${ROLE}" != "frontend" ] && [ "${ROLE}" != "backend" ]; then
                err "Invalid --role '${ROLE}'. Use 'frontend' or 'backend'."
                exit 1
            fi
            ;;
        --up)     ACTION="up"; shift ;;
        --build)  DO_BUILD=1; shift ;;
        --no-build) DO_BUILD=0; shift ;;
        --down)   ACTION="down"; shift ;;
        --logs)   ACTION="logs"; shift ;;
        --help|-h) usage; exit 0 ;;
        *)
            err "Unknown argument: $1"
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "${MODE}" ]; then
    err "No mode specified. Use --single or --distributed."
    usage >&2
    exit 1
fi

if [ "${MODE}" = "distributed" ] && [ "${ROLE}" != "frontend" ] && [ "${ROLE}" != "backend" ]; then
    err "Distributed mode requires --role frontend or --role backend."
    exit 1
fi

# Default action: up with build.
if [ -z "${ACTION}" ]; then
    ACTION="up"
fi

# ── Pre-flight checks ───────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    err "docker is not installed or not on PATH."
    exit 1
fi
if ! docker compose version >/dev/null 2>&1 && ! command -v docker-compose >/dev/null 2>&1; then
    err "Neither 'docker compose' nor 'docker-compose' is available."
    exit 1
fi

# Use the compose v2 plugin if available, otherwise fall back to the
# standalone binary.
if docker compose version >/dev/null 2>&1; then
    DC=(docker compose)
else
    DC=(docker-compose)
fi

if [ ! -f "${COMPOSE_DIR}/docker-compose.yml" ]; then
    err "Compose file not found: ${COMPOSE_DIR}/docker-compose.yml"
    exit 1
fi

if [ -f "${LEGACY_ENV_FILE}" ]; then
    warn "Legacy env file found at ${LEGACY_ENV_FILE}; this script now uses backend/.env and frontend/.env."
fi

# ── Compose file selection ──────────────────────────────────────────────
COMPOSE_FILES=(-f "${COMPOSE_DIR}/docker-compose.yml")

case "${MODE}" in
    single)
        warn_missing_env_file "${BACKEND_ENV_FILE}" "${REPO_ROOT}/data_forecaster/backend/.env.example"
        warn_missing_env_file "${FRONTEND_ENV_FILE}" "${REPO_ROOT}/data_forecaster/frontend/.env.example"
        load_env_file "${BACKEND_ENV_FILE}"
        load_env_file "${FRONTEND_ENV_FILE}"
        log "Mode: ${BOLD}single-machine${RESET}"
        ;;
    distributed)
        case "${ROLE}" in
            backend)
                warn_missing_env_file "${BACKEND_ENV_FILE}" "${REPO_ROOT}/data_forecaster/backend/.env.example"
                load_env_file "${BACKEND_ENV_FILE}"
                if [ ! -f "${COMPOSE_DIR}/docker-compose.backend.yml" ]; then
                    err "Backend compose file not found: ${COMPOSE_DIR}/docker-compose.backend.yml"
                    err "Ensure the distributed compose files have been created."
                    exit 1
                fi
                COMPOSE_FILES=(-f "${COMPOSE_DIR}/docker-compose.backend.yml")
                log "Mode: ${BOLD}distributed${RESET} (role: ${BOLD}backend${RESET})"
                ;;
            frontend)
                if [ ! -f "${COMPOSE_DIR}/docker-compose.distributed.yml" ]; then
                    err "Distributed override not found: ${COMPOSE_DIR}/docker-compose.distributed.yml"
                    err "Ensure the distributed compose files have been created."
                    exit 1
                fi
                COMPOSE_FILES+=(-f "${COMPOSE_DIR}/docker-compose.distributed.yml")
                warn_missing_env_file "${FRONTEND_ENV_FILE}" "${REPO_ROOT}/data_forecaster/frontend/.env.example"
                load_env_file "${FRONTEND_ENV_FILE}"
                log "Mode: ${BOLD}distributed${RESET} (role: ${BOLD}frontend${RESET})"

                if [ -z "${REMOTE_BACKEND_URL:-}" ]; then
                    # Try to read it from frontend/.env if not exported or loaded.
                    if [ -f "${FRONTEND_ENV_FILE}" ]; then
                        REMOTE_BACKEND_URL="$(grep -E '^REMOTE_BACKEND_URL=' "${FRONTEND_ENV_FILE}" 2>/dev/null | cut -d= -f2- || true)"
                    fi
                fi
                if [ -z "${REMOTE_BACKEND_URL:-}" ]; then
                    err "REMOTE_BACKEND_URL is not set. The frontend will not be able to reach the backend."
                    err "Set it in ${FRONTEND_ENV_FILE} or export it before running this script."
                    exit 1
                else
                    log "Remote backend URL: ${REMOTE_BACKEND_URL}"
                fi
                ;;
        esac
        ;;
esac

# ── Run the requested action ────────────────────────────────────────────
cd "${COMPOSE_DIR}"

case "${ACTION}" in
    up)
        BUILD_FLAG=""
        if [ "${DO_BUILD}" -eq 1 ]; then
            BUILD_FLAG="--build"
        fi
        log "Starting containers (detached)${BUILD_FLAG:+ with build}..."
        "${DC[@]}" "${COMPOSE_FILES[@]}" up -d ${BUILD_FLAG}
        log "Containers started. Use '${0} --${MODE} --logs' to tail logs."
        ;;
    down)
        log "Stopping and removing containers..."
        "${DC[@]}" "${COMPOSE_FILES[@]}" down
        log "Containers stopped."
        ;;
    logs)
        log "Tailing logs (Ctrl-C to stop)..."
        "${DC[@]}" "${COMPOSE_FILES[@]}" logs -f
        ;;
esac
