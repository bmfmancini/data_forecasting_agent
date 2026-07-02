"""
Backend API client for the Flask forecaster frontend.

Provides :class:`BackendAPIClient` which wraps every call to the FastAPI
forecasting backend.  The factory function :func:`get_api_client` resolves
the active backend URL and credentials from the database (with environment
variable fallback) so all callers remain decoupled from configuration details.

Timeout constants mirror those used by the original Streamlit frontend.
"""

from __future__ import annotations

import base64
from typing import Any

import requests
from flask import current_app

UPLOAD_TIMEOUT: int = 60
PREFLIGHT_TIMEOUT: int = 15
ANALYSIS_TIMEOUT: int = 30
JOB_STATUS_TIMEOUT: int = 10
CHAT_TIMEOUT: int = 60


class BackendAPIClient:
    """HTTP client for the FastAPI forecasting backend.

    Args:
        base_url:      Root URL of the backend service, e.g. ``http://localhost:8000``.
        api_username:  Optional username for the ``X-API-Username`` header.
        api_key:       Optional API key for the ``X-API-Key`` header.
                       Pass ``None`` when the backend does not require authentication.
    """

    def __init__(
        self,
        base_url: str,
        api_username: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_username = api_username
        self._api_key = api_key

    def _auth_headers(self) -> dict[str, str]:
        """Return authentication headers when credentials are configured.

        Returns:
            A dict with ``X-API-Username`` and ``X-API-Key`` keys, or an
            empty dict when no credentials are set.
        """
        if self._api_username and self._api_key:
            return {
                "X-API-Username": self._api_username,
                "X-API-Key": self._api_key,
            }
        return {}

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Merge auth headers with any extra headers.

        Args:
            extra: Optional additional headers to merge.

        Returns:
            Combined headers dict.
        """
        headers: dict[str, str] = self._auth_headers()
        if extra:
            headers.update(extra)
        return headers

    def upload_file(
        self,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> requests.Response:
        """Upload a CSV or XLSX file to the backend.

        Args:
            filename:     Original file name including extension.
            content:      Raw file bytes.
            content_type: MIME type string (e.g. ``text/csv``).

        Returns:
            The :class:`requests.Response` from ``POST /upload``.
        """
        return requests.post(
            f"{self._base_url}/upload",
            files={"file": (filename, content, content_type)},
            headers=self._headers(),
            timeout=UPLOAD_TIMEOUT,
        )

    def get_preflight(
        self,
        file_id: str,
        forecast_horizon: int,
        date_col: str,
        value_col: str,
    ) -> requests.Response:
        """Run preflight quality checks for the specified upload and columns.

        Args:
            file_id:          Identifier returned by the upload endpoint.
            forecast_horizon: Number of future periods to forecast.
            date_col:         Name of the date column in the uploaded file.
            value_col:        Name of the value column in the uploaded file.

        Returns:
            The :class:`requests.Response` from ``POST /preflight``.
        """
        return requests.post(
            f"{self._base_url}/preflight",
            json={
                "file_id": file_id,
                "forecast_horizon": forecast_horizon,
                "date_col": date_col,
                "value_col": value_col,
            },
            headers=self._headers(),
            timeout=PREFLIGHT_TIMEOUT,
        )

    def submit_analysis(self, payload: dict[str, Any]) -> requests.Response:
        """Submit an asynchronous analysis job.

        Args:
            payload: Dict conforming to the ``AnalyzeRequest`` schema:
                ``file_id``, ``forecast_horizon``, ``date_col``,
                ``value_col``, and optionally ``forced_model``,
                ``user_prompt``, ``preflight_options``.

        Returns:
            The :class:`requests.Response` from ``POST /analyze`` (HTTP 202).
        """
        return requests.post(
            f"{self._base_url}/analyze",
            json=payload,
            headers=self._headers(),
            timeout=ANALYSIS_TIMEOUT,
        )

    def get_job_status(self, job_id: str) -> requests.Response:
        """Poll the status of a previously submitted analysis job.

        Args:
            job_id: Identifier returned by :meth:`submit_analysis`.

        Returns:
            The :class:`requests.Response` from ``GET /jobs/{job_id}``.
        """
        return requests.get(
            f"{self._base_url}/jobs/{job_id}",
            headers=self._headers(),
            timeout=JOB_STATUS_TIMEOUT,
        )

    def send_chat(
        self,
        file_id: str | None,
        query: str,
    ) -> requests.Response:
        """Send a chat query to the backend.

        Args:
            file_id: Optional upload identifier to scope the query to a
                     specific dataset.  Pass ``None`` for general questions.
            query:   The user's natural language query.

        Returns:
            The :class:`requests.Response` from ``POST /chat``.
        """
        payload: dict[str, Any] = {"query": query}
        if file_id:
            payload["file_id"] = file_id
        return requests.post(
            f"{self._base_url}/chat",
            json=payload,
            headers=self._headers(),
            timeout=CHAT_TIMEOUT,
        )

    def health_check(self) -> requests.Response:
        """Perform a lightweight connectivity check against the backend.

        Returns:
            The :class:`requests.Response` from ``GET /health``.
        """
        return requests.get(
            f"{self._base_url}/health",
            headers=self._headers(),
            timeout=5,
        )

    # ── API User Management ───────────────────────────────────────────────

    def list_api_users(self) -> requests.Response:
        """List all API key users from the backend.

        Returns:
            The :class:`requests.Response` from ``GET /api-users``.
        """
        return requests.get(
            f"{self._base_url}/api-users",
            headers=self._headers(),
            timeout=JOB_STATUS_TIMEOUT,
        )

    def create_api_user(self, username: str, description: str) -> requests.Response:
        """Create a new API user on the backend.

        Args:
            username:    Unique username for the new API user.
            description: Human-readable description.

        Returns:
            The :class:`requests.Response` from ``POST /api-users``.
        """
        return requests.post(
            f"{self._base_url}/api-users",
            json={"username": username, "description": description},
            headers=self._headers(),
            timeout=ANALYSIS_TIMEOUT,
        )

    def rotate_api_key(self, user_id: int) -> requests.Response:
        """Rotate an API user's key on the backend.

        Args:
            user_id: Primary key of the API user.

        Returns:
            The :class:`requests.Response` from ``POST /api-users/{id}/rotate``.
        """
        return requests.post(
            f"{self._base_url}/api-users/{user_id}/rotate",
            headers=self._headers(),
            timeout=ANALYSIS_TIMEOUT,
        )

    def toggle_api_user(self, user_id: int, enabled: bool) -> requests.Response:
        """Enable or disable an API user on the backend.

        Args:
            user_id: Primary key of the API user.
            enabled: ``True`` to enable, ``False`` to disable.

        Returns:
            The :class:`requests.Response` from ``POST /api-users/{id}/toggle``.
        """
        return requests.post(
            f"{self._base_url}/api-users/{user_id}/toggle",
            json={"enabled": enabled},
            headers=self._headers(),
            timeout=ANALYSIS_TIMEOUT,
        )

    def delete_api_user(self, user_id: int) -> requests.Response:
        """Delete an API user on the backend.

        Args:
            user_id: Primary key of the API user to delete.

        Returns:
            The :class:`requests.Response` from ``DELETE /api-users/{id}``.
        """
        return requests.delete(
            f"{self._base_url}/api-users/{user_id}",
            headers=self._headers(),
            timeout=ANALYSIS_TIMEOUT,
        )

    def bootstrap_status(self) -> requests.Response:
        """Check whether a bootstrap API user still exists.

        Returns:
            The :class:`requests.Response` from
            ``GET /api-users/bootstrap-status``.
        """
        return requests.get(
            f"{self._base_url}/api-users/bootstrap-status",
            headers=self._headers(),
            timeout=JOB_STATUS_TIMEOUT,
        )


def get_api_client() -> BackendAPIClient:
    """Construct a :class:`BackendAPIClient` for the current request.

    The backend URL and optional credentials are resolved from the
    ``api_credentials`` table (label ``'default'``).  If no credentials are
    stored the client is returned without authentication headers, preserving
    backward compatibility with an unauthenticated backend (Phase 1).

    Returns:
        A configured :class:`BackendAPIClient` instance.
    """
    from db.db import query_db

    base_url: str = current_app.config.get("BACKEND_URL", "http://localhost:8000")
    api_username: str | None = None
    api_key: str | None = None

    row = query_db(
        """
        SELECT base_url, encrypted_username, encrypted_password
        FROM api_credentials
        WHERE label = 'default'
        LIMIT 1
        """,
        one=True,
    )

    if row and isinstance(row, dict):
        # Only use the stored URL if the app config is still the bare default,
        # so that BACKEND_URL env var always wins over stale DB values.
        stored_url = row.get("base_url", "")
        if stored_url and base_url == "http://localhost:8000":
            base_url = str(stored_url)

        enc_user = row.get("encrypted_username")
        enc_pass = row.get("encrypted_password")
        if enc_user and enc_pass:
            try:
                from db.crypto import decrypt

                api_username = decrypt(str(enc_user))
                api_key = decrypt(str(enc_pass))
            except Exception:
                api_username = None
                api_key = None

    return BackendAPIClient(
        base_url=base_url,
        api_username=api_username,
        api_key=api_key,
    )
