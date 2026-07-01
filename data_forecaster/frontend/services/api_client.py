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
        base_url: Root URL of the backend service, e.g. ``http://localhost:8000``.
        auth:     Optional ``(username, password)`` tuple for HTTP Basic Auth.
                  Pass ``None`` when the backend does not require authentication.
    """

    def __init__(
        self,
        base_url: str,
        auth: tuple[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = auth

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
            auth=self._auth,
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
            auth=self._auth,
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
            auth=self._auth,
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
            auth=self._auth,
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
            auth=self._auth,
            timeout=CHAT_TIMEOUT,
        )

    def health_check(self) -> requests.Response:
        """Perform a lightweight connectivity check against the backend.

        Returns:
            The :class:`requests.Response` from ``GET /health``.
        """
        return requests.get(
            f"{self._base_url}/health",
            auth=self._auth,
            timeout=5,
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
    auth: tuple[str, str] | None = None

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

                auth = (decrypt(str(enc_user)), decrypt(str(enc_pass)))
            except Exception:
                auth = None

    return BackendAPIClient(base_url=base_url, auth=auth)
