"""Validate candidate LLM settings without persisting them.

Validation is deliberately staged so the UI can tell an administrator whether
the provider host is reachable, the credentials are accepted, and the selected
model can produce a real response.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote

import httpx

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
_PING_PROMPT = "Connection test. Reply with exactly: pong"
_RESPONSE_PREVIEW_LIMIT = 500
_DIAGNOSTIC_PREVIEW_LIMIT = 500
_PING_MAX_OUTPUT_TOKENS = 64


@dataclass(frozen=True)
class LLMValidationResult:
    """The outcome of each LLM validation stage."""

    ok: bool = False
    url_reachable: bool = False
    credentials_valid: bool = False
    llm_responded: bool = False
    message: str = ""
    response: str | None = None
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def _failed(
    message: str,
    *,
    url_reachable: bool = False,
    credentials_valid: bool = False,
    diagnostic: str | None = None,
) -> LLMValidationResult:
    """Build a failed validation result with a safe provider diagnostic."""
    return LLMValidationResult(
        url_reachable=url_reachable,
        credentials_valid=credentials_valid,
        message=message,
        diagnostic=diagnostic,
    )


def _provider_error_diagnostic(response: httpx.Response) -> str:
    """Return the useful, non-sensitive portion of a provider error response.

    Providers use a few common shapes for errors.  Only known diagnostic
    fields are returned rather than echoing the complete body, which could
    include request data or credentials on a non-conforming provider.
    """
    detail = ""
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            for field in ("message", "detail", "status", "type", "code"):
                value = error.get(field)
                if isinstance(value, (str, int, float)) and str(value).strip():
                    detail = str(value).strip()
                    break
        elif isinstance(error, (str, int, float)) and str(error).strip():
            detail = str(error).strip()
        if not detail:
            for field in ("message", "detail"):
                value = payload.get(field)
                if isinstance(value, (str, int, float)) and str(value).strip():
                    detail = str(value).strip()
                    break

    status = f"HTTP {response.status_code}"
    if detail:
        return f"{status}: {detail[:_DIAGNOSTIC_PREVIEW_LIMIT]}"
    return status


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    """Extract text from a Gemini ``generateContent`` response."""
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    if not isinstance(parts, list):
        return ""
    return "".join(
        str(part.get("text", "")) for part in parts if isinstance(part, dict)
    ).strip()


def _extract_ollama_text(payload: dict[str, Any]) -> str:
    """Extract text from an Ollama chat response."""
    message = payload.get("message")
    if not isinstance(message, dict):
        return ""
    return str(message.get("content", "")).strip()


async def validate_llm_configuration(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
) -> LLMValidationResult:
    """Run reachability, authentication, and generation checks in order.

    The candidate settings are used in-memory only.  Failure results include a
    bounded diagnostic drawn only from standard provider error fields.
    """
    if provider not in {"gemini", "ollama", "ollama_cloud"}:
        return _failed("The selected LLM provider is not supported.")
    if not model.strip():
        return _failed("Enter a model name before testing the LLM.")

    provider_url = (
        _GEMINI_BASE_URL
        if provider == "gemini"
        else str(base_url or "").strip().rstrip("/")
    )
    if not provider_url:
        return _failed("Enter a base URL before testing the LLM.")
    if provider in {"gemini", "ollama_cloud"} and not api_key:
        return _failed("Enter an API key before testing the LLM.")

    headers = {"Content-Type": "application/json"}
    if provider in {"ollama", "ollama_cloud"} and api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = httpx.Timeout(30.0, connect=5.0)
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True
    ) as client:
        try:
            # Any HTTP response proves that the configured host is reachable.
            await client.get(provider_url)
        except httpx.RequestError:
            return _failed("The LLM URL could not be reached.")

        try:
            if provider == "gemini":
                credential_response = await client.get(
                    f"{provider_url}/v1beta/models",
                    headers={"x-goog-api-key": str(api_key)},
                )
            else:
                credential_response = await client.get(
                    f"{provider_url}/api/tags", headers=headers
                )
        except httpx.RequestError:
            return _failed(
                "The LLM URL became unavailable while checking credentials.",
                url_reachable=True,
            )

        if not credential_response.is_success:
            return _failed(
                "The LLM URL or API key was rejected. Check both values.",
                url_reachable=True,
                diagnostic=_provider_error_diagnostic(credential_response),
            )

        safe_model = quote(model.strip(), safe="")
        try:
            if provider == "gemini":
                ping_response = await client.post(
                    f"{provider_url}/v1beta/models/{safe_model}:generateContent",
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": str(api_key),
                    },
                    json={
                        "contents": [
                            {"parts": [{"text": _PING_PROMPT}]}
                        ],
                        "generationConfig": {
                            "maxOutputTokens": _PING_MAX_OUTPUT_TOKENS
                        },
                    },
                )
            else:
                ping_response = await client.post(
                    f"{provider_url}/api/chat",
                    headers=headers,
                    json={
                        "model": model.strip(),
                        "messages": [
                            {"role": "user", "content": _PING_PROMPT}
                        ],
                        "stream": False,
                        # Reasoning models such as gpt-oss can consume a very
                        # small generation budget entirely in hidden thinking,
                        # leaving ``message.content`` empty.  A connection
                        # probe needs a visible response, so turn thinking off.
                        "think": False,
                        "options": {"num_predict": _PING_MAX_OUTPUT_TOKENS},
                    },
                )
        except httpx.RequestError:
            return _failed(
                "Credentials were accepted, but the LLM did not respond to the ping.",
                url_reachable=True,
                credentials_valid=True,
            )

        if not ping_response.is_success:
            # A 403 can mean that the otherwise valid credential is not allowed
            # to use this particular model.  Only 401 conclusively invalidates
            # the credential-validation stage.
            credentials_rejected = ping_response.status_code == 401
            return _failed(
                (
                    "The LLM rejected the credentials while prompting the selected "
                    "model."
                    if credentials_rejected
                    else "Credentials were accepted, but the selected model rejected "
                    "the ping."
                ),
                url_reachable=True,
                credentials_valid=not credentials_rejected,
                diagnostic=_provider_error_diagnostic(ping_response),
            )

        try:
            ping_payload: dict[str, Any] = ping_response.json()
        except ValueError:
            return _failed(
                "The LLM returned an invalid response to the ping.",
                url_reachable=True,
                credentials_valid=True,
            )
        reply = (
            _extract_gemini_text(ping_payload)
            if provider == "gemini"
            else _extract_ollama_text(ping_payload)
        )
        if not reply:
            return _failed(
                "The LLM returned an empty response to the ping.",
                url_reachable=True,
                credentials_valid=True,
            )

    return LLMValidationResult(
        ok=True,
        url_reachable=True,
        credentials_valid=True,
        llm_responded=True,
        message="LLM connection test passed.",
        response=reply[:_RESPONSE_PREVIEW_LIMIT],
    )
