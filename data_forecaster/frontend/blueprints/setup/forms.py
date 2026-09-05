"""
WTForms form definitions for the first-run setup wizard blueprint.
"""

from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    FloatField,
    PasswordField,
    RadioField,
    StringField,
    SubmitField,
)
from wtforms.validators import URL, DataRequired, Length, NumberRange, Optional

LLM_PROVIDER_CHOICES: list[tuple[str, str]] = [
    ("gemini", "Google Gemini"),
    ("ollama", "Ollama (local)"),
    ("ollama_cloud", "Ollama Cloud"),
]


class BackendConnectionForm(FlaskForm):  # type: ignore[misc]
    """Step 1 — backend connection settings.

    Fields:
        base_url:   Root URL of the FastAPI backend.
        verify_ssl: Whether to verify the backend TLS certificate.
        submit:     Submission button.
    """

    base_url = StringField(
        "Backend API Base URL",
        validators=[DataRequired(), URL(require_tld=False)],
    )
    verify_ssl = BooleanField(
        "Verify backend SSL certificate",
        default=False,
    )
    submit = SubmitField("Test & Continue")


class LLMProviderForm(FlaskForm):  # type: ignore[misc]
    """Step 2 — LLM provider configuration.

    Fields:
        provider:    LLM provider selection.
        model:       Model name served by the provider.
        base_url:    Provider base URL (Ollama providers only).
        api_key:     Provider API key (optional; forwarded to the backend
                     only — never persisted by the frontend).
        temperature: Sampling temperature.
        submit:      Submission button.
    """

    provider = RadioField(
        "Provider",
        choices=LLM_PROVIDER_CHOICES,
        default="gemini",
        validators=[DataRequired()],
    )
    model = StringField(
        "Model",
        validators=[DataRequired(), Length(min=1, max=128)],
    )
    base_url = StringField(
        "Base URL",
        validators=[Optional(), URL(require_tld=False)],
    )
    api_key = PasswordField(
        "API Key",
        validators=[Optional(), Length(max=256)],
    )
    temperature = FloatField(
        "Temperature",
        default=0.1,
        validators=[DataRequired(), NumberRange(min=0.0, max=2.0)],
    )
    submit = SubmitField("Save & Continue")


class EnableAuthForm(FlaskForm):  # type: ignore[misc]
    """Step 3 — confirmation to enable backend API authentication.

    Fields:
        confirm: Acknowledgement that API auth will be enabled.
        submit:  Submission button.
    """

    confirm = BooleanField(
        "Enable API key authentication on the backend",
        default=True,
        validators=[DataRequired()],
    )
    submit = SubmitField("Continue")


class ModelsForm(FlaskForm):  # type: ignore[misc]
    """Step 4 — model enablement (CSRF token only).

    The model checkboxes are rendered dynamically from the backend's
    ``GET /models`` response, so they are plain HTML inputs named
    ``model_enabled``; this form carries only the CSRF token and submit.

    Fields:
        submit: Submission button.
    """

    submit = SubmitField("Save & Continue")


class AdminCreateForm(FlaskForm):  # type: ignore[misc]
    """Step 5 — first admin API user.

    Fields:
        username: Username for the first admin API user.
        api_key:  API key for the user (auto-generated, editable).
        submit:   Submission button.
    """

    username = StringField(
        "Admin API Username",
        validators=[DataRequired(), Length(min=1, max=64)],
    )
    api_key = StringField(
        "API Key",
        validators=[DataRequired(), Length(min=8, max=256)],
    )
    submit = SubmitField("Create Admin & Finish Setup")
