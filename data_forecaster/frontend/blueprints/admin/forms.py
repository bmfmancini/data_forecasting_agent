"""
WTForms form definitions for the administration blueprint.
"""

from __future__ import annotations

from flask_wtf import FlaskForm  # type: ignore[import-untyped]
from wtforms import (
    BooleanField,
    IntegerField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
)
from wtforms.validators import DataRequired, Length, NumberRange, Optional, URL


class UserCreateForm(FlaskForm):  # type: ignore[misc]
    """Form for creating a new application user.

    Fields:
        username:         Login name for the new user.
        password:         Initial password.
        confirm_password: Must match ``password``.
        role:             Role selection (``user`` or ``admin``).
        submit:           Submission button.
    """

    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=1, max=64)],
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=4)],
    )
    confirm_password = PasswordField(
        "Confirm Password",
        validators=[DataRequired()],
    )
    role = SelectField(
        "Role",
        choices=[("user", "User"), ("admin", "Administrator")],
        default="user",
    )
    submit = SubmitField("Create User")


class UserEditForm(FlaskForm):  # type: ignore[misc]
    """Form for editing an existing user account.

    Passwords may be left blank to keep the existing value.

    Fields:
        password:         New password (optional).
        confirm_password: Must match ``password`` when provided.
        role:             Role selection.
        active:           Whether the account is enabled.
        submit:           Submission button.
    """

    password = PasswordField(
        "New Password (leave blank to keep current)",
        validators=[Optional(), Length(min=4)],
    )
    confirm_password = PasswordField(
        "Confirm New Password",
        validators=[Optional()],
    )
    role = SelectField(
        "Role",
        choices=[("user", "User"), ("admin", "Administrator")],
    )
    active = BooleanField("Account active")
    force_password_reset = BooleanField("Force password change on next login")
    submit = SubmitField("Save Changes")


class APIConfigForm(FlaskForm):  # type: ignore[misc]
    """Form for configuring the backend API connection.

    Fields:
        base_url:          Root URL of the FastAPI backend.
        api_username:      Optional HTTP Basic Auth username.
        api_password:      Optional HTTP Basic Auth password.
        timeout:           Request timeout in seconds.
        submit:            Submission button.
    """

    base_url = StringField(
        "Backend API Base URL",
        validators=[DataRequired(), URL()],
    )
    api_username = StringField(
        "API Username (optional)",
        validators=[Optional(), Length(max=128)],
    )
    api_password = PasswordField(
        "API Key (optional)",
        validators=[Optional()],
    )
    timeout = IntegerField(
        "Timeout (seconds)",
        default=30,
        validators=[DataRequired(), NumberRange(min=1, max=300)],
    )
    submit = SubmitField("Save Configuration")


class APIKeyCreateForm(FlaskForm):  # type: ignore[misc]
    """Form for creating a new API key user.

    Fields:
        username:    Login name for the new API user.
        description: Human-readable description / purpose.
        submit:      Submission button.
    """

    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=1, max=64)],
    )
    description = StringField(
        "Description (optional)",
        validators=[Optional(), Length(max=256)],
    )
    submit = SubmitField("Create API User")
