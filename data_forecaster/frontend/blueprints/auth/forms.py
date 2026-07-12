"""
WTForms form definitions for the authentication blueprint.
"""

from __future__ import annotations

from flask_wtf import FlaskForm  # type: ignore[import-untyped]
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, EqualTo, Length, Regexp

PASSWORD_COMPLEXITY_MESSAGE = (
    "Password must include uppercase, lowercase, number, and special character."
)
PASSWORD_COMPLEXITY_RE = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)" r"(?=.*[^A-Za-z0-9]).+$"
PASSWORD_VALIDATORS = [
    DataRequired(),
    Length(min=8),
    Regexp(PASSWORD_COMPLEXITY_RE, message=PASSWORD_COMPLEXITY_MESSAGE),
]


class LoginForm(FlaskForm):  # type: ignore[misc]
    """Form for user authentication.

    Fields:
        username: The user's login name.
        password: The user's password.
        submit:   Submission button.
    """

    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=1, max=64)],
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired()],
    )
    submit = SubmitField("Log In")


class ChangePasswordForm(FlaskForm):  # type: ignore[misc]
    """Form for forced password rotation on first login.

    Fields:
        current_password:    The user's current password.
        new_password:        The desired new password.
        confirm_password:    Must match ``new_password``.
        submit:              Submission button.
    """

    current_password = PasswordField(
        "Current Password",
        validators=[DataRequired()],
    )
    new_password = PasswordField(
        "New Password",
        validators=PASSWORD_VALIDATORS,
    )
    confirm_password = PasswordField(
        "Confirm New Password",
        validators=[
            DataRequired(),
            EqualTo("new_password", message="Passwords must match."),
        ],
    )
    submit = SubmitField("Update Password")
