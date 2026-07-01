"""
WTForms form definitions for the authentication blueprint.
"""

from __future__ import annotations

from flask_wtf import FlaskForm  # type: ignore[import-untyped]
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Length


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
