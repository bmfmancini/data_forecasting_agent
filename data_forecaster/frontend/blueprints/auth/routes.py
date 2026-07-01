"""
Route handlers for the authentication blueprint.

Provides ``/auth/login`` and ``/auth/logout``.
"""

from __future__ import annotations

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user
from werkzeug.security import check_password_hash
from werkzeug.wrappers import Response

from blueprints.auth import auth_bp
from blueprints.auth.forms import LoginForm
from db.db import query_db
from models import User


@auth_bp.route("/login", methods=["GET", "POST"])
def login() -> str | Response:
    """Render and process the login form.

    Redirects authenticated users to the main application.  On a valid
    POST the user is logged in and redirected to the ``next`` parameter or
    the main index.  Invalid credentials result in an error flash.

    Returns:
        The rendered login template or a redirect response.
    """
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data or ""
        password = form.password.data or ""

        row = query_db(
            """
            SELECT u.id, u.username, u.password_hash, r.name AS role_name, u.active
            FROM users u
            JOIN roles r ON r.id = u.role_id
            WHERE u.username = ?
            """,
            (username,),
            one=True,
        )

        if (
            row
            and isinstance(row, dict)
            and bool(row.get("active"))
            and check_password_hash(str(row["password_hash"]), password)
        ):
            user = User(
                user_id=int(row["id"]),
                username=str(row["username"]),
                role_name=str(row["role_name"]),
                active=True,
            )
            login_user(user)
            next_page: str = request.args.get("next", "")
            if next_page and next_page.startswith("/"):
                return redirect(next_page)
            return redirect(url_for("main.index"))

        flash("Invalid username or password.", "danger")

    return render_template("auth/login.html", form=form)


@auth_bp.route("/logout")
def logout() -> Response:
    """Log out the current user and redirect to the login page.

    Returns:
        A redirect response to the login page.
    """
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
