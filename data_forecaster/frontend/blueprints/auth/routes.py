"""
Route handlers for the authentication blueprint.

Provides ``/auth/login`` and ``/auth/logout``.
"""

from __future__ import annotations

from flask import flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.wrappers import Response

from blueprints.auth import auth_bp
from blueprints.auth.forms import ChangePasswordForm, LoginForm
from blueprints.decorators import password_change_required
from db.db import execute_db, query_db
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
            SELECT u.id, u.username, u.password_hash, r.name AS role_name,
                   u.active, u.must_change_password
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
                must_change_password=bool(row.get("must_change_password", 0)),
            )
            login_user(user)
            if user.must_change_password:
                flash(
                    "You must change your default password before continuing.",
                    "warning",
                )
                return redirect(url_for("auth.change_password"))
            next_page: str = request.args.get("next", "")
            if next_page and next_page.startswith("/"):
                return redirect(next_page)
            return redirect(url_for("main.index"))

        flash("Invalid username or password.", "danger")

    return render_template("auth/login.html", form=form)


@auth_bp.route("/logout")
@password_change_required
def logout() -> Response:
    """Log out the current user and redirect to the login page.

    Returns:
        A redirect response to the login page.
    """
    logout_user()
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
@password_change_required
def change_password() -> str | Response:
    """Force a password change when ``must_change_password`` is set.

    On a valid POST, verifies the current password, updates the hash,
    clears the ``must_change_password`` flag, and redirects to the main
    index.  The user cannot navigate away from this page until the
    password is changed.

    Returns:
        The rendered change-password template or a redirect response.
    """

    form = ChangePasswordForm()
    if form.validate_on_submit():
        current_pw: str = str(form.current_password.data or "")
        new_pw: str = str(form.new_password.data or "")

        row = query_db(
            "SELECT password_hash FROM users WHERE id = ?",
            (current_user.id,),  # type: ignore[union-attr]
            one=True,
        )
        if not row or not isinstance(row, dict):
            flash("User not found.", "danger")
            return redirect(url_for("auth.logout"))

        if not check_password_hash(str(row["password_hash"]), current_pw):
            flash("Current password is incorrect.", "danger")
            return render_template("auth/change_password.html", form=form)

        if check_password_hash(str(row["password_hash"]), new_pw):
            flash(
                "New password must be different from the current password.",
                "danger",
            )
            return render_template("auth/change_password.html", form=form)

        new_hash: str = generate_password_hash(new_pw)
        execute_db(
            "UPDATE users SET password_hash = ?, must_change_password = 0 WHERE id = ?",
            (new_hash, current_user.id),  # type: ignore[union-attr]
        )
        flash("Password updated successfully.", "success")
        return redirect(url_for("main.index"))

    return render_template("auth/change_password.html", form=form)
