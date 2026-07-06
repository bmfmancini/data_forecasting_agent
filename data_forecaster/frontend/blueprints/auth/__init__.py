"""Auth blueprint package."""

from flask import Blueprint

auth_bp: Blueprint = Blueprint("auth", __name__, url_prefix="/auth")

from blueprints.auth import routes  # noqa: E402, F401
