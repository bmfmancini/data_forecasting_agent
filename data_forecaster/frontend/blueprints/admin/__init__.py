"""Admin blueprint package."""

from flask import Blueprint

admin_bp: Blueprint = Blueprint("admin", __name__, url_prefix="/admin")

from blueprints.admin import routes  # noqa: E402, F401
