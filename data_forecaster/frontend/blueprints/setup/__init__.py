"""Setup wizard blueprint package."""

from flask import Blueprint

setup_bp: Blueprint = Blueprint("setup", __name__, url_prefix="/setup")

from blueprints.setup import routes  # noqa: E402, F401
