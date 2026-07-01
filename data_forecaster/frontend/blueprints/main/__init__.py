"""Main blueprint package."""

from flask import Blueprint

main_bp: Blueprint = Blueprint("main", __name__)

from blueprints.main import routes  # noqa: E402, F401
