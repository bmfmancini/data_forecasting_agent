"""
Flask extension instances for the forecaster frontend.

Each extension is created here without being bound to an application.
They are initialised against a concrete Flask app inside the
application factory (``app.create_app``).
"""

from __future__ import annotations

from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

login_manager: LoginManager = LoginManager()
csrf: CSRFProtect = CSRFProtect()
