"""
WSGI entry point for production deployment with Gunicorn.

Usage:
    gunicorn wsgi:application
"""

from __future__ import annotations

from app import create_app

application = create_app("production")
