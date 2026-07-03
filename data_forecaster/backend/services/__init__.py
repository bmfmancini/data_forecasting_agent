"""Service layer for the Data Forecaster backend.

Contains business logic extracted from route handlers to keep
``main.py`` thin.  Services manage in-memory stores, job queuing,
pipeline orchestration, chat, and the RAG knowledge base.
"""