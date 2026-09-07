"""Section edits layered over the original generated report."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_VISUAL = re.compile(r"\[VISUAL:[A-Z_]+\]")


class EditConflict(ValueError):
    """The section changed since the editor was opened."""


def report_sections(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Split generated section headings, retaining stable original section IDs."""
    original = str(result.get("original_report", result.get("report", "")))
    chunks = re.split(r"(?m)(?=^## (?!#))", original)
    sections = []
    overrides = result.get("section_edits") or {}
    for chunk in chunks:
        if not chunk.strip():
            continue
        chunk = re.sub(r"\n\s*---\s*$", "", chunk).strip()
        key = str(len(sections))
        match = re.match(r"## ([^\n]+)\n?", chunk)
        title = match.group(1) if match else "Report introduction"
        body = chunk[match.end() :].strip() if match else chunk
        is_dashboard = bool(
            re.fullmatch(r"(?:1\.\s*)?Executive Dashboard", title, re.IGNORECASE)
        )
        edit = overrides.get(key, {})
        title = edit.get("title", title)
        body = edit.get("body", body)
        removed = bool(edit.get("removed", False))
        version = hashlib.sha256(
            json.dumps([title, body, removed]).encode()
        ).hexdigest()
        sections.append(
            {
                "id": key,
                "title": title,
                "body": body,
                "is_dashboard": is_dashboard,
                "edit_body": _VISUAL.sub("", body).strip(),
                "removed": removed,
                "edited": bool(edit),
                "version": version,
            }
        )
    return sections


def effective_markdown(result: dict[str, Any]) -> str:
    """Compose the visible report used by both web and PDF views."""
    if not result.get("section_edits"):
        return str(result.get("original_report", result.get("report", "")))
    return "\n\n---\n\n".join(
        f"## {s['title']}\n\n{s['body']}"
        for s in report_sections(result)
        if not s["removed"]
    )


def apply_section_edit(
    result: dict[str, Any],
    section_id: str,
    action: str,
    version: str,
    title: str = "",
    body: str = "",
) -> dict[str, Any]:
    """Validate and return a new override map without changing forecast data."""
    sections = {s["id"]: s for s in report_sections(result)}
    if section_id not in sections:
        raise ValueError("Unknown report section.")
    section = sections[section_id]
    if version != section["version"]:
        raise EditConflict(
            "This section changed in another tab. Reload the report before editing it again."
        )
    edits = dict(result.get("section_edits") or {})
    if action == "reset":
        edits.pop(section_id, None)
    elif action in {"remove", "restore"}:
        edits[section_id] = {
            "title": section["title"],
            "body": section["body"],
            "removed": action == "remove",
        }
    elif action == "save":
        title = title.strip()
        if not title or len(title) > 200 or "\n" in title or "\r" in title:
            raise ValueError(
                "Use a section title between 1 and 200 characters on one line."
            )
        if len(body) > 100000:
            raise ValueError("Section text must be at most 100,000 characters.")
        if re.search(r"(?m)^## (?!#)", body):
            raise ValueError(
                "Use the title field for the section heading; use ### for subheadings."
            )
        # Keep the existing figures without exposing chart tokens in the editor.
        figures = _VISUAL.findall(section["body"])
        body = _VISUAL.sub("", body).strip()
        if figures:
            body += "\n\n" + "\n\n".join(figures)
        edits[section_id] = {"title": title, "body": body, "removed": False}
    else:
        raise ValueError("Unknown editing action.")
    return edits
