"""Report edits preserve originals and charts while sharing web/PDF content."""
import pytest
from data_forecaster.frontend.services.report_editing import (
    EditConflict, apply_section_edit, effective_markdown, report_sections,
)

ORIGINAL = "## 1. Summary\n\nOriginal wording.\n\n---\n\n## 2. Forecast\n\nForecast wording.\n\n[VISUAL:FORECAST]"


def test_edit_remove_restore_and_reset():
    result = {"original_report": ORIGINAL}
    section = report_sections(result)[1]
    result["section_edits"] = apply_section_edit(result, "1", "save", section["version"], "Outlook", "My **revised** wording.")
    assert "My **revised** wording." in effective_markdown(result)
    assert "Forecast wording." not in effective_markdown(result)
    assert "[VISUAL:FORECAST]" in effective_markdown(result)
    section = report_sections(result)[1]
    result["section_edits"] = apply_section_edit(result, "1", "remove", section["version"])
    assert "Outlook" not in effective_markdown(result)
    assert "[VISUAL:FORECAST]" not in effective_markdown(result)
    result["section_edits"] = apply_section_edit(result, "1", "restore", report_sections(result)[1]["version"])
    assert "My **revised** wording." in effective_markdown(result)
    result["section_edits"] = apply_section_edit(result, "1", "reset", report_sections(result)[1]["version"])
    assert effective_markdown(result) == ORIGINAL


def test_stale_editor_cannot_overwrite_and_figures_are_not_editable_tokens():
    result = {"original_report": ORIGINAL}
    section = report_sections(result)[1]
    assert "VISUAL" not in section["edit_body"]
    result["section_edits"] = apply_section_edit(result, "1", "remove", section["version"])
    with pytest.raises(EditConflict):
        apply_section_edit(result, "1", "save", section["version"], "Old tab", "Oops")


def test_legacy_unsectioned_reports_can_be_edited():
    result = {"report": "Legacy report"}
    section = report_sections(result)[0]
    assert section["edit_body"] == "Legacy report"
    assert apply_section_edit(result, "0", "save", section["version"], "Report", "Updated")


def test_invalid_fields_rejected():
    result = {"report": ORIGINAL}
    version = report_sections(result)[0]["version"]
    for title, body in [("", "hello"), ("title\ninjection", "hello"), ("title", "## New section")]:
        with pytest.raises(ValueError):
            apply_section_edit(result, "0", "save", version, title, body)
