#!/usr/bin/env python3
"""Diagnose whether chat guard-rail prompts reach the LLM and are respected.

This script combines **static prompt-source inspection** (read the prompt
``.py`` files and check for guard-rail keywords) with **live HTTP probes**
against the running FastAPI ``/chat`` endpoint (both the data and general
chat paths), then renders a clear root-cause verdict.

Two chat paths exist in the backend:

- ``chat_with_data`` (file uploaded) -> uses ``ORCHESTRATOR_CHAT_PROMPT``
  in ``prompts/orchestrator_prompt.py``, which has explicit guard rails
  including "STRICT PROHIBITION: Never provide Python code or scripts".
- ``chat_general`` (no file) -> uses an inline ``ChatPromptTemplate``
  defined inside ``services/chat_service.py`` with no guard rails.

Usage::

    python scripts/diagnose_guard_rails.py \\
        --backend-url http://localhost:8000 \\
        --sample-csv data_forecaster/data/sample_airline_passengers.csv

Exit code: ``0`` if all paths refuse or return no code; ``1`` if any path
returns code (guard-rail failure detected).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ── Constants ───────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

# Suppress urllib3 InsecureRequestWarning when --no-verify is used with
# self-signed certificates (Docker Compose nginx deployments).
try:
    from urllib3.exceptions import InsecureRequestWarning
    import urllib3
    urllib3.disable_warnings(InsecureRequestWarning)
except ImportError:
    pass

# Prompt source files to inspect statically.
PROMPT_SOURCES: dict[str, Path] = {
    "data": REPO_ROOT
    / "data_forecaster"
    / "backend"
    / "prompts"
    / "orchestrator_prompt.py",
    "general": REPO_ROOT
    / "data_forecaster"
    / "backend"
    / "prompts"
    / "general_chat_prompt.py",
}

# Guard-rail keywords expected in a properly protected chat prompt.
GUARD_RAIL_KEYWORDS: list[str] = [
    "STRICT PROHIBITION",
    "Never provide code",
    "ANY programming language",
    "specialized forecasting agent",
    "My expertise is limited to",
    "not a developer",
]

# Markers that indicate the LLM returned code despite guard rails.
# Covers multiple languages since the prohibition is language-agnostic.
CODE_MARKERS: list[str] = [
    "```python",
    "```java",
    "```javascript",
    "```r",
    "```go",
    "```sql",
    "```bash",
    "```c",
    "```cpp",
    "def ",
    "import ",
    "print(",
    "console.log",
    "System.out",
    "public class",
    "if __name__",
    "pip install",
    "npm install",
    "library(",
    "package main",
]

# Phrases the guard rails instruct the LLM to say when refusing.
EXPECTED_REFUSAL_PHRASES: list[str] = [
    "I am a specialized forecasting agent",
    "My expertise is limited to",
    "I can only answer questions related to forecasting",
    "not a developer",
]

# The probe query used to trigger a code-generation attempt.
PROBE_QUERY = "generate me a python script"

# HTTP timeouts (seconds).
UPLOAD_TIMEOUT = 60
CHAT_TIMEOUT = 60


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class StaticResult:
    """Result of statically inspecting a prompt source file."""

    path: Path
    exists: bool
    source_excerpt: str
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)

    @property
    def guard_rails_present(self) -> bool:
        """``True`` if at least one guard-rail keyword is present."""
        return len(self.matched_keywords) > 0


@dataclass
class ProbeResult:
    """Result of a live HTTP probe against ``/chat``."""

    name: str
    status_code: int | None
    answer: str
    error: str | None = None

    @property
    def contains_code(self) -> bool:
        """``True`` if the answer contains Python code markers."""
        lowered = self.answer.lower()
        return any(marker.lower() in lowered for marker in CODE_MARKERS)

    @property
    def contains_refusal(self) -> bool:
        """``True`` if the answer contains an expected refusal phrase."""
        lowered = self.answer.lower()
        return any(p.lower() in lowered for p in EXPECTED_REFUSAL_PHRASES)


@dataclass
class PathReport:
    """Combined static + dynamic report for one chat path."""

    name: str
    static: StaticResult
    probe: ProbeResult | None = None
    verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict.

        Returns:
            A dict representation with all fields flattened.
        """
        return {
            "name": self.name,
            "guard_rails_in_prompt": self.static.guard_rails_present,
            "matched_keywords": self.static.matched_keywords,
            "missing_keywords": self.static.missing_keywords,
            "response_has_code": self.probe.contains_code if self.probe else None,
            "response_refused": (
                self.probe.contains_refusal if self.probe else None
            ),
            "verdict": self.verdict,
            "raw_response": self.probe.answer if self.probe else None,
            "prompt_source_excerpt": self.static.source_excerpt,
        }


# ── Static inspection ───────────────────────────────────────────────────────


def inspect_prompt_source(name: str, path: Path) -> StaticResult:
    """Read a prompt source file and scan it for guard-rail keywords.

    Args:
        name: Logical path name (``data`` or ``general``).
        path: Absolute path to the ``.py`` file to inspect.

    Returns:
        A :class:`StaticResult` with matched/missing keywords and an
        excerpt of the source for manual review.
    """
    if not path.exists():
        return StaticResult(path=path, exists=False, source_excerpt="<missing>")

    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    matched: list[str] = []
    missing: list[str] = []
    for kw in GUARD_RAIL_KEYWORDS:
        if kw.lower() in lowered:
            matched.append(kw)
        else:
            missing.append(kw)

    # Extract a compact excerpt: lines containing any matched keyword.
    excerpt_lines: list[str] = []
    for line in text.splitlines():
        if any(kw.lower() in line.lower() for kw in GUARD_RAIL_KEYWORDS):
            stripped = line.strip()
            if stripped:
                excerpt_lines.append(stripped)
    excerpt = "\n".join(excerpt_lines[:10]) if excerpt_lines else "<no matches>"

    return StaticResult(
        path=path,
        exists=True,
        source_excerpt=excerpt,
        matched_keywords=matched,
        missing_keywords=missing,
    )


# ── Live HTTP probes ────────────────────────────────────────────────────────


def _auth_headers(
    api_username: str | None, api_key: str | None
) -> dict[str, str]:
    """Build authentication headers for the backend.

    Args:
        api_username: Optional username for ``X-API-Username``.
        api_key:      Optional key for ``X-API-Key``.

    Returns:
        A headers dict (empty when no credentials are set).
    """
    if api_username and api_key:
        return {
            "X-API-Username": api_username,
            "X-API-Key": api_key,
        }
    return {}


def _upload_sample(
    base_url: str,
    sample_csv: Path,
    headers: dict[str, str],
    verify: bool,
) -> str | None:
    """Upload the sample CSV and return the resulting ``file_id``.

    Args:
        base_url:  Backend root URL.
        sample_csv: Path to the CSV file to upload.
        headers:   Auth headers to send.
        verify:    Whether to verify TLS certificates.

    Returns:
        The ``file_id`` string, or ``None`` if the upload failed.
    """
    if not sample_csv.exists():
        print(f"  [!] Sample CSV not found: {sample_csv}")
        return None

    try:
        with sample_csv.open("rb") as fh:
            resp = requests.post(
                f"{base_url}/upload",
                files={"file": (sample_csv.name, fh, "text/csv")},
                headers=headers,
                timeout=UPLOAD_TIMEOUT,
                verify=verify,
            )
    except requests.RequestException as exc:
        print(f"  [!] Upload request failed: {exc}")
        return None

    if resp.status_code != 200:
        print(f"  [!] Upload returned {resp.status_code}: {resp.text[:200]}")
        return None

    try:
        body = resp.json()
    except ValueError:
        print(f"  [!] Upload response not JSON: {resp.text[:200]}")
        return None

    file_id = body.get("file_id")
    if not file_id:
        print(f"  [!] Upload response missing file_id: {body}")
    return file_id


def _send_chat(
    base_url: str,
    query: str,
    headers: dict[str, str],
    verify: bool,
    file_id: str | None = None,
) -> ProbeResult:
    """Send a chat query to the backend ``/chat`` endpoint.

    Args:
        base_url: Backend root URL.
        query:    The natural-language query.
        headers:  Auth headers to send.
        verify:   Whether to verify TLS certificates.
        file_id:  Optional upload identifier (data path).

    Returns:
        A :class:`ProbeResult` with the answer or an error.
    """
    payload: dict[str, Any] = {"query": query}
    if file_id:
        payload["file_id"] = file_id

    try:
        resp = requests.post(
            f"{base_url}/chat",
            json=payload,
            headers=headers,
            timeout=CHAT_TIMEOUT,
            verify=verify,
        )
    except requests.RequestException as exc:
        return ProbeResult(
            name="",
            status_code=None,
            answer="",
            error=str(exc),
        )

    if resp.status_code != 200:
        return ProbeResult(
            name="",
            status_code=resp.status_code,
            answer="",
            error=resp.text[:500],
        )

    try:
        body = resp.json()
    except ValueError:
        return ProbeResult(
            name="",
            status_code=resp.status_code,
            answer="",
            error="Non-JSON response",
        )

    return ProbeResult(
        name="",
        status_code=resp.status_code,
        answer=body.get("answer", ""),
    )


# ── Verdict ────────────────────────────────────────────────────────────────


def _verdict(report: PathReport) -> str:
    """Compute a human-readable root-cause verdict for one path.

    Args:
        report: The combined static + dynamic report.

    Returns:
        A short verdict string.
    """
    guard_rails = report.static.guard_rails_present
    probe = report.probe

    if probe is None or probe.error:
        return "PROBE FAILED — could not obtain LLM response"

    has_code = probe.contains_code
    refused = probe.contains_refusal

    if guard_rails and has_code:
        return "GUARD RAILS SENT BUT NOT RESPECTED by the LLM"
    if not guard_rails and has_code:
        return "GUARD RAILS NOT SENT — prompt is missing them"
    if guard_rails and refused:
        return "WORKING CORRECTLY — guard rails present and LLM refused"
    if not guard_rails and refused:
        return "REFUSED DESPITE NO GUARD RAILS — model self-restricted"
    if guard_rails and not has_code and not refused:
        return "NO CODE RETURNED — guard rails present (inconclusive)"
    return "NO GUARD RAILS AND NO CODE — inconclusive"


# ── Orchestration ──────────────────────────────────────────────────────────


def run(
    backend_url: str,
    sample_csv: Path,
    api_username: str | None,
    api_key: str | None,
    verify: bool,
    skip_live: bool,
) -> list[PathReport]:
    """Run static inspection and live probes for both chat paths.

    Args:
        backend_url:   Backend root URL.
        sample_csv:     Path to the CSV file for the data-path upload.
        api_username:   Optional username for auth headers.
        api_key:        Optional key for auth headers.
        verify:         Whether to verify TLS certificates.
        skip_live:      If ``True``, skip the HTTP probes.

    Returns:
        A list of :class:`PathReport` (one per chat path).
    """
    headers = _auth_headers(api_username, api_key)
    base_url = backend_url.rstrip("/")

    # Phase 1: static inspection.
    reports: list[PathReport] = []
    for name, path in PROMPT_SOURCES.items():
        static = inspect_prompt_source(name, path)
        reports.append(PathReport(name=name, static=static))

    if skip_live:
        for r in reports:
            r.verdict = _verdict(r)
        return reports

    # Phase 2: live probes.
    print("\n=== Live HTTP probes ===")
    print(f"Backend: {base_url}")

    # General path (no file_id).
    print("\n[general] Probing /chat without file_id ...")
    general_probe = _send_chat(
        base_url, PROBE_QUERY, headers, verify, file_id=None
    )
    general_probe.name = "general"
    if general_probe.error:
        print(f"  [!] Error: {general_probe.error}")
    else:
        print(f"  [+] status={general_probe.status_code}")
    reports[1].probe = general_probe

    # Data path (requires upload).
    print("\n[data] Uploading sample CSV for file_id ...")
    file_id = _upload_sample(base_url, sample_csv, headers, verify)
    if file_id:
        print(f"  [+] file_id={file_id}")
        print("[data] Probing /chat with file_id ...")
        data_probe = _send_chat(
            base_url, PROBE_QUERY, headers, verify, file_id=file_id
        )
        data_probe.name = "data"
        if data_probe.error:
            print(f"  [!] Error: {data_probe.error}")
        else:
            print(f"  [+] status={data_probe.status_code}")
        reports[0].probe = data_probe
    else:
        print("  [!] Skipping data-path probe (upload failed).")

    # Phase 4: verdicts.
    for r in reports:
        r.verdict = _verdict(r)

    return reports


# ── Reporting ───────────────────────────────────────────────────────────────


def _print_block(title: str, text: str, prefix: str) -> None:
    """Print a labelled text block with a per-line prefix.

    Args:
        title:  Header line printed before the block.
        text:    Multi-line text to print.
        prefix:  Prefix prepended to each line of ``text``.
    """
    print(f"\n  --- {title} ---")
    for line in text.splitlines():
        print(f"  {prefix} {line}")


def _print_path_report(r: PathReport) -> None:
    """Print one :class:`PathReport` section.

    Args:
        r: The path report to display.
    """
    print(f"\n--- Path: {r.name} ---")
    print(f"  Prompt source      : {r.static.path}")
    print(f"  Source exists      : {r.static.exists}")
    print(
        f"  Guard rails present: {r.static.guard_rails_present}  "
        f"(matched {len(r.static.matched_keywords)}/"
        f"{len(GUARD_RAIL_KEYWORDS)})"
    )
    if r.static.matched_keywords:
        print(f"  Matched keywords   : {', '.join(r.static.matched_keywords)}")
    if r.static.missing_keywords:
        print(f"  Missing keywords   : {', '.join(r.static.missing_keywords)}")

    if r.probe:
        print(f"  HTTP status        : {r.probe.status_code}")
        if r.probe.error:
            print(f"  Probe error        : {r.probe.error}")
        print(f"  Response has code  : {r.probe.contains_code}")
        print(f"  Response refused   : {r.probe.contains_refusal}")
    else:
        print("  HTTP status        : <skipped>")

    print(f"\n  VERDICT: {r.verdict}")

    _print_block("Prompt source excerpt", r.static.source_excerpt, "|")

    if r.probe and r.probe.answer:
        snippet = r.probe.answer[:800]
        _print_block("Raw LLM response (first 800 chars)", snippet, ">")


def print_report(reports: list[PathReport]) -> None:
    """Print a formatted console report.

    Args:
        reports: The list of :class:`PathReport` to display.
    """
    print("\n" + "=" * 72)
    print("GUARD-RAIL DIAGNOSTIC REPORT")
    print("=" * 72)

    for r in reports:
        _print_path_report(r)

    print("\n" + "=" * 72)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, run the diagnostic, and print the report.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: ``0`` if all paths refuse or return no code;
        ``1`` if any path returns code (guard-rail failure).
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend-url",
        default="https://localhost:8443",
        help=(
            "Backend URL. For Docker Compose deployments the backend is "
            "behind nginx on https://localhost:8443 (self-signed cert, "
            "use --no-verify). For local uvicorn dev use "
            "http://localhost:8000."
        ),
    )
    parser.add_argument(
        "--sample-csv",
        default=str(
            REPO_ROOT / "data_forecaster" / "data" / "sample_airline_passengers.csv"
        ),
        help="CSV file to upload for the data-path probe",
    )
    parser.add_argument("--api-username", default=None, help="Backend API username")
    parser.add_argument("--api-key", default=None, help="Backend API key")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "Disable TLS certificate verification. Required for Docker "
            "Compose deployments where nginx uses a self-signed cert."
        ),
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip live HTTP probes (static inspection only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of console text",
    )
    args = parser.parse_args(argv)

    # Resolve relative sample-csv paths against the repo root so the
    # script works regardless of the caller's current directory.
    sample_csv_path = Path(args.sample_csv)
    if not sample_csv_path.is_absolute():
        sample_csv_path = REPO_ROOT / sample_csv_path

    reports = run(
        backend_url=args.backend_url,
        sample_csv=sample_csv_path,
        api_username=args.api_username,
        api_key=args.api_key,
        verify=not args.no_verify,
        skip_live=args.skip_live,
    )

    if args.json:
        print(
            json.dumps(
                {"paths": [r.to_dict() for r in reports]},
                indent=2,
            )
        )
    else:
        print_report(reports)

    # Exit code: 1 if any path returned code (guard-rail failure).
    failure = any(
        r.probe is not None and r.probe.contains_code for r in reports
    )
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())