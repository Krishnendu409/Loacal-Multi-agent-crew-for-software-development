"""Optional lightweight research helpers for source-cited context."""

from __future__ import annotations

from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def fetch_research_context(
    urls: list[str],
    *,
    timeout_seconds: int = 10,
    max_chars_per_source: int = 2000,
) -> str:
    """Fetch plain-text snippets from URLs and return a cited context block."""
    snippets: list[str] = []
    for url in urls:
        if not isinstance(url, str) or not url.strip():
            continue
        clean_url = url.strip()
        parsed = urlparse(clean_url)
        if parsed.scheme not in {"http", "https"}:
            snippets.append(
                f"### Source: {clean_url}\n(skipped: unsupported URL scheme; only http/https allowed)"
            )
            continue
        try:
            req = Request(
                clean_url,
                headers={"User-Agent": "Local-Multi-Agent-Crew/1.0 (+research-mode)"},
            )
            with urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310
                raw = response.read()
            text = raw.decode("utf-8", errors="ignore")
            collapsed = " ".join(text.split())
            if not collapsed:
                continue
            snippets.append(f"### Source: {clean_url}\n{collapsed[:max_chars_per_source]}")
        except (OSError, URLError):
            snippets.append(f"### Source: {clean_url}\n(unavailable during this run)")
    if not snippets:
        return ""
    return "## Optional Research Context (cite sources)\n\n" + "\n\n".join(snippets)
