"""Optional lightweight research helpers for source-cited context."""

from __future__ import annotations

import logging
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# BUG FIX: strip HTML tags from fetched content so agents get plain text, not raw HTML
import re as _re
_HTML_TAG_RE = _re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = _re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = _HTML_TAG_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


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
                f"### Source: {clean_url}\n(skipped: only http/https URLs are supported)"
            )
            continue
        try:
            req = Request(
                clean_url,
                headers={"User-Agent": "Local-Multi-Agent-Crew/1.0 (+research-mode)"},
            )
            with urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310
                raw = response.read()
            # BUG FIX: respect content-type charset if present
            content_type = ""
            if hasattr(response, "headers"):
                content_type = response.headers.get("Content-Type", "")
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip() or "utf-8"
            text = raw.decode(charset, errors="ignore")
            # Strip HTML if content looks like HTML
            if "<html" in text.lower()[:1000] or "<!doctype" in text.lower()[:100]:
                text = _strip_html(text)
            collapsed = _MULTI_SPACE_RE.sub(" ", text).strip()
            if not collapsed:
                continue
            snippets.append(f"### Source: {clean_url}\n{collapsed[:max_chars_per_source]}")
        except (OSError, URLError) as exc:
            logger.debug("Research fetch failed for %s: %s", clean_url, exc)
            snippets.append(f"### Source: {clean_url}\n(unavailable during this run)")
    if not snippets:
        return ""
    return "## Optional Research Context (cite sources)\n\n" + "\n\n".join(snippets)
