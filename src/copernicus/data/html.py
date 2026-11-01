"""Strip HTML tags and entities from web text."""

from __future__ import annotations

import logging
import re
from html import unescape

logger = logging.getLogger(__name__)

__all__ = ["strip_html"]


def strip_html(text: str) -> str:
    """Strip HTML tags and entities from web text.

    Block-level elements become line breaks, list items become ``- `` lines,
    image alt text and link text are kept, and all other tags are removed.

    Args:
        text: Raw HTML string.

    Returns:
        Clean text with the HTML removed and paragraph breaks preserved. An empty
        input yields an empty string.
    """
    if not text:
        return ""

    out = text

    out = re.sub(r"<br\s*/?>", "\n", out)
    out = re.sub(r"</p>", "\n", out)
    out = re.sub(r"<p[^>]*>", "", out)
    out = re.sub(r"</div>", "\n", out)
    out = re.sub(r"<div[^>]*>", "", out)
    out = re.sub(r"</blockquote>", "\n", out)
    out = re.sub(r"<blockquote[^>]*>", "\n> ", out)

    out = re.sub(r"<li[^>]*>", "\n- ", out)
    out = re.sub(r"</li>", "", out)
    out = re.sub(r"</?[ou]l[^>]*>", "\n", out)

    out = re.sub(r"<h[1-6][^>]*>", "\n", out)
    out = re.sub(r"</h[1-6]>", "\n", out)

    out = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', r"\1", out)
    out = re.sub(r"<img[^>]*>", "", out)

    out = re.sub(r"<a[^>]*>([^<]*)</a>", r"\1", out)

    out = re.sub(r"<[^>]+>", " ", out)

    out = unescape(out)

    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r" {2,}", " ", out)
    out = re.sub(r"^\s+", "", out, flags=re.MULTILINE)

    return out.strip()
