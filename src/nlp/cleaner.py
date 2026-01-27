"""HTML cleaning and text normalization for job descriptions."""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def clean_html(html: str) -> str:
    """Strip HTML tags and return plain text, preserving line breaks."""
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Replace <br>, <p>, <li> with newlines for structure preservation
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for tag in soup.find_all(["p", "li", "div", "h1", "h2", "h3", "h4", "h5", "h6"]):
            tag.insert_before("\n")
            tag.insert_after("\n")
        text = soup.get_text()
    except Exception:
        text = html
    # Normalize whitespace within lines but preserve line breaks
    lines = text.split("\n")
    lines = [re.sub(r"[\t\r\u00A0]+", " ", line).strip() for line in lines]
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """Normalize whitespace in already-cleaned text to single line."""
    if not text:
        return ""
    text = re.sub(r"\r|\t|\u00A0", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_matching(text: str) -> str:
    """Lowercase and collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()
