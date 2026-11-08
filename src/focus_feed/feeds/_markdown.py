"""Markdown conversion and CDATA sanitization helpers for feeds."""

import re

import markdown as _md

# URL schemes allowed in markdown links to prevent XSS (e.g. javascript:).
# Only http, https, mailto, relative paths, and fragment identifiers are allowed.
_SAFE_URL_SCHEMES = re.compile(r"^(https?://|mailto:|/|#|\.\.)")


def _markdown_to_html(text: str | None) -> str:
    """Convert markdown text to HTML using the ``markdown`` library.

    Uses the third-party ``markdown`` library (a required dependency) for
    conversion.  The ``nl2br`` extension is enabled so that single newlines
    within paragraphs are rendered as ``<br>`` elements.

    The ``markdown`` library handles HTML escaping of text content to
    prevent XSS vulnerabilities.  URLs in links are additionally validated
    to only allow safe schemes (``http://``, ``https://``, ``mailto:``,
    relative paths, and fragment identifiers).  Links with unsafe schemes
    (e.g. ``javascript:``) have their ``<a>`` tags removed but the link
    text is preserved.

    If the entire output is a single ``<p>`` wrapper (i.e. the input was a
    plain paragraph with no other markdown elements), the outer ``<p>`` and
    ``</p>`` tags are stripped to avoid redundant wrappers.  Legitimate
         ``<p>`` tags from actual multi-paragraph markdown are preserved.

    The resulting HTML will be wrapped in CDATA sections in RSS feed output
    to preserve HTML formatting for feed readers.

    Args:
        text: The markdown string to convert, or ``None``.

    Returns:
        An HTML string.  Returns an empty string when *text* is ``None``
        or empty.
    """
    if text is None:
        return ""

    # Fix 3-space indentation for nested list items (LLM quirk)
    text = re.sub(r"\n   ([-*+])", r"\n    \1", text)

    # A sequence of standalone `---` lines is common in LLM-generated
    # summaries, but Python-Markdown can interpret the second line as a
    # setext heading underline (e.g. `---\n---` becomes `<h2>---</h2>`).
    # Use an equivalent horizontal-rule marker only for consecutive runs;
    # leave single `---` lines untouched so valid setext headings still work.
    text = re.sub(
        r"(?m)(^[ \t]*---[ \t]*(?:\r?\n|$)){2,}",
        lambda match: match.group().replace("---", "***"),
        text,
    )

    # Use the markdown library with the nl2br extension (single newlines → <br>).
    # Deregister the html_block preprocessor and html inline pattern so that
    # raw HTML in the input is escaped rather than passed through, preventing
    # XSS vulnerabilities (the markdown library does not sanitize by default).
    md = _md.Markdown(extensions=["nl2br"])
    md.preprocessors.deregister("html_block")
    md.inlinePatterns.deregister("html")
    result = md.convert(text)

    # Sanitize URLs in <a> tags — only allow safe schemes.  Unsafe links
    # have their <a> tags removed but the link text is preserved.
    result = re.sub(
        r'<a\s[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        lambda m: m.group(2) if not _SAFE_URL_SCHEMES.match(m.group(1)) else m.group(0),
        result,
        flags=re.DOTALL,
    )

    # Replace XHTML-style <br /> with HTML-style <br> for consistency.
    result = result.replace("<br />", "<br>")

    # Strip the outer <p> wrapper when the entire output is a single
    # <p> element.  This avoids redundant paragraph tags for plain-text
    # descriptions while preserving legitimate <p> tags from actual
    # multi-paragraph markdown content.
    if result.startswith("<p>") and result.endswith("</p>"):
        open_count = result.count("<p>")
        close_count = result.count("</p>")
        if open_count == 1 and close_count == 1:
            result = result[3:-4]

    return result


def _sanitize_cdata_content(text: str) -> str:
    """Sanitize content for safe CDATA wrapping.

    CDATA sections cannot contain the sequence ``]]>`` which would prematurely
    close the CDATA block. This function replaces any occurrence with ``]]&gt;``
    to prevent CDATA termination.

    Args:
        text: The HTML string to sanitize for CDATA.

    Returns:
        The sanitized string safe for CDATA wrapping.
    """
    return text.replace("]]>", "]]&gt;")
