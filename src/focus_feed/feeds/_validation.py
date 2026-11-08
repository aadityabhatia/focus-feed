"""RSS feed validation helpers."""

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

from focus_feed.feeds._mime import PODCAST_MIME_TYPES

_DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"


@dataclass
class FeedValidationResult:
    """Structured result of validating an RSS feed.

    Attributes:
        feed_name: Name of the feed being validated (e.g., ``"podcast.xml"``).
        valid: ``True`` when no errors were found.
        errors: List of error messages. Non-empty implies ``valid`` is ``False``.
        warnings: List of warning messages. Non-empty does not affect ``valid``.
    """

    feed_name: str
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return ``True`` if the feed passed all validation checks."""
        return self.valid and not self.errors


def validate_feed(
    xml_content: str,
    feed_name: str = "feed",
    is_podcast: bool = False,
) -> FeedValidationResult:
    """Validate an RSS 2.0 feed.

    Performs the following checks:

    - **XML well-formedness** — the content can be parsed by
      :mod:`xml.etree.ElementTree`.  This also verifies that XML special
      characters (``<``, ``>``, ``&``, etc.) are properly escaped.
    - **RSS 2.0 compliance** — the root element is ``<rss>`` with
      ``version="2.0"``.
    - **Item required elements** — each ``<item>`` has a non-empty
      ``<title>``, ``<link>``, ``<guid>``, and a ``<pubDate>`` *or*
      ``<dc:date>``.
    - **Podcasts** — each ``<item>`` has an ``<enclosure>`` tag with
      ``url``, a supported media MIME type, and ``length`` attributes.
    - **Articles** — no enclosure validation is performed.

    Args:
        xml_content: Raw XML string of the feed to validate.
        feed_name: Human-readable name of the feed (used in messages).
        is_podcast: When ``True``, enforce podcast-specific enclosure
            checks.  When ``False`` (articles), enclosure checks are skipped.

    Returns:
        A :class:`FeedValidationResult` with any errors and warnings found.
    """
    result = FeedValidationResult(feed_name=feed_name)

    # --- XML well-formedness (also verifies proper character escaping) ---
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        result.errors.append(f"XML is not well-formed: {exc}")
        result.valid = False
        return result

    # --- Root element <rss> with version="2.0" ---
    if root.tag != "rss":
        result.errors.append(f"Root element is <{root.tag}>, expected <rss>")
        result.valid = False
        return result

    version = root.get("version")
    if version != "2.0":
        result.errors.append(f"RSS version is '{version}', expected '2.0'")
        result.valid = False

    # --- <channel> element ---
    channel = root.find("channel")
    if channel is None:
        result.errors.append("Missing <channel> element")
        result.valid = False
        return result

    # --- Per-item validation ---
    items = channel.findall("item")
    if not items:
        result.warnings.append("No <item> elements found in feed")

    for index, item in enumerate(items, 1):
        _validate_item(item, index, is_podcast, result)

    if result.errors:
        result.valid = False

    return result


def _validate_item(
    item: ET.Element,
    index: int,
    is_podcast: bool,
    result: FeedValidationResult,
) -> None:
    """Validate a single ``<item>`` element and append issues to *result*.

    Args:
        item: The ``<item>`` ElementTree element.
        index: 1-based position of the item (for error messages).
        is_podcast: Whether the enclosing feed is a podcast feed.
        result: The accumulating validation result to append to.
    """
    prefix = f"Item {index}"

    # --- <title> ---
    title = item.find("title")
    if title is None or not (title.text or "").strip():
        result.errors.append(f"{prefix}: missing or empty <title>")

    # --- <link> ---
    link = item.find("link")
    if link is None or not (link.text or "").strip():
        result.errors.append(f"{prefix}: missing or empty <link>")

    # --- <pubDate> or <dc:date> ---
    pub_date = item.find("pubDate")
    dc_date = item.find(f"{{{_DC_NAMESPACE}}}date")
    if pub_date is None and dc_date is None:
        result.errors.append(f"{prefix}: missing <pubDate> or <dc:date>")

    # --- <guid> ---
    guid = item.find("guid")
    if guid is None or not (guid.text or "").strip():
        result.errors.append(f"{prefix}: missing or empty <guid>")

    # --- Podcast: <enclosure> with url, type, length ---
    if is_podcast:
        enclosure = item.find("enclosure")
        if enclosure is None:
            result.errors.append(f"{prefix}: missing <enclosure> (required for podcast feed)")
        else:
            enc_url = enclosure.get("url")
            enc_type = enclosure.get("type")
            enc_length = enclosure.get("length")

            if not enc_url:
                result.errors.append(f"{prefix}: enclosure missing 'url' attribute")
            if not enc_type:
                result.errors.append(f"{prefix}: enclosure missing 'type' attribute")
            elif enc_type not in PODCAST_MIME_TYPES:
                result.errors.append(
                    f"{prefix}: enclosure type is '{enc_type}', "
                    "expected a supported media MIME type"
                )
            if not enc_length:
                result.errors.append(f"{prefix}: enclosure missing 'length' attribute")
            else:
                try:
                    length_val = int(enc_length)
                    if length_val < 0:
                        result.errors.append(
                            f"{prefix}: enclosure length is negative ({length_val})"
                        )
                except ValueError:
                    result.errors.append(
                        f"{prefix}: enclosure length '{enc_length}' is not a valid integer"
                    )
