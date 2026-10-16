"""Feed generator for local-feed-generator.

Reads JSON metadata files from the data directory, filters out error entries,
and generates RSS 2.0 feeds (podcast.xml and articles.xml) for consumption by
mobile podcast apps (e.g., Antennapod) and RSS readers.

Design references:
    - design.md section 3.4 (Feed Output)
    - design.md section 10.1 (Feed Format: RSS 2.0)
    - tasks.md T4.1-T4.4
    - design.md section 5.2 (Feed Generator responsibilities)

    The feed generator:
    - Scans ``DATA_DIR`` recursively for all ``*.json`` files.
    - Filters out metadata files where ``feed_eligible`` is ``False``
      (i.e. files with incomplete step statuses or error state).
    - Sorts items by folder name (ascending) then filename (ascending).
    - Calculates absolute URLs as ``BASE_URL + filepath``.
    - Writes feeds to ``DATA_DIR/podcast.xml`` and ``DATA_DIR/articles.xml``.
    - Supports per-folder feed generation via ``feed.json`` files.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import html
import json
import logging
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from feedgen.feed import FeedGenerator as _FeedGenerator

from local_feed_generator.config import get_config
from local_feed_generator.metadata import Metadata

_logger = logging.getLogger("local_feed_generator")

# MIME type for podcast enclosure ``type`` attribute (design doc section 3.4).
_PODCAST_MIME_TYPE = "audio/mp4"

# Feed titles for the two generated feeds.
_PODCAST_FEED_TITLE = "Local Podcast Feed"
_ARTICLES_FEED_TITLE = "Local Articles Feed"

# Dublin Core namespace URI (used for <dc:date> elements).
_DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"

# URL schemes allowed in markdown links to prevent XSS (e.g. javascript:).
# Only http, https, mailto, relative paths, and fragment identifiers are allowed.
_SAFE_URL_SCHEMES = re.compile(r"^(https?://|mailto:|/|#|\.\.)")


def _markdown_to_html(text: str | None) -> str:
    """Convert markdown text to HTML with XSS protection.

    Attempts to use the third-party ``markdown`` library if it is installed.
    If the library is not available, falls back to a built-in simple converter
    that handles:

    - **Headers**: ``#`` through ``######`` (h1-h6)
    - **Bold**: ``**text**`` → ``<strong>text</strong>``
    - *Italic*: ``*text*`` or ``_text_`` → ``<em>text</em>``
    - Links: ``[text](url)`` → ``<a href="url">text</a>`` (safe schemes only)
    - Lists: unordered (``-``, ``*``, ``+``) and ordered (``1.``)
    - Paragraphs (double-newline separation) and line breaks (``<br>``)

    All text content is HTML-escaped to prevent XSS vulnerabilities.  URLs
    in links are validated to only allow safe schemes (``http://``,
    ``https://``, ``mailto:``, relative paths, and fragment identifiers).

    Args:
        text: The markdown string to convert, or ``None``.

    Returns:
        An HTML string.  Returns an empty string when *text* is ``None``
        or empty.
    """
    if text is None:
        return ""

    # Try to use the markdown library if available.
    try:
        import markdown as _md  # type: ignore[import-untyped]

        result = _md.markdown(text)
        return str(result)
    except ImportError:
        pass

    # Fall back to the built-in simple converter.
    return _simple_markdown_to_html(text)


def _simple_markdown_to_html(text: str) -> str:
    """Convert markdown to HTML using a simple built-in parser.

    Handles basic markdown elements with proper HTML escaping to prevent
    XSS.  Text is escaped first, then markdown syntax is applied to the
    escaped text so that user-supplied HTML is always rendered as text.

    Args:
        text: The markdown string to convert.

    Returns:
        An HTML string.
    """
    # Escape HTML special characters first to prevent XSS.
    # This ensures any raw HTML in the input is rendered as text.
    text = html.escape(text)

    lines = text.split("\n")
    html_parts: list[str] = []
    in_list: str | None = None  # 'ul', 'ol', or None
    in_paragraph = False
    paragraph_lines: list[str] = []

    def _flush_paragraph() -> None:
        nonlocal in_paragraph, paragraph_lines
        if in_paragraph and paragraph_lines:
            content = "\n".join(paragraph_lines)
            content = _convert_inline_markdown(content)
            # Convert single newlines within a paragraph to <br>.
            content = content.replace("\n", "<br>\n")
            html_parts.append(f"<p>{content}</p>")
        in_paragraph = False
        paragraph_lines = []

    def _flush_list() -> None:
        nonlocal in_list
        if in_list:
            html_parts.append(f"</{in_list}>")
            in_list = None

    for line in lines:
        stripped = line.strip()

        # Empty line — close paragraph and list.
        if not stripped:
            _flush_paragraph()
            _flush_list()
            continue

        # Headers (######, #####, ####, ###, ##, #).
        # Check in reverse order so longer prefixes match first.
        if stripped.startswith("###### "):
            _flush_paragraph()
            _flush_list()
            html_parts.append(f"<h6>{_convert_inline_markdown(stripped[7:])}</h6>")
        elif stripped.startswith("##### "):
            _flush_paragraph()
            _flush_list()
            html_parts.append(f"<h5>{_convert_inline_markdown(stripped[6:])}</h5>")
        elif stripped.startswith("#### "):
            _flush_paragraph()
            _flush_list()
            html_parts.append(f"<h4>{_convert_inline_markdown(stripped[5:])}</h4>")
        elif stripped.startswith("### "):
            _flush_paragraph()
            _flush_list()
            html_parts.append(f"<h3>{_convert_inline_markdown(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            _flush_paragraph()
            _flush_list()
            html_parts.append(f"<h2>{_convert_inline_markdown(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            _flush_paragraph()
            _flush_list()
            html_parts.append(f"<h1>{_convert_inline_markdown(stripped[2:])}</h1>")
        # Unordered list items (-, *, +).
        elif re.match(r"^[-*+]\s", stripped):
            _flush_paragraph()
            if in_list != "ul":
                _flush_list()
                html_parts.append("<ul>")
                in_list = "ul"
            item_content = re.sub(r"^[-*+]\s", "", stripped)
            html_parts.append(f"<li>{_convert_inline_markdown(item_content)}</li>")
        # Ordered list items (1., 2., etc.).
        elif re.match(r"^\d+\.\s", stripped):
            _flush_paragraph()
            if in_list != "ol":
                _flush_list()
                html_parts.append("<ol>")
                in_list = "ol"
            item_content = re.sub(r"^\d+\.\s", "", stripped)
            html_parts.append(f"<li>{_convert_inline_markdown(item_content)}</li>")
        # Regular text — part of a paragraph.
        else:
            if not in_paragraph:
                _flush_list()
                in_paragraph = True
            paragraph_lines.append(line)

    # Close any open tags.
    _flush_paragraph()
    _flush_list()

    return "\n".join(html_parts)


def _convert_inline_markdown(text: str) -> str:
    """Convert inline markdown elements in text.

    Handles ``**bold**``, ``*italic*``, ``_italic_``, and ``[text](url)``
    links.  Text is assumed to be already HTML-escaped.  URLs are validated
    to prevent XSS via ``javascript:`` or other dangerous schemes.

    Args:
        text: HTML-escaped text containing markdown inline syntax.

    Returns:
        Text with inline markdown converted to HTML.
    """
    # **bold** → <strong>bold</strong>
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

    # *italic* → <em>italic</em>
    # Match single asterisks that are not part of ** (already converted).
    text = re.sub(r"(?<!\*)\*(?!\*)([^\*]+?)\*(?!\*)", r"<em>\1</em>", text)

    # _italic_ → <em>italic</em>
    # Match underscores that are not adjacent to word characters on the opening
    # side (to avoid matching things like file_name or 100_thing).  Content
    # can include spaces.
    text = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<em>\1</em>", text)

    # [text](url) → <a href="url">text</a> (only safe URL schemes)
    def _replace_link(match: re.Match[str]) -> str:
        link_text = match.group(1)
        url = match.group(2)
        # Only allow safe URL schemes to prevent XSS.
        if _SAFE_URL_SCHEMES.match(url):
            return f'<a href="{url}">{link_text}</a>'
        # Discard unsafe links but keep the text.
        return link_text

    text = re.sub(r"\[(.+?)\]\((.+?)\)", _replace_link, text)

    return text


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
      ``url``, ``type="audio/mp4"``, and ``length`` attributes.
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
            elif enc_type != _PODCAST_MIME_TYPE:
                result.errors.append(
                    f"{prefix}: enclosure type is '{enc_type}', expected '{_PODCAST_MIME_TYPE}'"
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


class FeedGenerator:
    """Generates RSS 2.0 feeds from JSON metadata files.

    Scans the data directory for ``*.json`` metadata files, filters out
    error entries, and produces two RSS feeds:

    - ``podcast.xml`` — one item per MP4 file with an audio enclosure.
    - ``articles.xml`` — one item per PDF or HTML file.

    Supports per-folder feed generation: when a folder contains a
    ``feed.json`` file, feeds are generated inside that folder using the
    title and description from the ``feed.json``.

    Attributes:
        data_dir: Absolute path to the data directory.
        base_url: Base URL used to construct absolute enclosure/source URLs.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the feed generator.

        Args:
            data_dir: Path to the data directory. When ``None``, the value
                is loaded from the application configuration (``DATA_DIR``).
            base_url: Base URL for constructing absolute URLs. When ``None``,
                the value is loaded from the application configuration
                (``BASE_URL``).
        """
        if data_dir is None or base_url is None:
            config = get_config()
        if data_dir is None:
            data_dir = config.data_dir
        if base_url is None:
            base_url = config.base_url

        self.data_dir: Path = Path(data_dir).resolve()
        self.base_url: str = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Metadata discovery
    # ------------------------------------------------------------------

    def _discover_metadata_files(self) -> list[Metadata]:
        """Recursively discover and load all valid JSON metadata files.

        Walks ``data_dir`` for ``*.json`` files, attempts to parse each as a
        :class:`~local_feed_generator.metadata.Metadata` object, and filters
        out entries where ``feed_eligible`` is ``False`` (i.e. incomplete
        processing or error state).

        Files that cannot be parsed as valid metadata are logged as warnings
        and skipped, so a single corrupt JSON file does not abort feed
        generation.

        Returns:
            A list of :class:`Metadata` instances with ``feed_eligible == True``,
            sorted by folder name (ascending) then filename (ascending).
        """
        if not self.data_dir.exists():
            _logger.warning("Data directory does not exist: %s", self.data_dir)
            return []

        if not self.data_dir.is_dir():
            _logger.warning("Data path is not a directory: %s", self.data_dir)
            return []

        metadata_list: list[Metadata] = []

        for root, _dirs, files in os.walk(self.data_dir):
            root_path = Path(root)
            for filename in files:
                if not filename.lower().endswith(".json"):
                    continue

                json_path = root_path / filename

                # Skip orphaned JSON files (no corresponding source file).
                if self._is_orphaned_json(json_path):
                    _logger.warning(
                        "Skipping orphaned JSON file (no source): %s",
                        json_path,
                    )
                    continue

                try:
                    metadata = Metadata.from_json_file(json_path)
                except Exception as exc:
                    _logger.warning(
                        "Failed to parse metadata file %s: %s",
                        json_path,
                        exc,
                    )
                    continue

                if not metadata.feed_eligible:
                    _logger.debug(
                        "Skipping ineligible metadata file: %s",
                        json_path,
                    )
                    continue

                metadata_list.append(metadata)

        metadata_list.sort(key=self._sort_key)
        return metadata_list

    def _is_orphaned_json(self, json_path: Path) -> bool:
        """Check if a JSON metadata file is orphaned (no corresponding source).

        A JSON file is orphaned when removing the ``.json`` suffix does not
        yield an existing file with a supported source extension.

        Args:
            json_path: Path to the ``.json`` file to check.

        Returns:
            ``True`` if the JSON file is orphaned, ``False`` otherwise.
        """
        # feed.json is a configuration file, not orphaned metadata.
        if json_path.name == "feed.json":
            return False

        source_filename = json_path.name[: -len(".json")]
        source_path = json_path.parent / source_filename

        # The stripped name must have a supported extension to be a real source.
        supported_extensions = (".mp4", ".pdf", ".html")
        if source_path.suffix.lower() not in supported_extensions:
            return True

        return not source_path.exists()

    @staticmethod
    def _sort_key(metadata: Metadata) -> tuple[str, str]:
        """Sort key: folder name (ascending), then filename (ascending).

        The folder name is the parent directory of the file's ``filepath``,
        and the filename is the last path component. This implements the
        sorting requirement from design doc section 3.4:
        "Folder name (ascending), then filename (ascending)".

        Args:
            metadata: A :class:`Metadata` instance.

        Returns:
            A tuple of ``(folder_name, filename)`` for sorting.
        """
        filepath = Path(metadata.filepath)
        folder_name = filepath.parent.name if filepath.parent.name else ""
        filename = filepath.name
        return (folder_name, filename)

    def _absolute_url(self, filepath: str) -> str:
        """Construct an absolute URL from BASE_URL and a relative filepath.

        Args:
            filepath: Relative path from the data directory root
                (e.g., ``"videos/lecture.mp4"``).

        Returns:
            Absolute URL string (e.g., ``"https://example.com/videos/lecture.mp4"``).
        """
        return f"{self.base_url}/{filepath}"

    def _feed_rel_path(self, output_dir: Path, feed_filename: str) -> str:
        """Compute the feed file path relative to ``data_dir``.

        Used for constructing the ``rel="self"`` link in the feed.  For
        root-level feeds this is simply the feed filename (e.g.
        ``"podcast.xml"``).  For per-folder feeds it includes the folder
        prefix (e.g. ``"Course1/podcast.xml"``).

        Args:
            output_dir: The directory where the feed file will be written.
            feed_filename: The feed file name (e.g. ``"podcast.xml"``).

        Returns:
            The feed file path relative to ``data_dir``.
        """
        if output_dir == self.data_dir:
            return feed_filename
        try:
            rel_dir = output_dir.relative_to(self.data_dir)
            return str(rel_dir / feed_filename)
        except ValueError:
            return feed_filename

    # ------------------------------------------------------------------
    # Per-folder feed.json discovery and metadata filtering
    # ------------------------------------------------------------------

    def _discover_feed_json_dirs(self) -> list[tuple[Path, dict[str, str]]]:
        """Walk ``data_dir`` recursively and find all ``feed.json`` files.

        Each ``feed.json`` provides a per-folder feed configuration with
        ``title`` and ``description`` keys.  Folders without a ``feed.json``
        are skipped (root-level feeds use default titles for backward
        compatibility).

        Returns:
            A list of ``(folder_path, config)`` tuples where *folder_path*
            is the directory containing the ``feed.json`` and *config* is
            the parsed JSON content (with at least ``title`` and
            ``description`` keys).  Invalid or unreadable ``feed.json``
            files are logged as warnings and skipped.
        """
        results: list[tuple[Path, dict[str, str]]] = []

        for feed_json_path in self.data_dir.rglob("feed.json"):
            try:
                config: dict[str, str] = json.loads(feed_json_path.read_text(encoding="utf-8"))
                results.append((feed_json_path.parent, config))
            except (json.JSONDecodeError, OSError) as exc:
                _logger.warning("Failed to parse feed.json at %s: %s", feed_json_path, exc)

        return results

    def _filter_metadata_for_folder(
        self,
        metadata_list: list[Metadata],
        folder: Path,
    ) -> list[Metadata]:
        """Filter a metadata list to only include items within *folder*.

        The ``filepath`` field on each :class:`Metadata` is relative to
        ``self.data_dir``.  This method returns only those items whose
        filepath starts with the folder's relative path.

        Args:
            metadata_list: Full list of discovered metadata items.
            folder: Absolute path to the folder to filter by.

        Returns:
            A filtered list containing only metadata items whose source
            files reside within *folder* (or *folder* itself for root-level).
        """
        try:
            rel_folder = folder.relative_to(self.data_dir)
        except ValueError:
            # Folder is not within data_dir — return empty.
            return []

        folder_str = str(rel_folder)
        if folder_str == ".":
            # Root data directory — include everything.
            return list(metadata_list)

        prefix = folder_str + "/"
        return [m for m in metadata_list if m.filepath.startswith(prefix)]

    # ------------------------------------------------------------------
    # Podcast feed (MP4 files)
    # ------------------------------------------------------------------

    def generate_podcast_feed(
        self,
        output_dir: Path | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata_items: list[Metadata] | None = None,
    ) -> list[Metadata]:
        """Generate the podcast RSS feed (``podcast.xml``).

        Uses the ``feedgen`` library with the podcast extension to produce
        a valid RSS 2.0 feed with iTunes podcast extensions. Only MP4 files
        with ``feed_eligible == True`` are included.

        For each item:
            - **title**: from metadata ``final_title`` computed property.
            - **description**: from metadata ``summary`` (LLM-generated key
              takeaways for MP4 files), falling back to ``description``
              for backward compatibility.  Markdown is converted to HTML.
            - **link**: absolute URL to the source MP4 file.
            - **enclosure**: URL, type ``audio/mp4``, and length (file size).
            - **publication date**: from metadata ``date``.
            - **GUID**: based on the filepath (as a permalink URL).

        The feed is written to ``output_dir/podcast.xml``, overwriting any
        existing file (idempotent regeneration).

        Args:
            output_dir: Directory to write the feed file.  Defaults to
                ``self.data_dir``.
            title: Feed title.  Defaults to ``_PODCAST_FEED_TITLE``.
            description: Feed description.  Defaults to ``_PODCAST_FEED_TITLE``.
            metadata_items: Pre-discovered metadata items to use instead of
                discovering them.  When ``None`` (default), metadata is
                discovered via :meth:`_discover_metadata_files`.

        Returns:
            The list of :class:`Metadata` items included in the podcast feed.

        Raises:
            OSError: If the feed file cannot be written.
        """
        if output_dir is None:
            output_dir = self.data_dir
        if title is None:
            title = _PODCAST_FEED_TITLE
        if description is None:
            description = _PODCAST_FEED_TITLE

        if metadata_items is None:
            all_metadata = self._discover_metadata_files()
            mp4_items = [m for m in all_metadata if m.filetype == "mp4"]
        else:
            mp4_items = [m for m in metadata_items if m.filetype == "mp4"]

        if not mp4_items:
            _logger.info("No MP4 files found; skipping podcast feed generation for %s", output_dir)
            return []

        fg = _FeedGenerator()
        fg.load_extension("podcast")

        # Compute the feed file path relative to data_dir for the self-link.
        feed_rel_path = self._feed_rel_path(output_dir, "podcast.xml")

        # Feed-level metadata
        fg.title(title)
        fg.link(href=self._absolute_url(feed_rel_path), rel="self")
        fg.link(href=self.base_url)
        fg.description(description)
        fg.language("en")

        for metadata in mp4_items:
            entry = fg.add_entry()
            entry.id(str(metadata.filepath))
            entry.guid(self._absolute_url(metadata.filepath))
            entry.title(metadata.final_title)

            # Description from summary field (holds LLM key takeaways for MP4),
            # falling back to description for backward compatibility.
            # Convert markdown to HTML for safe rendering in feed readers.
            desc_text = metadata.summary or metadata.description or ""
            entry.description(_markdown_to_html(desc_text))

            # Link to the source file.
            source_url = self._absolute_url(metadata.filepath)
            entry.link(href=source_url)

            # Enclosure with URL, type, and length.
            entry.enclosure(
                url=source_url,
                length=metadata.size,
                type=_PODCAST_MIME_TYPE,
            )

            # Publication date.
            pub_date = (
                datetime.fromtimestamp(metadata.date, tz=UTC)
                if metadata.date
                else datetime.now(UTC)
            )
            entry.pubDate(pub_date)

        output_path = output_dir / "podcast.xml"
        fg.rss_file(str(output_path), pretty=True)
        _logger.info("Generated podcast feed: %s (%d items)", output_path, len(mp4_items))

        return mp4_items

    # ------------------------------------------------------------------
    # Articles feed (PDF/HTML files)
    # ------------------------------------------------------------------

    def generate_articles_feed(
        self,
        output_dir: Path | None = None,
        title: str | None = None,
        description: str | None = None,
        metadata_items: list[Metadata] | None = None,
    ) -> list[Metadata]:
        """Generate the articles RSS feed (``articles.xml``).

        Uses the ``feedgen`` library to produce a standard RSS 2.0 feed.
        Only PDF and HTML files with ``feed_eligible == True`` are included.

        For each item:
            - **title**: from metadata ``final_title`` computed property.
            - **description**: from metadata ``content`` (extracted text) if
              available, otherwise from ``description``, otherwise empty.
              Markdown is converted to HTML.
            - **link**: absolute URL to the source file.
            - **publication date**: from metadata ``date``.
            - **GUID**: based on the filepath (as a permalink URL).

        The feed is written to ``output_dir/articles.xml``, overwriting any
        existing file (idempotent regeneration).

        Args:
            output_dir: Directory to write the feed file.  Defaults to
                ``self.data_dir``.
            title: Feed title.  Defaults to ``_ARTICLES_FEED_TITLE``.
            description: Feed description.  Defaults to
                ``_ARTICLES_FEED_TITLE``.
            metadata_items: Pre-discovered metadata items to use instead of
                discovering them.  When ``None`` (default), metadata is
                discovered via :meth:`_discover_metadata_files`.

        Returns:
            The list of :class:`Metadata` items included in the articles feed.

        Raises:
            OSError: If the feed file cannot be written.
        """
        if output_dir is None:
            output_dir = self.data_dir
        if title is None:
            title = _ARTICLES_FEED_TITLE
        if description is None:
            description = _ARTICLES_FEED_TITLE

        if metadata_items is None:
            all_metadata = self._discover_metadata_files()
            article_items = [m for m in all_metadata if m.filetype in ("pdf", "html")]
        else:
            article_items = [m for m in metadata_items if m.filetype in ("pdf", "html")]

        if not article_items:
            _logger.info(
                "No PDF/HTML files found; skipping articles feed generation for %s",
                output_dir,
            )
            return []

        fg = _FeedGenerator()

        # Compute the feed file path relative to data_dir for the self-link.
        feed_rel_path = self._feed_rel_path(output_dir, "articles.xml")

        # Feed-level metadata
        fg.title(title)
        fg.link(href=self._absolute_url(feed_rel_path), rel="self")
        fg.link(href=self.base_url)
        fg.description(description)
        fg.language("en")

        for metadata in article_items:
            entry = fg.add_entry()
            entry.id(str(metadata.filepath))
            entry.guid(self._absolute_url(metadata.filepath))
            entry.title(metadata.final_title)

            # Description from content (if available), fallback to
            # description field, then empty string.
            # Convert markdown to HTML for safe rendering in feed readers.
            desc_text = metadata.content or metadata.description or ""
            entry.description(_markdown_to_html(desc_text))

            # Link to the source file.
            source_url = self._absolute_url(metadata.filepath)
            entry.link(href=source_url)

            # Publication date.
            pub_date = (
                datetime.fromtimestamp(metadata.date, tz=UTC)
                if metadata.date
                else datetime.now(UTC)
            )
            entry.pubDate(pub_date)

        output_path = output_dir / "articles.xml"
        fg.rss_file(str(output_path), pretty=True)
        _logger.info(
            "Generated articles feed: %s (%d items)",
            output_path,
            len(article_items),
        )

        return article_items

    # ------------------------------------------------------------------
    # Combined feed generation
    # ------------------------------------------------------------------

    def _generate_feeds_for_folder(
        self,
        folder: Path,
        title: str | None,
        description: str | None,
        metadata_items: list[Metadata],
    ) -> dict[str, list[Metadata]]:
        """Generate podcast and/or articles feeds for a specific folder.

        This method is used for per-folder feed generation when a folder
        contains a ``feed.json`` file.  It generates:

        - ``podcast.xml`` — only if MP4 files are present in the folder's
          subtree.
        - ``articles.xml`` — only if PDF or HTML files are present in the
          folder's subtree.

        Feeds are written to *folder*.  If generation of either feed fails,
        the error is logged and the other feed continues to be generated.

        Args:
            folder: Absolute path to the folder where feeds will be written.
            title: Feed title from ``feed.json``, or ``None`` to use the
                default title for the feed type.
            description: Feed description from ``feed.json``, or ``None`` to
                use the default description for the feed type.
            metadata_items: Pre-discovered metadata items to use for this
                folder.

        Returns:
            A dictionary with keys ``"podcast"`` and ``"articles"``, each
            mapping to the list of :class:`Metadata` items included in the
            respective feed.  If a feed was not generated (no eligible files
            of that type or generation failed), its value will be an empty
            list.
        """
        results: dict[str, list[Metadata]] = {
            "podcast": [],
            "articles": [],
        }

        mp4_items = [m for m in metadata_items if m.filetype == "mp4"]
        article_items = [m for m in metadata_items if m.filetype in ("pdf", "html")]

        # Generate podcast.xml only if MP4 files are present.
        if mp4_items:
            try:
                results["podcast"] = self.generate_podcast_feed(
                    output_dir=folder,
                    title=title,
                    description=description,
                    metadata_items=metadata_items,
                )
            except Exception as exc:
                _logger.error(
                    "Failed to generate podcast feed for %s: %s",
                    folder,
                    exc,
                )

        # Generate articles.xml only if PDF/HTML files are present.
        if article_items:
            try:
                results["articles"] = self.generate_articles_feed(
                    output_dir=folder,
                    title=title,
                    description=description,
                    metadata_items=metadata_items,
                )
            except Exception as exc:
                _logger.error(
                    "Failed to generate articles feed for %s: %s",
                    folder,
                    exc,
                )

        return results

    def generate_all_feeds(self, validate: bool = False) -> dict[str, list[Metadata]]:
        """Generate root-level and per-folder RSS feeds.

        This method performs two phases of feed generation:

        1. **Root-level feeds**: Generates ``podcast.xml`` and
           ``articles.xml`` in ``data_dir``.  Root-level feeds are only
           generated if a ``data_dir/feed.json`` file exists; its
           ``title`` and ``description`` are used for the root-level feeds.
           ``podcast.xml`` is only created if MP4 files are present;
           ``articles.xml`` is only created if PDF/HTML files are present.

        2. **Per-folder feeds**: Walks ``data_dir`` recursively for
           ``feed.json`` files.  For each folder containing a
           ``feed.json``, generates ``podcast.xml`` (only if MP4 files
           are present in that folder tree) and ``articles.xml`` (only
           if PDF/HTML files are present).  Feeds are written to the
           folder containing the ``feed.json``.

        If any individual feed generation fails, the error is logged and
        other feeds continue to be generated.

        When *validate* is ``True``, :meth:`validate_output` is called after
        generation and any validation errors or warnings are logged.  Note
        that validation currently only covers root-level feeds.

        This method is **idempotent**: feeds are always regenerated from the
        current metadata, overwriting any existing feed files.

        Args:
            validate: When ``True``, validate the generated root-level feeds
                and log any errors or warnings found.

        Returns:
            A dictionary with keys ``"podcast"`` and ``"articles"``, each
            mapping to the list of :class:`Metadata` items included in the
            respective root-level feed.  If a feed generation failed, its
            value will be an empty list.
        """
        results: dict[str, list[Metadata]] = {
            "podcast": [],
            "articles": [],
        }

        # --- Phase 1: Root-level feeds (only if feed.json exists) ---
        root_feed_json = self.data_dir / "feed.json"

        # Only generate root-level feeds if feed.json exists
        if root_feed_json.exists():
            root_title: str | None = None
            root_description: str | None = None
            try:
                root_config = json.loads(root_feed_json.read_text(encoding="utf-8"))
                root_title = root_config.get("title")
                root_description = root_config.get("description")
            except (json.JSONDecodeError, OSError) as exc:
                _logger.warning(
                    "Failed to parse root feed.json at %s: %s",
                    root_feed_json,
                    exc,
                )
                root_title = None
                root_description = None

            # Generate podcast feed (only if MP4 files exist - handled in generate_podcast_feed)
            try:
                results["podcast"] = self.generate_podcast_feed(
                    output_dir=self.data_dir,
                    title=root_title,
                    description=root_description,
                )
            except Exception as exc:
                _logger.error("Failed to generate podcast feed: %s", exc)

            # Generate articles feed (only if PDF/HTML files exist)
            try:
                results["articles"] = self.generate_articles_feed(
                    output_dir=self.data_dir,
                    title=root_title,
                    description=root_description,
                )
            except Exception as exc:
                _logger.error("Failed to generate articles feed: %s", exc)
        else:
            _logger.info("No feed.json at %s; skipping root-level feed generation", self.data_dir)

        # --- Phase 2: Per-folder feeds ---
        # Walk for feed.json files in subdirectories (skip root data_dir
        # which is already handled above).
        all_metadata = self._discover_metadata_files()

        for folder_path, config in self._discover_feed_json_dirs():
            # Skip root data_dir (already handled above).
            if folder_path == self.data_dir:
                continue

            # Use None as default so the individual feed methods fall back
            # to their own type-specific default titles/descriptions.
            folder_title = config.get("title")
            folder_description = config.get("description")

            # Filter metadata to only items within this folder tree.
            folder_metadata = self._filter_metadata_for_folder(all_metadata, folder_path)

            self._generate_feeds_for_folder(
                folder=folder_path,
                title=folder_title,
                description=folder_description,
                metadata_items=folder_metadata,
            )

        # Generate index.html listing all feeds.
        try:
            self._generate_index_html()
        except Exception as exc:
            _logger.error("Failed to generate index.html: %s", exc)

        # Optionally validate the generated feeds.
        if validate:
            self._log_validation_results(self.validate_output())

        return results

    # ------------------------------------------------------------------
    # Feed validation (T7.2)
    # ------------------------------------------------------------------

    def validate_output(self) -> dict[str, FeedValidationResult]:
        """Validate both ``podcast.xml`` and ``articles.xml``.

        Reads each feed file from ``data_dir`` and runs
        :func:`validate_feed` on its content.  Podcast feeds are checked
        for enclosure tags; article feeds are not.

        Returns:
            A dictionary with keys ``"podcast"`` and ``"articles"``, each
            mapping to a :class:`FeedValidationResult`.  If a feed file
            does not exist, the result will contain an error indicating
            the missing file.
        """
        results: dict[str, FeedValidationResult] = {}

        # --- podcast.xml ---
        podcast_path = self.data_dir / "podcast.xml"
        if podcast_path.exists():
            xml_content = podcast_path.read_text(encoding="utf-8")
            results["podcast"] = validate_feed(
                xml_content,
                feed_name="podcast.xml",
                is_podcast=True,
            )
        else:
            results["podcast"] = FeedValidationResult(
                feed_name="podcast.xml",
                valid=False,
                errors=["Feed file does not exist: podcast.xml"],
            )

        # --- articles.xml ---
        articles_path = self.data_dir / "articles.xml"
        if articles_path.exists():
            xml_content = articles_path.read_text(encoding="utf-8")
            results["articles"] = validate_feed(
                xml_content,
                feed_name="articles.xml",
                is_podcast=False,
            )
        else:
            results["articles"] = FeedValidationResult(
                feed_name="articles.xml",
                valid=False,
                errors=["Feed file does not exist: articles.xml"],
            )

        return results

    @staticmethod
    def _log_validation_results(
        results: dict[str, FeedValidationResult],
    ) -> None:
        """Log validation results for each feed.

        Args:
            results: Mapping of feed name to :class:`FeedValidationResult`.
        """
        for name, result in results.items():
            for error in result.errors:
                _logger.error("Feed validation error [%s]: %s", name, error)
            for warning in result.warnings:
                _logger.warning("Feed validation warning [%s]: %s", name, warning)
            if result.is_valid:
                _logger.info("Feed validation passed [%s]", name)

    # ------------------------------------------------------------------
    # Index HTML generation
    # ------------------------------------------------------------------

    def _generate_index_html(self) -> None:
        """Generate or update an ``index.html`` listing all XML feed files.

        Discovers all ``*.xml`` files in the data directory tree using
        :meth:`Path.rglob`, categorizes them by feed type, and writes an HTML
        index file to ``data_dir / index.html``.

        The index includes:
        - A title "Feed Index".
        - Sections organized by feed type (Podcast Feeds, Articles Feeds,
          and an "Other Feeds" section for any additional XML files).
        - Links using absolute URLs (``BASE_URL + relative path``).
        - HTML-escaped filenames and URLs for special characters.

        If no XML feeds are found, an index with empty sections is still
        generated.  An existing ``index.html`` is overwritten.

        Raises:
            OSError: If the index file cannot be written.
        """
        # Check if data directory exists.
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            _logger.warning(
                "Data directory does not exist, skipping index.html generation: %s",
                self.data_dir,
            )
            return

        # Discover all XML feed files in the data directory tree.
        xml_files = sorted(self.data_dir.rglob("*.xml"))

        # Categorize feeds by type based on filename.
        podcast_feeds: list[tuple[str, str]] = []
        articles_feeds: list[tuple[str, str]] = []
        other_feeds: list[tuple[str, str]] = []

        for xml_path in xml_files:
            rel_path = str(xml_path.relative_to(self.data_dir))
            abs_url = self._absolute_url(rel_path)
            feed_name = xml_path.name

            if feed_name == "podcast.xml":
                podcast_feeds.append((rel_path, abs_url))
            elif feed_name == "articles.xml":
                articles_feeds.append((rel_path, abs_url))
            else:
                other_feeds.append((rel_path, abs_url))

        # Sort each category by relative path for deterministic output.
        podcast_feeds.sort(key=lambda x: x[0])
        articles_feeds.sort(key=lambda x: x[0])
        other_feeds.sort(key=lambda x: x[0])

        # Build HTML content.
        lines: list[str] = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append('    <meta charset="utf-8">')
        lines.append("    <title>Feed Index</title>")
        lines.append("    <style>")
        lines.append("        body { font-family: Arial, sans-serif; margin: 20px; }")
        lines.append("        h1 { color: #333; }")
        lines.append("        ul { list-style: none; padding: 0; }")
        lines.append("        li { margin: 10px 0; }")
        lines.append("        a { color: #0066cc; }")
        lines.append("    </style>")
        lines.append("</head>")
        lines.append("<body>")
        lines.append("    <h1>Feed Index</h1>")

        # Podcast Feeds section.
        lines.append("    <h2>Podcast Feeds</h2>")
        lines.append("    <ul>")
        for rel_path, abs_url in podcast_feeds:
            escaped_path = html.escape(rel_path)
            escaped_url = html.escape(abs_url)
            lines.append(f'        <li><a href="{escaped_url}">{escaped_path}</a></li>')
        lines.append("    </ul>")

        # Articles Feeds section.
        lines.append("    <h2>Articles Feeds</h2>")
        lines.append("    <ul>")
        for rel_path, abs_url in articles_feeds:
            escaped_path = html.escape(rel_path)
            escaped_url = html.escape(abs_url)
            lines.append(f'        <li><a href="{escaped_url}">{escaped_path}</a></li>')
        lines.append("    </ul>")

        # Other feeds section (only if there are non-standard XML files).
        if other_feeds:
            lines.append("    <h2>Other Feeds</h2>")
            lines.append("    <ul>")
            for rel_path, abs_url in other_feeds:
                escaped_path = html.escape(rel_path)
                escaped_url = html.escape(abs_url)
                lines.append(f'        <li><a href="{escaped_url}">{escaped_path}</a></li>')
            lines.append("    </ul>")

        lines.append("</body>")
        lines.append("</html>")

        html_content = "\n".join(lines)
        index_path = self.data_dir / "index.html"
        index_path.write_text(html_content, encoding="utf-8")
        _logger.info(
            "Generated index.html: %s (%d feeds)",
            index_path,
            len(xml_files),
        )
