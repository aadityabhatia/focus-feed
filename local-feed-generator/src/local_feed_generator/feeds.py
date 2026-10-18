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
    - Scans ``data_dir`` recursively for all ``*.json`` files.
    - Filters out metadata files where ``feed_eligible`` is ``False``
      (i.e. files with incomplete step statuses or error state).
    - Sorts items by folder name (ascending) then filename (ascending).
    - Calculates absolute URLs as ``base_url + filepath``.
    - Writes feeds to ``data_dir/podcast.xml`` and ``data_dir/articles.xml``.
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
import markdown as _md

from local_feed_generator.config import get_config
from local_feed_generator.metadata import Metadata
from local_feed_generator.producer import _compute_file_hash

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

    Args:
        text: The markdown string to convert, or ``None``.

    Returns:
        An HTML string.  Returns an empty string when *text* is ``None``
        or empty.
    """
    if text is None:
        return ""

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


def _get_summarization_attribution(metadata: Metadata) -> str:
    """Generate attribution string for feed item descriptions.

    Returns a string like "Summarized by {model} at {datetime}" in local timezone,
    or empty string if attribution cannot be determined.

    Since all feed-eligible files now require ``summarize`` to be complete,
    this always uses ``summarize.model`` and ``summarize.end_time``.
    """
    # Get config for llm_model fallback; return empty string if config unavailable.
    try:
        config = get_config()
        llm_model = config.llm_model
    except Exception:
        llm_model = None

    # All feed-eligible files have summarize complete, so always use it.
    step = metadata.summarize
    model = step.model if step else None

    # Fall back to config llm_model if model is still None.
    if not model:
        model = llm_model

    # If no model could be determined, return empty string.
    if not model:
        return ""

    # Extract timestamp: prefer step.end_time, then metadata.date, then publication_date.
    if step is not None and step.end_time is not None:
        timestamp = step.end_time
    elif metadata.date is not None:
        timestamp = metadata.date
    elif metadata.publication_date is not None:
        timestamp = metadata.publication_date
    else:
        return ""

    # Convert timestamp to local timezone and format.
    dt = datetime.fromtimestamp(timestamp, tz=UTC).astimezone()
    formatted_dt = dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    return f"Summarized by {model} at {formatted_dt}"


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
        data_dir: str | Path,
        base_url: str | None = None,
    ) -> None:
        """Initialize the feed generator.

        Args:
            data_dir: Path to the data directory. This is a required parameter;
                the value is no longer loaded from the application configuration.
            base_url: Base URL for constructing absolute URLs. When ``None``,
                the value is loaded from the application configuration
                (``base_url``).
        """
        if base_url is None:
            config = get_config()
            base_url = config.base_url

        self.data_dir: Path = Path(data_dir).resolve()
        self.base_url: str = base_url.rstrip("/")
        self.data_dir_basename: str = self.data_dir.name

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
        total_count = 0

        _logger.debug("Starting metadata discovery in %s", self.data_dir)

        for root, _dirs, files in os.walk(self.data_dir):
            root_path = Path(root)
            for filename in files:
                if not filename.lower().endswith(".json"):
                    continue

                total_count += 1
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
        _logger.info(
            "Discovered %d metadata files (%d feed-eligible)",
            total_count,
            len(metadata_list),
        )
        return metadata_list

    def _update_missing_hashes(self) -> int:
        """Scan all JSON metadata files and add missing hash fields.

        Walks ``data_dir`` recursively for ``*.json`` files (excluding
        ``feed.json`` configuration files).  For each metadata file that:

        - Has a corresponding source file that exists (not orphaned), **and**
        - Does **not** already have a non-empty ``hash`` field,

        computes the SHA256 hash of the source file and writes it back to
        the JSON file.  This ensures all feed items can use hash-based
        guids, eliminating legacy files without hashes.

        Files that cannot be parsed as valid JSON, or whose source file
        cannot be read, are logged as warnings and skipped so that a single
        corrupt file does not abort the update pass.

        Returns:
            The number of JSON files that were updated with a hash.
        """
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            return 0

        updated_count = 0

        for root, _dirs, files in os.walk(self.data_dir):
            root_path = Path(root)
            for filename in files:
                if not filename.lower().endswith(".json"):
                    continue
                if filename == "feed.json":
                    continue

                json_path = root_path / filename

                # Skip orphaned JSON files (no corresponding source file).
                if self._is_orphaned_json(json_path):
                    continue

                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    _logger.warning(
                        "Failed to read JSON file %s during hash update: %s",
                        json_path,
                        exc,
                    )
                    continue

                # Skip if hash is already present and non-empty.
                if data.get("hash"):
                    continue

                # Determine the source file path (strip .json suffix).
                source_filename = filename[: -len(".json")]
                source_path = root_path / source_filename

                # Compute the SHA256 hash of the source file.
                file_hash = _compute_file_hash(str(source_path))

                if not file_hash:
                    _logger.warning(
                        "Failed to compute hash for %s; skipping hash update",
                        source_path,
                    )
                    continue

                # Update the JSON dict with the hash.
                data["hash"] = file_hash

                # Write back atomically (write to temp file, then rename).
                try:
                    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
                    tmp_path.write_text(
                        json.dumps(data, indent=2),
                        encoding="utf-8",
                    )
                    tmp_path.replace(json_path)
                    updated_count += 1
                    _logger.info("Added hash to %s", json_path)
                except OSError as exc:
                    _logger.warning(
                        "Failed to write updated JSON to %s: %s",
                        json_path,
                        exc,
                    )

        return updated_count

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
        """Construct an absolute URL from ``base_url`` and a relative filepath.

        ``base_url`` is expected to already point to where ``data_dir`` is served, so
        the ``data_dir`` basename is NOT prepended to the filepath.

        Args:
            filepath: Relative path from the data directory root
                (e.g., ``"videos/lecture.mp4"``).

        Returns:
            Absolute URL string (e.g.,
            ``"https://example.com/videos/lecture.mp4"``).
        """
        return f"{self.base_url}/{filepath}"

    def _compute_guid(self, metadata: Metadata) -> str:
        """Compute the GUID for a feed item.

        When the ``hash`` field is present in the metadata, the last 12
        characters of the hash value are used as the guid.  Otherwise, the
        file path is used as a fallback (design doc section 3.4, FR-020).

        Args:
            metadata: The :class:`Metadata` instance for the feed item.

        Returns:
            The guid string — either the last 12 chars of the hash, or the
            absolute URL constructed from the filepath.
        """
        if metadata.hash:
            return metadata.hash[-12:]
        return self._absolute_url(metadata.filepath)

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
        link: str | None = None,
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
            - **GUID**: last 12 characters of the ``hash`` field when present,
              otherwise the absolute URL of the file path (FR-020).

        The feed is written to ``output_dir/podcast.xml``, overwriting any
        existing file (idempotent regeneration).

        Args:
            output_dir: Directory to write the feed file.  Defaults to
                ``self.data_dir``.
            title: Feed title.  Defaults to ``_PODCAST_FEED_TITLE``.
            description: Feed description.  Defaults to ``_PODCAST_FEED_TITLE``.
            link: Feed-level channel link URL.  When ``None`` (default),
                ``self.base_url`` is used.  If provided (e.g. from
                ``feed.json``), it overrides the default.
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
        if link is None:
            link = self.base_url

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
        fg.link(href=link)
        fg.description(description)
        fg.language("en")

        for metadata in mp4_items:
            entry = fg.add_entry()
            entry.id(str(metadata.filepath))
            entry.guid(self._compute_guid(metadata))
            entry.title(metadata.final_title)

            # Description from summary field (holds LLM key takeaways for MP4),
            # falling back to description for backward compatibility.
            # Convert markdown to HTML for safe rendering in feed readers.
            desc_text = metadata.summary or metadata.description or ""
            # Append summarization attribution before HTML conversion.
            attribution = _get_summarization_attribution(metadata)
            if attribution:
                desc_text = f"{desc_text}\n\n{attribution}" if desc_text else attribution
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
        link: str | None = None,
        metadata_items: list[Metadata] | None = None,
    ) -> list[Metadata]:
        """Generate the articles RSS feed (``articles.xml``).

        Uses the ``feedgen`` library to produce a standard RSS 2.0 feed.
        Only PDF and HTML files with ``feed_eligible == True`` are included.

        For each item:
            - **title**: from metadata ``final_title`` computed property.
            - **description**: from metadata ``summary`` if available,
              otherwise from ``description``, otherwise empty.
              Markdown is converted to HTML.
            - **link**: absolute URL to the source file.
            - **publication date**: from metadata ``publication_date``
              (document metadata date) if available, otherwise from
              ``date`` (file modification time).
            - **GUID**: last 12 characters of the ``hash`` field when present,
              otherwise the absolute URL of the file path (FR-020).

        The feed is written to ``output_dir/articles.xml``, overwriting any
        existing file (idempotent regeneration).

        Args:
            output_dir: Directory to write the feed file.  Defaults to
                ``self.data_dir``.
            title: Feed title.  Defaults to ``_ARTICLES_FEED_TITLE``.
            description: Feed description.  Defaults to
                ``_ARTICLES_FEED_TITLE``.
            link: Feed-level channel link URL.  When ``None`` (default),
                ``self.base_url`` is used.  If provided (e.g. from
                ``feed.json``), it overrides the default.
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
        if link is None:
            link = self.base_url

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
        fg.link(href=link)
        fg.description(description)
        fg.language("en")

        for metadata in article_items:
            entry = fg.add_entry()
            entry.id(str(metadata.filepath))
            entry.guid(self._compute_guid(metadata))
            entry.title(metadata.final_title)

            # Description from summary (if available), fallback to
            # description field, then empty string.
            # Convert markdown to HTML for safe rendering in feed readers.
            desc_text = metadata.summary or metadata.description or ""
            # Append summarization attribution before HTML conversion.
            attribution = _get_summarization_attribution(metadata)
            if attribution:
                desc_text = f"{desc_text}\n\n{attribution}" if desc_text else attribution
            entry.description(_markdown_to_html(desc_text))

            # Link to the source file.
            source_url = self._absolute_url(metadata.filepath)
            entry.link(href=source_url)

            # Publication date: use document metadata date if available,
            # falling back to file modification date.
            pub_date_value = metadata.publication_date or metadata.date
            pub_date = (
                datetime.fromtimestamp(pub_date_value, tz=UTC)
                if pub_date_value
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
        link: str | None,
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
            link: Feed-level channel link URL from ``feed.json``, or ``None``
                to use ``self.base_url`` as the default.
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

        _logger.info("Generating feeds for folder: %s", folder)

        # Generate podcast.xml only if MP4 files are present.
        if mp4_items:
            try:
                results["podcast"] = self.generate_podcast_feed(
                    output_dir=folder,
                    title=title,
                    description=description,
                    link=link,
                    metadata_items=metadata_items,
                )
            except Exception as exc:
                _logger.error(
                    "Failed to generate podcast feed for %s: %s",
                    folder,
                    exc,
                )
        else:
            _logger.debug("Skipping podcast feed for %s: no MP4 items", folder)

        # Generate articles.xml only if PDF/HTML files are present.
        if article_items:
            try:
                results["articles"] = self.generate_articles_feed(
                    output_dir=folder,
                    title=title,
                    description=description,
                    link=link,
                    metadata_items=metadata_items,
                )
            except Exception as exc:
                _logger.error(
                    "Failed to generate articles feed for %s: %s",
                    folder,
                    exc,
                )
        else:
            _logger.debug("Skipping articles feed for %s: no PDF/HTML items", folder)

        _logger.info(
            "Feed generation for %s complete: %d podcast, %d articles",
            folder,
            len(results["podcast"]),
            len(results["articles"]),
        )
        return results

    def generate_all_feeds(self, validate: bool = False) -> dict[str, list[Metadata]]:
        """Generate root-level and per-folder RSS feeds.

        This method performs three phases:

         0. **Hash update**: Scans all JSON metadata files for missing
            ``hash`` fields and computes the SHA256 of the corresponding
            source file, writing the hash back to disk.  This ensures all
            feed items use hash-based guids (FR-020).

          1. **Root-level feeds**: Generates ``podcast.xml`` and
             ``articles.xml`` in ``data_dir``.  Root-level feeds are only
             generated if a ``data_dir/feed.json`` file exists; its
             ``title``, ``description``, and optional ``link`` are used
             for the root-level feeds.  When ``link`` is omitted in
              ``feed.json``, the channel link defaults to ``base_url``.
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

        # --- Phase 0: Update missing hash fields in JSON metadata ---
        # Before generating feeds, ensure all existing JSON metadata files
        # have a hash field by computing the SHA256 of the source file.
        # This ensures all feed items use hash-based guids.
        self._update_missing_hashes()

        _logger.info("Starting feed generation for all folders")

        # --- Phase 1: Root-level feeds (only if feed.json exists) ---
        root_feed_json = self.data_dir / "feed.json"

        _logger.info(
            "Root feed.json %s; %s root-level feed generation",
            root_feed_json,
            "starting" if root_feed_json.exists() else "skipping",
        )

        # Only generate root-level feeds if feed.json exists
        if root_feed_json.exists():
            root_title: str | None = None
            root_description: str | None = None
            root_link: str | None = None
            try:
                root_config = json.loads(root_feed_json.read_text(encoding="utf-8"))
                root_title = root_config.get("title")
                root_description = root_config.get("description")
                root_link = root_config.get("link")
            except (json.JSONDecodeError, OSError) as exc:
                _logger.warning(
                    "Failed to parse root feed.json at %s: %s",
                    root_feed_json,
                    exc,
                )
                root_title = None
                root_description = None
                root_link = None

            # Generate podcast feed (only if MP4 files exist - handled in generate_podcast_feed)
            try:
                results["podcast"] = self.generate_podcast_feed(
                    output_dir=self.data_dir,
                    title=root_title,
                    description=root_description,
                    link=root_link,
                )
            except Exception as exc:
                _logger.error("Failed to generate podcast feed: %s", exc)

            # Generate articles feed (only if PDF/HTML files exist)
            try:
                results["articles"] = self.generate_articles_feed(
                    output_dir=self.data_dir,
                    title=root_title,
                    description=root_description,
                    link=root_link,
                )
            except Exception as exc:
                _logger.error("Failed to generate articles feed: %s", exc)
        else:
            _logger.info("No feed.json at %s; skipping root-level feed generation", self.data_dir)
            # Clean up any stale root-level feed files from a previous run
            # when no feed.json exists at the data directory root.
            for stale_feed in ("podcast.xml", "articles.xml"):
                stale_path = self.data_dir / stale_feed
                if stale_path.exists():
                    try:
                        stale_path.unlink()
                        _logger.info("Removed stale root-level feed: %s", stale_path)
                    except OSError as exc:
                        _logger.warning("Failed to remove stale feed %s: %s", stale_path, exc)

        # --- Phase 2: Per-folder feeds ---
        # Walk for feed.json files in subdirectories (skip root data_dir
        # which is already handled above).
        all_metadata = self._discover_metadata_files()

        for folder_path, config in self._discover_feed_json_dirs():
            _logger.info("Generating feeds for folder: %s", folder_path)
            # Skip root data_dir (already handled above).
            if folder_path == self.data_dir:
                continue

            # Use None as default so the individual feed methods fall back
            # to their own type-specific default titles/descriptions.
            folder_title = config.get("title")
            folder_description = config.get("description")
            folder_link = config.get("link")

            # Filter metadata to only items within this folder tree.
            folder_metadata = self._filter_metadata_for_folder(all_metadata, folder_path)

            self._generate_feeds_for_folder(
                folder=folder_path,
                title=folder_title,
                description=folder_description,
                link=folder_link,
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

        _logger.info(
            "Feed generation complete: %d podcast items, %d article items",
            len(results["podcast"]),
            len(results["articles"]),
        )
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
        - Links using absolute URLs (``base_url + relative path``).
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
