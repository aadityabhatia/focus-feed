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
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
import os
from pathlib import Path
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

    # ------------------------------------------------------------------
    # Podcast feed (MP4 files)
    # ------------------------------------------------------------------

    def generate_podcast_feed(self) -> list[Metadata]:
        """Generate the podcast RSS feed (``podcast.xml``).

        Uses the ``feedgen`` library with the podcast extension to produce
        a valid RSS 2.0 feed with iTunes podcast extensions. Only MP4 files
        with ``feed_eligible == True`` are included.

        For each item:
            - **title**: from metadata ``title``.
            - **description**: from metadata ``summary`` (LLM-generated key
              takeaways for MP4 files), falling back to ``description``
              for backward compatibility.
            - **link**: absolute URL to the source MP4 file.
            - **enclosure**: URL, type ``audio/mp4``, and length (file size).
            - **publication date**: from metadata ``date``.
            - **GUID**: based on the filepath (as a permalink URL).

        The feed is written to ``DATA_DIR/podcast.xml``, overwriting any
        existing file (idempotent regeneration).

        Returns:
            The list of :class:`Metadata` items included in the podcast feed.

        Raises:
            OSError: If the feed file cannot be written.
        """
        all_metadata = self._discover_metadata_files()
        mp4_items = [m for m in all_metadata if m.filetype == "mp4"]

        fg = _FeedGenerator()
        fg.load_extension("podcast")

        # Feed-level metadata
        fg.title(_PODCAST_FEED_TITLE)
        fg.link(href=self._absolute_url("podcast.xml"), rel="self")
        fg.link(href=self.base_url)
        fg.description(_PODCAST_FEED_TITLE)
        fg.language("en")

        for metadata in mp4_items:
            entry = fg.add_entry()
            entry.id(str(metadata.filepath))
            entry.guid(self._absolute_url(metadata.filepath))
            entry.title(metadata.title)

            # Description from summary field (holds LLM key takeaways for MP4),
            # falling back to description for backward compatibility.
            description = metadata.summary or metadata.description or ""
            entry.description(description)

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
            pub_date = datetime.fromtimestamp(metadata.date, tz=UTC) if metadata.date else datetime.now(UTC)
            entry.pubDate(pub_date)

        output_path = self.data_dir / "podcast.xml"
        fg.rss_file(str(output_path), pretty=True)
        _logger.info("Generated podcast feed: %s (%d items)", output_path, len(mp4_items))

        return mp4_items

    # ------------------------------------------------------------------
    # Articles feed (PDF/HTML files)
    # ------------------------------------------------------------------

    def generate_articles_feed(self) -> list[Metadata]:
        """Generate the articles RSS feed (``articles.xml``).

        Uses the ``feedgen`` library to produce a standard RSS 2.0 feed.
        Only PDF and HTML files with ``feed_eligible == True`` are included.

        For each item:
            - **title**: from metadata ``title``.
            - **description**: from metadata ``content`` (extracted text) if
              available, otherwise from ``description``, otherwise empty.
            - **link**: absolute URL to the source file.
            - **publication date**: from metadata ``date``.
            - **GUID**: based on the filepath (as a permalink URL).

        The feed is written to ``DATA_DIR/articles.xml``, overwriting any
        existing file (idempotent regeneration).

        Returns:
            The list of :class:`Metadata` items included in the articles feed.

        Raises:
            OSError: If the feed file cannot be written.
        """
        all_metadata = self._discover_metadata_files()
        article_items = [m for m in all_metadata if m.filetype in ("pdf", "html")]

        fg = _FeedGenerator()

        # Feed-level metadata
        fg.title(_ARTICLES_FEED_TITLE)
        fg.link(href=self._absolute_url("articles.xml"), rel="self")
        fg.link(href=self.base_url)
        fg.description(_ARTICLES_FEED_TITLE)
        fg.language("en")

        for metadata in article_items:
            entry = fg.add_entry()
            entry.id(str(metadata.filepath))
            entry.guid(self._absolute_url(metadata.filepath))
            entry.title(metadata.title)

            # Description from content (if available), fallback to
            # description field, then empty string.
            description = metadata.content or metadata.description or ""
            entry.description(description)

            # Link to the source file.
            source_url = self._absolute_url(metadata.filepath)
            entry.link(href=source_url)

            # Publication date.
            pub_date = datetime.fromtimestamp(metadata.date, tz=UTC) if metadata.date else datetime.now(UTC)
            entry.pubDate(pub_date)

        output_path = self.data_dir / "articles.xml"
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

    def generate_all_feeds(self, validate: bool = False) -> dict[str, list[Metadata]]:
        """Generate both podcast and articles feeds.

        Calls :meth:`generate_podcast_feed` and
        :meth:`generate_articles_feed` in sequence. If one feed generation
        fails, the error is logged and the other feed is still generated.

        When *validate* is ``True``, :meth:`validate_output` is called after
        generation and any validation errors or warnings are logged.

        This method is **idempotent**: feeds are always regenerated from the
        current metadata, overwriting any existing feed files.

        Args:
            validate: When ``True``, validate the generated feeds and log
                any errors or warnings found.

        Returns:
            A dictionary with keys ``"podcast"`` and ``"articles"``, each
            mapping to the list of :class:`Metadata` items included in the
            respective feed. If a feed generation failed, its value will be
            an empty list.
        """
        results: dict[str, list[Metadata]] = {
            "podcast": [],
            "articles": [],
        }

        # Generate podcast feed.
        try:
            results["podcast"] = self.generate_podcast_feed()
        except Exception as exc:
            _logger.error("Failed to generate podcast feed: %s", exc)

        # Generate articles feed.
        try:
            results["articles"] = self.generate_articles_feed()
        except Exception as exc:
            _logger.error("Failed to generate articles feed: %s", exc)

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
