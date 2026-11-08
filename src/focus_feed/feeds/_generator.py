"""Feed generator for Focus Feed.

Reads JSON metadata files from the data directory, filters out error entries,
and generates RSS 2.0 feeds (podcast.xml and articles.xml) for consumption by
mobile podcast apps (e.g., Antennapod) and RSS readers. Podcast feeds include
supported audio and video media files.

Design references:
    - design.md section 3.4 (Feed Output)
    - design.md section 10.1 (Feed Format: RSS 2.0)
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

from collections.abc import Iterable
from datetime import UTC, datetime
import hashlib
import html
import json
import logging
import os
from pathlib import Path
from typing import Any

from feedgen.feed import FeedGenerator as _FeedGenerator

from focus_feed.config import Settings, get_config
from focus_feed.errors import log_error
from focus_feed.feeds._markdown import _markdown_to_html, _sanitize_cdata_content
from focus_feed.feeds._mime import PODCAST_MIME_TYPES as _PODCAST_MIME_TYPES  # noqa: F401
from focus_feed.feeds._mime import get_mime_type
from focus_feed.feeds._validation import (
    FeedValidationResult,
    validate_feed,
)
from focus_feed.filetypes import DOCUMENT_FILETYPES, MEDIA_FILETYPES
from focus_feed.metadata import Metadata, write_metadata_json
from focus_feed.metadata_io import is_orphaned_json

_logger = logging.getLogger("focus_feed")

# Feed titles for the two generated feeds.
_PODCAST_FEED_TITLE = "Local Podcast Feed"
_ARTICLES_FEED_TITLE = "Local Articles Feed"


def _is_orphaned_json(json_path: Path) -> bool:
    """Check if a JSON metadata file is orphaned (no corresponding source).

    A JSON file is orphaned when removing the ``.json`` suffix does not
    yield an existing file with a supported source extension.

    Args:
        json_path: Path to the ``.json`` file to check.

    Returns:
        ``True`` if the JSON file is orphaned, ``False`` otherwise.
    """
    return is_orphaned_json(json_path)


def _get_summarization_attribution(metadata: Metadata) -> str:
    """Generate attribution string for feed item descriptions.

    Returns a string like "Summarized by {model} at {datetime}" in local timezone,
    or empty string if attribution cannot be determined.

    Since all feed-eligible files now require ``summarize`` to be complete,
    this always uses ``summarize.model`` and ``summarize.end_time``.
    """
    # Get config for llm_model fallback; return empty string if config unavailable.
    try:
        from focus_feed.feeds import get_config as package_get_config

        config = package_get_config()
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


class FeedGenerator:
    """Generates RSS 2.0 feeds from JSON metadata files.

    Scans the data directory for ``*.json`` metadata files, filters out
    error entries, and produces two RSS feeds:

    - ``podcast.xml`` — one item per supported audio/video file with an audio
      enclosure.
    - ``articles.xml`` — one item per PDF or HTML file.

    Supports per-folder feed generation: when a folder contains a
    ``feed.json`` file, feeds are generated inside that folder using the
    title and description from the ``feed.json``.

    Attributes:
        data_dir: Absolute path to the data directory.
        base_url: Base URL used to construct absolute enclosure/source URLs.
        persist_hashes: Whether computed source hashes are saved to metadata.
        _eligible_paths: Optional allow-list of source paths to discover.
    """

    def __init__(
        self,
        data_dir: str | Path,
        base_url: str | None = None,
        eligible_paths: Iterable[Path] | None = None,
        persist_hashes: bool = True,
        config: Settings | None = None,
    ) -> None:
        """Initialize the feed generator.

        Args:
            data_dir: Path to the data directory. This is a required parameter;
                the value is no longer loaded from the application configuration.
            base_url: Base URL for constructing absolute URLs. When ``None``,
                the value is loaded from the application configuration
                (``base_url``).
            eligible_paths: Optional allow-list of source file paths to
                include in feed discovery.
            persist_hashes: Whether to save computed hashes to metadata files.
            config: Optional application settings. When ``None``, settings
                are loaded using ``get_config()``.
        """
        self._config = config
        if base_url is None:
            resolved_config = self._config if self._config is not None else get_config()
            self._config = resolved_config
            base_url = resolved_config.base_url

        self.data_dir: Path = Path(data_dir).resolve()
        self.base_url: str = base_url.rstrip("/")
        self.data_dir_basename: str = self.data_dir.name
        self.persist_hashes: bool = persist_hashes
        self._eligible_paths: frozenset[Path] | None = (
            None
            if eligible_paths is None
            else frozenset(Path(path).resolve() for path in eligible_paths)
        )

        if not self.data_dir.exists():
            _logger.critical("Data directory does not exist: %s", self.data_dir)

    # ------------------------------------------------------------------
    # Metadata discovery
    # ------------------------------------------------------------------

    def _discover_metadata_files(self) -> list[Metadata]:
        """Recursively discover and load all valid JSON metadata files.

        Walks ``data_dir`` for ``*.json`` files, attempts to parse each as a
        :class:`~focus_feed.metadata.Metadata` object, and filters
        out entries where ``feed_eligible`` is ``False`` (i.e. incomplete
        processing or error state).

        ``feed.json`` is a feed configuration file, not a per-source metadata
        file, and is explicitly excluded from discovery.

        Files that cannot be parsed as valid metadata are logged as warnings
        and skipped, so a single corrupt JSON file does not abort feed
        generation.

        Returns:
            A list of :class:`~focus_feed.metadata.Metadata` objects
            for all feed-eligible files, sorted by folder name then filename.
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

                # feed.json is a feed configuration file, not a per-source
                # metadata file — skip it before any parsing attempts.
                if filename == "feed.json":
                    continue

                total_count += 1
                json_path = root_path / filename

                # Skip orphaned JSON files (no corresponding source file).
                if _is_orphaned_json(json_path):
                    _logger.warning(
                        "Skipping orphaned JSON file (no source): %s",
                        json_path,
                    )
                    continue

                try:
                    metadata = Metadata.from_json_file(json_path)
                    source_path = json_path.with_name(json_path.name.removesuffix(".json"))
                    metadata._source_filepath = str(source_path.relative_to(self.data_dir))
                except Exception as exc:
                    _logger.warning(
                        "Failed to parse metadata file %s: %s",
                        json_path,
                        exc,
                    )
                    continue

                if (
                    self._eligible_paths is not None
                    and source_path.resolve() not in self._eligible_paths
                ):
                    _logger.debug(
                        "Skipping metadata outside eligible paths: %s",
                        json_path,
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
        filepath = Path(metadata.source_filepath)
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
        return self._absolute_url(metadata.source_filepath)

    def _ensure_hashes(self, metadata_items: list[Metadata]) -> None:
        """Compute missing source hashes, optionally persisting them to metadata."""
        for metadata in metadata_items:
            if metadata.hash:
                continue

            source_path = self.data_dir / metadata.source_filepath
            try:
                digest = hashlib.sha256()
                with source_path.open("rb") as source_file:
                    for chunk in iter(lambda: source_file.read(8192), b""):
                        digest.update(chunk)
                metadata.hash = digest.hexdigest()

                if not self.persist_hashes:
                    continue

                metadata_path = source_path.with_name(source_path.name + ".json")
                stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(stored_metadata, dict):
                    stored_metadata["hash"] = metadata.hash
                    stored_title = stored_metadata.get("title")
                    if isinstance(stored_title, str):
                        stored_metadata["title"] = {
                            "filename": stored_title,
                            "extracted": None,
                            "generated": None,
                        }
                    write_metadata_json(stored_metadata, source_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                _logger.warning("Failed to compute or store hash for %s: %s", source_path, exc)

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

    def _discover_feed_json_dirs(self) -> list[tuple[Path, dict[str, Any]]]:
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
        results: list[tuple[Path, dict[str, Any]]] = []

        for feed_json_path in self.data_dir.rglob("feed.json"):
            _logger.debug("Found feed.json at %s", feed_json_path)
            try:
                config: dict[str, Any] = json.loads(feed_json_path.read_text(encoding="utf-8"))
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
        return [m for m in metadata_list if m.source_filepath.startswith(prefix)]

    # ------------------------------------------------------------------
    # Podcast feed (audio/video media files)
    # ------------------------------------------------------------------

    def generate_podcast_feed(
        self,
        output_dir: Path | None = None,
        title: str | None = None,
        description: str | None = None,
        link: str | None = None,
        language: str | None = None,
        author: str | None = None,
        image: dict[str, str] | None = None,
        metadata_items: list[Metadata] | None = None,
    ) -> list[Metadata]:
        """Generate the podcast RSS feed (``podcast.xml``).

        Uses the ``feedgen`` library with the podcast extension to produce
        a valid RSS 2.0 feed with iTunes podcast extensions. Only supported
        media files with ``feed_eligible == True`` are included.

        For each item:
            - **title**: from metadata ``final_title`` computed property.
            - **description**: from metadata ``summary`` (LLM-generated key
              takeaways for media files), falling back to ``description``
              for backward compatibility.  Markdown is converted to HTML.
            - **link**: absolute URL to the source media file.
            - **enclosure**: URL, media-specific MIME type, and length (file size).
            - **publication date**: from ``summarize.end_time``.
            - **GUID**: last 12 characters of the ``hash`` field when present,
              otherwise the absolute URL of the file path (FR-020).

        The feed is written to ``output_dir/podcast.xml``, overwriting any
        existing file (idempotent regeneration).

        Args:
            output_dir: Directory to write the feed file.  Defaults to
                ``self.data_dir``.
            title: Feed title.  Defaults to ``_PODCAST_FEED_TITLE``.
            description: Feed description.  Defaults to an empty string.
            link: Feed-level channel link URL.  When ``None`` (default),
                ``self.base_url`` is used.  If provided (e.g. from
                ``feed.json``), it overrides the default.
            language: Feed language (e.g., ``"en-us"``).  When ``None``
                (default), no language tag is emitted.
            author: Feed author name.  When ``None`` (default), no author is
                emitted.
            image: Feed image dict with ``url``, ``title``, and ``link`` keys.
                When ``None`` (default), no image tag is emitted.
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
        if link is None:
            link = self.base_url

        if metadata_items is None:
            all_metadata = self._discover_metadata_files()
            media_items = [m for m in all_metadata if m.filetype in MEDIA_FILETYPES]
        else:
            media_items = [m for m in metadata_items if m.filetype in MEDIA_FILETYPES]

        if not media_items:
            _logger.info(
                "No media files found; skipping podcast feed generation for %s",
                output_dir,
            )
            return []

        self._ensure_hashes(media_items)

        fg = _FeedGenerator()
        fg.load_extension("podcast")

        # Compute the feed file path relative to data_dir for the self-link.
        feed_rel_path = self._feed_rel_path(output_dir, "podcast.xml")

        # Feed-level metadata
        fg.title(title)
        fg.link(href=self._absolute_url(feed_rel_path), rel="self")
        fg.link(href=link)
        description = description or title
        fg.description(description)
        if language:
            fg.language(language)
        if author:
            fg.author({"name": author})
        if image:
            fg.image(url=image["url"], title=image.get("title", ""), link=image.get("link", ""))

        for metadata in media_items:
            # Publication date: from summarize.end_time (no fallback).
            if metadata.summarize is None or metadata.summarize.end_time is None:
                _logger.error(
                    "Missing summarize.end_time for %s, skipping",
                    metadata.filename,
                )
                continue
            entry = fg.add_entry()
            entry.id(metadata.source_filepath)
            entry.guid(self._compute_guid(metadata))
            entry.title(metadata.final_title)

            # Description from summary field (holds LLM key takeaways for media),
            # falling back to description for backward compatibility.
            # Convert markdown to HTML for safe rendering in feed readers.
            desc_text = metadata.summary or metadata.description or ""
            # Append summarization attribution before HTML conversion.
            attribution = _get_summarization_attribution(metadata)
            if attribution:
                desc_text = f"{desc_text}\n\n{attribution}" if desc_text else attribution
            entry.content(
                content=_sanitize_cdata_content(_markdown_to_html(desc_text)),
                type="CDATA",
            )

            # Link to the source file.
            source_url = self._absolute_url(metadata.source_filepath)
            entry.link(href=source_url)

            # Enclosure with URL, type, and length.
            entry.enclosure(
                url=source_url,
                length=metadata.size,
                type=get_mime_type(metadata.filetype or ""),
            )

            pub_date = datetime.fromtimestamp(metadata.summarize.end_time, tz=UTC)
            entry.pubDate(pub_date)

        output_path = output_dir / "podcast.xml"
        try:
            fg.rss_file(str(output_path), pretty=True)
        except OSError as exc:
            _logger.warning("Failed to write feed file %s: %s", output_path, exc)
            log_error(exc, filepath=str(output_path), stage="feed_write")
            raise
        _logger.debug("Wrote feed file: %s", output_path)
        _logger.info("Generated podcast feed: %s (%d items)", output_path, len(media_items))

        return media_items

    # ------------------------------------------------------------------
    # Articles feed (PDF/HTML files)
    # ------------------------------------------------------------------

    def generate_articles_feed(
        self,
        output_dir: Path | None = None,
        title: str | None = None,
        description: str | None = None,
        link: str | None = None,
        language: str | None = None,
        author: str | None = None,
        image: dict[str, str] | None = None,
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
            - **publication date**: from ``summarize.end_time``.
            - **GUID**: last 12 characters of the ``hash`` field when present,
              otherwise the absolute URL of the file path (FR-020).

        The feed is written to ``output_dir/articles.xml``, overwriting any
        existing file (idempotent regeneration).

        Args:
            output_dir: Directory to write the feed file.  Defaults to
                ``self.data_dir``.
            title: Feed title.  Defaults to ``_ARTICLES_FEED_TITLE``.
            description: Feed description.  Defaults to an empty string.
            link: Feed-level channel link URL.  When ``None`` (default),
                ``self.base_url`` is used.  If provided (e.g. from
                ``feed.json``), it overrides the default.
            language: Feed language (e.g., ``"en-us"``).  When ``None``
                (default), no language tag is emitted.
            author: Feed author name.  When ``None`` (default), no author is
                emitted.
            image: Feed image dict with ``url``, ``title``, and ``link`` keys.
                When ``None`` (default), no image tag is emitted.
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
        if link is None:
            link = self.base_url

        if metadata_items is None:
            all_metadata = self._discover_metadata_files()
            article_items = [m for m in all_metadata if m.filetype in DOCUMENT_FILETYPES]
        else:
            article_items = [m for m in metadata_items if m.filetype in DOCUMENT_FILETYPES]

        if not article_items:
            _logger.info(
                "No document files found; skipping articles feed generation for %s",
                output_dir,
            )
            return []

        self._ensure_hashes(article_items)

        fg = _FeedGenerator()

        # Compute the feed file path relative to data_dir for the self-link.
        feed_rel_path = self._feed_rel_path(output_dir, "articles.xml")

        # Feed-level metadata
        fg.title(title)
        fg.link(href=self._absolute_url(feed_rel_path), rel="self")
        fg.link(href=link)
        description = description or title
        fg.description(description)
        if language:
            fg.language(language)
        if author:
            fg.author({"name": author})
        if image:
            fg.image(url=image["url"], title=image.get("title", ""), link=image.get("link", ""))

        for metadata in article_items:
            # Publication date: from summarize.end_time (no fallback).
            if metadata.summarize is None or metadata.summarize.end_time is None:
                _logger.error(
                    "Missing summarize.end_time for %s, skipping",
                    metadata.filename,
                )
                continue
            entry = fg.add_entry()
            entry.id(metadata.source_filepath)
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
            entry.content(
                content=_sanitize_cdata_content(_markdown_to_html(desc_text)),
                type="CDATA",
            )

            # Link to the source file.
            source_url = self._absolute_url(metadata.source_filepath)
            entry.link(href=source_url)

            pub_date = datetime.fromtimestamp(metadata.summarize.end_time, tz=UTC)
            entry.pubDate(pub_date)

        output_path = output_dir / "articles.xml"
        try:
            fg.rss_file(str(output_path), pretty=True)
        except OSError as exc:
            _logger.warning("Failed to write feed file %s: %s", output_path, exc)
            log_error(exc, filepath=str(output_path), stage="feed_write")
            raise
        _logger.debug("Wrote feed file: %s", output_path)
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
        language: str | None,
        author: str | None,
        image: dict[str, str] | None,
        metadata_items: list[Metadata],
    ) -> dict[str, list[Metadata]]:
        """Generate podcast and/or articles feeds for a specific folder.

        This method is used for per-folder feed generation when a folder
        contains a ``feed.json`` file.  It generates:

        - ``podcast.xml`` — only if audio/video media files are present in
          the folder's subtree.
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
            language: Feed language from ``feed.json``, or ``None`` to omit.
            author: Feed author from ``feed.json``, or ``None`` to omit.
            image: Feed image dict from ``feed.json``, or ``None`` to omit.
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

        media_items = [m for m in metadata_items if m.filetype in MEDIA_FILETYPES]
        article_items = [m for m in metadata_items if m.filetype in DOCUMENT_FILETYPES]

        _logger.info("Generating feeds for folder: %s", folder)

        # Generate podcast.xml only if supported media files are present.
        if media_items:
            try:
                results["podcast"] = self.generate_podcast_feed(
                    output_dir=folder,
                    title=title,
                    description=description,
                    link=link,
                    language=language,
                    author=author,
                    image=image,
                    metadata_items=metadata_items,
                )
            except Exception as exc:
                log_error(
                    exc,
                    filepath=str(folder),
                    stage="podcast_feed_generation",
                )
        else:
            _logger.debug("Skipping podcast feed for %s: no media items", folder)

        # Generate articles.xml only if PDF/HTML files are present.
        if article_items:
            try:
                results["articles"] = self.generate_articles_feed(
                    output_dir=folder,
                    title=title,
                    description=description,
                    link=link,
                    language=language,
                    author=author,
                    image=image,
                    metadata_items=metadata_items,
                )
            except Exception as exc:
                log_error(
                    exc,
                    filepath=str(folder),
                    stage="articles_feed_generation",
                )
        else:
            _logger.debug("Skipping articles feed for %s: no PDF/HTML items", folder)

        # Warn if no feed-eligible files were found in this folder despite
        # having a feed.json configuration file.
        if not media_items and not article_items:
            _logger.warning("No feed-eligible files found in %s", str(folder))

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

        1. **Root-level feeds**: Generates ``podcast.xml`` and
           ``articles.xml`` in ``data_dir``. Root-level feeds are only
           generated if a ``data_dir/feed.json`` file exists; its title,
           description, and optional link are used for the root-level feeds.

        2. **Per-folder feeds**: Walks ``data_dir`` recursively for
           ``feed.json`` files and generates feeds in each configured folder.

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
        _logger.info("Starting feed generation")

        results: dict[str, list[Metadata]] = {
            "podcast": [],
            "articles": [],
        }

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
            root_language: str | None = None
            root_author: str | None = None
            root_image: dict[str, str] | None = None
            try:
                root_config = json.loads(root_feed_json.read_text(encoding="utf-8"))
                root_title = root_config.get("title")
                root_description = root_config.get("description")
                root_link = root_config.get("link")
                root_language = root_config.get("language")
                root_author = root_config.get("author")
                root_image = root_config.get("image")
            except (json.JSONDecodeError, OSError) as exc:
                _logger.warning(
                    "Failed to parse root feed.json at %s: %s",
                    root_feed_json,
                    exc,
                )
                root_title = None
                root_description = None
                root_link = None
                root_language = None
                root_author = None
                root_image = None

            # Generate podcast feed (media presence is handled by the generator)
            try:
                results["podcast"] = self.generate_podcast_feed(
                    output_dir=self.data_dir,
                    title=root_title,
                    description=root_description,
                    link=root_link,
                    language=root_language,
                    author=root_author,
                    image=root_image,
                )
            except Exception as exc:
                log_error(exc, filepath=str(self.data_dir), stage="podcast_feed_generation")

            # Generate articles feed (only if PDF/HTML files exist)
            try:
                results["articles"] = self.generate_articles_feed(
                    output_dir=self.data_dir,
                    title=root_title,
                    description=root_description,
                    link=root_link,
                    language=root_language,
                    author=root_author,
                    image=root_image,
                )
            except Exception as exc:
                log_error(exc, filepath=str(self.data_dir), stage="articles_feed_generation")
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
            folder_language = config.get("language")
            folder_author = config.get("author")
            folder_image = config.get("image")

            # Filter metadata to only items within this folder tree.
            folder_metadata = self._filter_metadata_for_folder(all_metadata, folder_path)

            self._generate_feeds_for_folder(
                folder=folder_path,
                title=folder_title,
                description=folder_description,
                link=folder_link,
                language=folder_language,
                author=folder_author,
                image=folder_image,
                metadata_items=folder_metadata,
            )

        # Generate an index in every directory containing a feed.json file,
        # even when its contents are malformed. Index generation handles
        # invalid configuration gracefully.
        feed_dirs = sorted(
            {
                feed_json_path.parent
                for feed_json_path in self.data_dir.rglob("feed.json")
                if feed_json_path.is_file()
            }
        )
        for folder_path in feed_dirs:
            try:
                self._generate_index_html(output_dir=folder_path)
            except Exception as exc:
                log_error(exc, filepath=str(folder_path), stage="index_html_generation")

        # Remove index pages left behind when the corresponding feed.json
        # has been removed since the previous generation.
        for index_path in self.data_dir.rglob("index.html"):
            if not index_path.is_file() or (index_path.parent / "feed.json").is_file():
                continue
            try:
                index_path.unlink()
                _logger.info("Removed stale index.html: %s", index_path)
            except OSError as exc:
                _logger.warning("Failed to remove stale index.html %s: %s", index_path, exc)

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

        _logger.info("Validating feeds")

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
                log_error(ValueError(error), filepath=name, stage="feed_validation")
            for warning in result.warnings:
                _logger.warning("Feed validation warning [%s]: %s", name, warning)
            if result.is_valid:
                _logger.info("Feed validation passed [%s]", name)

    # ------------------------------------------------------------------
    # Index HTML generation
    # ------------------------------------------------------------------

    def _generate_index_html(self, output_dir: Path | None = None) -> None:
        """Generate or update an ``index.html`` listing subtree XML feed files.

        Discovers all ``*.xml`` files in *output_dir* and its subdirectories,
        categorizes them by feed type, and writes an HTML index to
        ``output_dir / index.html``.  The index is generated only for
        directories containing a ``feed.json`` configuration.

        The index includes:
        - A title "Feed Index".
        - Sections organized by feed type (Podcast Feeds, Articles Feeds,
          and an "Other Feeds" section for any additional XML files).
        - Links using absolute URLs (``base_url + relative path``), with
          relative paths as link text.
        - Feed titles and descriptions from ``feed.json`` when available.
        - HTML-escaped titles, descriptions, filenames, and URLs.

        Paths displayed in the index are relative to *output_dir*, while
        feed links use absolute URLs rooted at ``base_url``.  If no XML feeds
        are found, an index with empty sections is still generated. An
        existing ``index.html`` is overwritten.

        Raises:
            OSError: If the index file cannot be written.
        """
        if output_dir is None:
            output_dir = self.data_dir

        # Check if target directory exists.
        if not output_dir.exists() or not output_dir.is_dir():
            _logger.warning(
                "Index directory does not exist, skipping index.html generation: %s",
                output_dir,
            )
            return
        if not (output_dir / "feed.json").is_file():
            _logger.debug(
                "No feed.json in %s; skipping index.html generation",
                output_dir,
            )
            return

        _logger.info("Generating index.html in %s", output_dir)

        # Discover XML feeds in the target directory and its subtree.
        xml_files = sorted(output_dir.rglob("*.xml"))

        # Categorize feeds by type based on filename.
        podcast_feeds: list[tuple[str, str, str, str]] = []
        articles_feeds: list[tuple[str, str, str, str]] = []
        other_feeds: list[tuple[str, str, str, str]] = []
        feed_configs: dict[Path, dict[str, Any] | None] = {}

        for xml_path in xml_files:
            rel_path = str(xml_path.relative_to(output_dir))
            data_rel_path = str(xml_path.relative_to(self.data_dir))
            abs_url = self._absolute_url(data_rel_path)
            feed_name = xml_path.name
            feed_dir = xml_path.parent
            feed_json_path = feed_dir / "feed.json"
            if feed_dir not in feed_configs:
                try:
                    config = (
                        json.loads(feed_json_path.read_text(encoding="utf-8"))
                        if feed_json_path.exists()
                        else None
                    )
                    feed_configs[feed_dir] = config if isinstance(config, dict) else None
                except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                    _logger.warning("Failed to parse feed.json at %s: %s", feed_json_path, exc)
                    feed_configs[feed_dir] = None

            config = feed_configs[feed_dir]
            configured_title = config.get("title") if config else None
            title = (
                configured_title if isinstance(configured_title, str) and configured_title else ""
            )
            configured_description = config.get("description") if config else None
            description = configured_description if isinstance(configured_description, str) else ""
            feed = (rel_path, abs_url, title, description)

            if feed_name == "podcast.xml":
                podcast_feeds.append(feed)
            elif feed_name == "articles.xml":
                articles_feeds.append(feed)
            else:
                other_feeds.append(feed)

        # Sort each category by relative path for deterministic output.
        podcast_feeds.sort(key=lambda x: x[0])
        articles_feeds.sort(key=lambda x: x[0])
        other_feeds.sort(key=lambda x: x[0])

        # Build HTML content.
        lines: list[str] = []

        def append_feed_item(feed: tuple[str, str, str, str]) -> None:
            rel_path, abs_url, title, description = feed
            escaped_path = html.escape(rel_path)
            escaped_url = html.escape(abs_url, quote=True)
            escaped_title = html.escape(title)
            escaped_description = html.escape(description)
            title_prefix = f"<strong>{escaped_title}</strong>: " if escaped_title else ""
            if escaped_description:
                lines.append(
                    f'        <li>{title_prefix}<a href="{escaped_url}">{escaped_path}</a>'
                    f"<br><small>{escaped_description}</small></li>"
                )
            else:
                lines.append(
                    f'        <li>{title_prefix}<a href="{escaped_url}">{escaped_path}</a></li>'
                )

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
        for feed in podcast_feeds:
            append_feed_item(feed)
        lines.append("    </ul>")

        # Articles Feeds section.
        lines.append("    <h2>Articles Feeds</h2>")
        lines.append("    <ul>")
        for feed in articles_feeds:
            append_feed_item(feed)
        lines.append("    </ul>")

        # Other feeds section (only if there are non-standard XML files).
        if other_feeds:
            lines.append("    <h2>Other Feeds</h2>")
            lines.append("    <ul>")
            for feed in other_feeds:
                append_feed_item(feed)
            lines.append("    </ul>")

        lines.append("</body>")
        lines.append("</html>")

        html_content = "\n".join(lines)
        index_path = output_dir / "index.html"
        try:
            index_path.write_text(html_content, encoding="utf-8")
        except OSError as exc:
            log_error(exc, filepath=str(index_path), stage="index_html_write")
            raise
        _logger.info(
            "Generated index.html: %s (%d feeds)",
            index_path,
            len(xml_files),
        )
