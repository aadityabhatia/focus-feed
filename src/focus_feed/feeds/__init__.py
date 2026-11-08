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

from focus_feed.config import get_config
from focus_feed.feeds._generator import (
    FeedGenerator,
    _get_summarization_attribution,
    _is_orphaned_json,
)
from focus_feed.feeds._markdown import _markdown_to_html, _sanitize_cdata_content
from focus_feed.feeds._mime import _PODCAST_MIME_TYPES, get_mime_type
from focus_feed.feeds._validation import FeedValidationResult, validate_feed
from focus_feed.filetypes import DOCUMENT_FILETYPES, MEDIA_FILETYPES

__all__ = [
    "DOCUMENT_FILETYPES",
    "MEDIA_FILETYPES",
    "_PODCAST_MIME_TYPES",
    "FeedGenerator",
    "FeedValidationResult",
    "_get_summarization_attribution",
    "_is_orphaned_json",
    "_markdown_to_html",
    "_sanitize_cdata_content",
    "get_config",
    "get_mime_type",
    "validate_feed",
]
