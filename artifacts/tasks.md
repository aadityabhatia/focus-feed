# Local Feed Generator - Implementation Plan

**Project:** `local-feed-generator`  
**Date:** 2026-10-16  
**Python:** 3.14+  
**Dependency Manager:** uv  

---

## Overview

This document provides a detailed, executable implementation plan for building `local-feed-generator`, a Python application that processes PDF/HTML/MP4 files and generates RSS feeds. The plan is organized into phases with ordered tasks, dependencies, and completion criteria.

**Latest Updates:**
- Added support for title as dict with source tracking (filename, extracted, generated, final)
- Added new config variables: SUMMARY_PROMPT, TITLE_PROMPT, SUMMARY_REFLECTION_PROMPT
- Added reflection.md support for reflection questions in summarization
- Added per-folder feed generation with feed.json configuration
- Added markdown to HTML conversion for feed descriptions
- Added millisecond precision timestamps (UNIX epoch with float)
- Added module name prefixing for titles
- Added whitespace/control character normalization in extracted content
- Added content extraction for PDF/HTML files (not just metadata)
- Added ffprobe integration for audio metadata (duration, creation_time)
- Added conditional feed generation: feeds only where feed.json exists AND applicable files present
- Added index.html generation for feed discovery
- Added NO root feeds without feed.json requirement
- Optional dependencies: faster-whisper, av in separate group
- HTML/PDF files now get content extracted and included in feeds
- summarize command now processes PDF/HTML files

---

## Phase 1: Project Setup and Dependencies

### Objective
Establish the project foundation with dependency management, directory structure, and basic configuration.

### Tasks

- [x] T1.1: Initialize Project Structure
- **Description:** Create the project directory structure as specified in design doc section 5.1
- **Actions:**
  - Create `local-feed-generator/` directory
  - Create `src/local_feed_generator/` package directory
  - Create `src/local_feed_generator/services/` subpackage directory
  - Create all `__init__.py` files for packages
- **Completion Criteria:**
  - Directory structure matches design doc exactly
  - All Python packages are importable
  - `src/local_feed_generator/__init__.py` exists
  - `src/local_feed_generator/services/__init__.py` exists

- [x] T1.2: Create pyproject.toml
- **Description:** Set up project metadata and dependencies using uv
- **Actions:**
  - Create `pyproject.toml` with project name `local-feed-generator`
  - Specify Python requirement
  - Add all dependencies from design doc section 4.1
  - Add optional dependency group for local transcription (faster-whisper, av)
  - Configure uv as the build system
- **Dependencies:**
  - Required packages: `feedgen`, `watchdog`, `openai`, `pydantic-settings`, `pydantic`, `click`, `pypdf`, `beautifulsoup4`, `python-dotenv`
  - Optional group `local-transcription`: `faster-whisper`, `av>=11.0,<19.0`
- **Completion Criteria:**
  - `pyproject.toml` is valid and can be parsed by uv
  - All dependencies are listed
  - Optional dependency group configured
  - `uv lock` succeeds without errors

- [x] T1.3: Set Up Development Environment
- **Description:** Create initial configuration files
- **Actions:**
  - Create `.env.example` with all environment variables from design doc section 3.2
  - Include new required variables: SUMMARY_PROMPT, TITLE_PROMPT
  - Include new optional variable: SUMMARY_REFLECTION_PROMPT
  - Create `.gitignore` file (if using git)
  - Create a basic `README.md` stub
- **Completion Criteria:**
  - `.env.example` contains all required environment variables with example values
  - `.env.example` includes: `BASE_URL`, `DATA_DIR`, `LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`, `SUMMARY_PROMPT`, `TITLE_PROMPT`, `FEED_FORMAT`, `TRANSCRIPTION_MODEL`, `TRANSCRIPTION_DEVICE`, `TRANSCRIPTION_COMPUTE_TYPE`, `SUMMARY_REFLECTION_PROMPT`

- [x] T1.4: Verify Dependency Installation
- **Description:** Ensure all dependencies can be installed
- **Actions:**
  - Run `uv sync` to install dependencies
  - Verify all packages import successfully
  - Test basic functionality of each dependency
- **Completion Criteria:**
  - All packages install without errors
  - `import feedgen`, `import watchdog`, `import openai`, etc. all succeed
  - Optional dependencies install with `uv sync --optional-group local-transcription`

---

## Phase 2: Core Modules (Config, Scanner, Metadata)

### Objective
Implement the foundational modules for configuration, file scanning, and metadata handling with new features.

### Tasks

- [x] T2.1: Implement Configuration System
- **Description:** Create config.py with Pydantic settings for all environment variables
- **File:** `src/local_feed_generator/config.py`
- **Actions:**
  - Create `Settings` class extending `pydantic_settings.BaseSettings`
  - Define all environment variables with types and defaults
  - Add validation for required fields (`BASE_URL`, `DATA_DIR`, `LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`)
  - Add new required fields: `SUMMARY_PROMPT`, `TITLE_PROMPT`
  - Add new optional field: `SUMMARY_REFLECTION_PROMPT` (defaults to None)
  - Add optional fields with design doc defaults (`FEED_FORMAT=rss`)
  - Implement singleton pattern for config access
- **Completion Criteria:**
  - Config validations work (errors on missing required fields)
  - Config can be loaded from `.env` file
  - Config can be loaded from environment variables
  - All new prompt variables are supported
  - All values from design doc section 3.2 are supported

- [x] T2.2: Implement Metadata Model
- **Description:** Create metadata.py with Pydantic models for JSON schema (Version 2.0)
- **File:** `src/local_feed_generator/metadata.py`
- **Actions:**
  - Create `Metadata` dataclass/model matching schema from design doc section 3.3
  - **NEW:** `title` field is now a dict with keys: `filename`, `extracted`, `generated`, `final` (can also be a plain string for backward compatibility)
  - **NEW:** Add `extracted_title` field (string or None)
  - **NEW:** `date` field is float (millisecond precision UNIX epoch)
  - **NEW:** Add `duration` field (float, MP4 only, optional)
  - **NEW:** Add `creation_time` field (ISO-8601 string, MP4 only, optional)
  - Include all fields: `version`, `title`, `extracted_title`, `description`, `summary`, `content`, `filename`, `filepath`, `filetype`, `date`, `duration`, `creation_time`, `size`, `transcribe`, `summarize`, `extract`, `feed_eligible`, `error`
  - `transcribe`, `summarize`, `extract` are step maps with fields: `status` (null | "in_progress" | "complete"), `start_time` (float, ms precision), `end_time` (float, ms precision), `input_size`, `output_size`, `words`, `model`, `input_tokens`, `output_tokens`
  - Add validation: `filetype` must be "mp4", "pdf", or "html"
  - Add validation: `error` field is optional
  - Add validation: MP4 files should have blank `description`, PDF/HTML files should have extracted metadata in `description`
  - Add `final_title` computed property to extract the final title string from the title dict
  - Add `feed_eligible` computed property to check step map completion
  - Implement serialization/deserialization methods
  - Ensure serialization excludes computed properties (feed_eligible, final_title)
  - Add backwards compatibility: warn and ignore deprecated fields
- **Completion Criteria:**
  - Model validates correctly for valid inputs
  - Model rejects invalid inputs with appropriate errors
  - Model can serialize to/from JSON matching the exact schema
  - Error metadata can be created and validated
  - Deprecated fields trigger warnings when loaded
  - Title dict handled correctly (both dict and string formats)
  - Millisecond precision timestamps handled correctly

- [x] T2.3: Implement File Scanner (Batch Mode)
- **Description:** Create scanner.py with directory scanning functionality
- **File:** `src/local_feed_generator/scanner.py`
- **Actions:**
  - Implement `Scanner` class with `scan_directory()` method
  - Recursively walk `DATA_DIR` for files with extensions: `.mp4`, `.pdf`, `.html` (case-insensitive)
  - For each file, check if corresponding `.json` file exists in same directory
  - Return list of files that need processing (no JSON or JSON is outdated or incomplete)
  - Implement filename validation for special characters (log warnings only)
  - Implement orphaned JSON detection (JSON file exists but source file is missing)
  - Implement feed.json detection (skip feed.xml, articles.xml, feed.json, index.html, etc.)
  - **NEW:** Check for complete step maps when determining if processing is needed
  - **NEW:** Support millisecond precision in file modification time comparisons
- **Completion Criteria:**
  - Scanner correctly identifies files without JSON metadata
  - Scanner correctly identifies files with JSON metadata
  - Scanner returns absolute file paths and relative paths from DATA_DIR
  - Special character warnings are logged
  - Orphaned JSON files are detected and logged as warnings
  - Feed.json files are detected and skipped
  - Incomplete step maps trigger reprocessing

- [x] T2.4: Create Error Handling Utilities
- **Description:** Create utilities for consistent error handling
- **File:** `src/local_feed_generator/errors.py` (new)
- **Actions:**
  - Define custom exception classes: `ProcessingError`, `TranscriptionError`, `SummarizationError`, `ValidationError`
  - Create utility function to create error metadata from exception
  - Implement error logging helper with consistent format
  - **NEW:** Ensure error metadata includes step map preservation
- **Completion Criteria:**
  - Custom exceptions are defined and can be raised
  - Error metadata creation works for all error types
  - Error logging includes timestamp, file, and error message
  - Step maps are preserved in error metadata when available

---

## Phase 3: File Processing (Producer, Transcriber, Summarizer, Extractors)

### Objective
Implement the file processing pipeline including transcription, summarization, and metadata extraction with new features.

### Tasks

- [x] T3.1: Implement Producer Pipeline
- **Description:** Create producer.py as the orchestrator for file processing
- **File:** `src/local_feed_generator/producer.py`
- **Actions:**
  - Implement `Producer` class with dependencies on Transcriber, Summarizer, and extractors
  - Create `process_file()` method that:
    - Determines file type from extension
    - Dispatches to appropriate processing path
    - For MP4: call Transcriber (stores result in `content` field), then Summarizer (stores result in `summary` field), sets `description` to blank, then generate title, build title dict, apply module prefix
    - For PDF/HTML: call appropriate extractor (stores result in `description` and `content` fields), build title dict, set `summary` to null, apply module prefix
    - **NEW:** Build title as dict with sources (filename, extracted, generated, final)
    - **NEW:** Apply module name prefix to final title
    - **NEW:** Check for reflection.md and pass questions to summarizer
    - **NEW:** Extract audio metadata (duration, creation_time) using ffprobe
    - Writes JSON metadata file adjacent to source file with step maps
    - Handles errors and writes error metadata with preserved step maps
  - Implement `process_all()` method for batch processing
  - Create `generate_title()` helper method that uses LLM if source metadata unavailable
  - **NEW:** Create helper for module name prefixing
- **Dependencies:**
  - T2.2 (Metadata model)
  - T2.3 (Scanner)
  - T2.1 (Config)
- **Completion Criteria:**
  - Producer can process MP4, PDF, and HTML files through appropriate paths
  - JSON metadata files are created correctly with all new fields
  - Error handling writes error metadata with appropriate error details
  - Step maps are properly populated for each processing step
  - Title dict is properly built with all sources
  - Module prefixing works correctly
  - Reflection questions are passed to summarizer when available
  - Audio metadata is extracted when ffprobe is available

- [x] T3.2: Implement Transcription Service
- **Description:** Create transcriber.py for GPU-accelerated audio transcription
- **File:** `src/local_feed_generator/services/transcriber.py`
- **Actions:**
  - Implement `Transcriber` class
  - Load `faster-whisper` model with configuration from env vars:
    - Model: `TRANSCRIPTION_MODEL` (default: `large-v3-turbo`)
    - Device: `TRANSCRIPTION_DEVICE` (default: `cuda`)
    - Compute type: `TRANSCRIPTION_COMPUTE_TYPE` (default: `float16`)
  - Implement model singleton pattern (load once, reuse for all files)
  - Implement `transcribe()` method that:
    - Accepts file path as input
    - Returns markdown transcript without timestamps
    - Tracks processing in transcribe step map with timing (millisecond precision) and metrics
    - Handles errors gracefully
  - Add resource cleanup method for model unloading
  - **NEW:** Use millisecond precision for timestamps in step maps
  - **NEW:** Add CUDA availability check with clear error message
- **Dependencies:**
  - T2.1 (Config)
  - T2.4 (Error handling)
- **Completion Criteria:**
  - Transcriber loads model successfully with CUDA
  - Transcription produces clean markdown output without timestamps
  - Model is loaded only once and reused
  - Millisecond precision timestamps in step maps
  - Errors are handled and propagated appropriately
  - Step map is populated with status, timing (ms precision), and metrics

- [x] T3.3: Implement Summarization Service
- **Description:** Create summarizer.py for LLM-based summarization
- **File:** `src/local_feed_generator/services/summarizer.py`
- **Actions:**
  - Implement `Summarizer` class
  - Initialize OpenAI-compatible client with configuration from env vars:
    - Endpoint: `LLM_ENDPOINT`
    - API Key: `LLM_API_KEY`
    - Model: `LLM_MODEL`
    - **NEW:** System prompt from `SUMMARY_PROMPT` (required)
    - **NEW:** Reflection prompt from `SUMMARY_REFLECTION_PROMPT` (optional)
    - **NEW:** Title prompt from `TITLE_PROMPT` (required)
  - Implement `summarize()` method that:
    - Accepts transcript or extracted content text as input
    - Accepts optional `reflection_questions` parameter
    - Uses `SUMMARY_REFLECTION_PROMPT` when reflection_questions provided, else `SUMMARY_PROMPT`
    - Returns key takeaways as string
    - Tracks processing in summarize step map with timing (millisecond precision) and metrics
    - Handles API errors with retry logic (exponential backoff)
    - **NEW:** Works with both MP4 transcripts and PDF/HTML extracted content
  - Implement `generate_title()` method for generating titles from content using `TITLE_PROMPT`
  - **NEW:** Use millisecond precision for timestamps in step maps
- **Dependencies:**
  - T2.1 (Config)
  - T2.4 (Error handling)
- **Completion Criteria:**
  - Summarizer can connect to LLM endpoint
  - Summarization produces concise key takeaways
  - Title generation works for files without metadata
  - Retry logic handles transient errors
  - Step map is populated with status, timing (ms precision), and metrics
  - Reflection questions are properly prepended to transcript
  - Works with both MP4 transcripts and PDF/HTML extracted content

- [x] T3.4: Implement PDF Metadata and Content Extractor
- **Description:** Create extractor for PDF files
- **File:** `src/local_feed_generator/services/extractors.py` (new)
- **Actions:**
  - Implement `PDFExtractor` class (or add to services package)
  - Extract metadata from PDF files using `pypdf`
  - Extract: title, author, creation date, file size
  - **NEW:** Extract plaintext content from all pages
  - **NEW:** Store extracted metadata in `description` field
  - **NEW:** Store extracted content in `content` field
  - Set `summary` and `extracted_title` appropriately
  - Implement fallback to filename if no title found
  - Track processing in extract step map with timing (millisecond precision) and metrics
  - **NEW:** Remove control characters that are invalid in XML from extracted content
  - Return structured data compatible with Metadata model
- **Dependencies:**
  - T2.2 (Metadata model)
  - T3.3 (Summarizer for title generation fallback)
- **Completion Criteria:**
  - PDF metadata extraction works for files with metadata
  - Content extraction works from all pages
  - Title generation fallback works for files without metadata
  - Date is extracted in UNIX epoch with millisecond precision
  - File size is calculated correctly
  - Control characters are removed from extracted content
  - Step map is populated with status, timing (ms precision), and metrics

- [x] T3.5: Implement HTML Metadata and Content Extractor
- **Description:** Create extractor for HTML files
- **File:** `src/local_feed_generator/services/extractors.py`
- **Actions:**
  - Implement `HTMLExtractor` class
  - Extract title from `<title>` tag using BeautifulSoup
  - **NEW:** Store extracted title in `extracted_title` field (not used as actual title)
  - Extract other metadata if available (description meta tags, date, etc.)
  - **NEW:** Extract plaintext content from HTML body (tags removed)
  - Store extracted metadata in `description` field
  - **NEW:** Store extracted content in `content` field
  - Set `summary` to null for HTML files
  - Implement fallback to filename if no title found
  - Track processing in extract step map with timing (millisecond precision) and metrics
  - **NEW:** Remove control characters that are invalid in XML from extracted content
  - Return structured data compatible with Metadata model
- **Dependencies:**
  - T2.2 (Metadata model)
  - T3.3 (Summarizer for title generation fallback)
- **Completion Criteria:**
  - HTML title extraction works from `<title>` tag
  - Content extraction works from body
  - Title generation fallback works for files without title
  - Date is extracted or uses file modification time with ms precision
  - File size is calculated correctly
  - Control characters are removed from extracted content
  - Step map is populated with status, timing (ms precision), and metrics

- [x] T3.6: Implement Audio Metadata Extractor
- **Description:** Create function to extract audio metadata using ffprobe
- **File:** `src/local_feed_generator/services/extractors.py`
- **Actions:**
  - Implement `get_audio_metadata()` function
  - Use subprocess to call ffprobe command-line tool
  - Parse JSON output from ffprobe
  - Extract: type (format name), duration (float seconds), creation_time (ISO-8601 string), bit_rate (optional)
  - Handle ffprobe not installed gracefully (return None)
  - **NEW:** Duration extracted with float precision
  - **NEW:** Creation_time extracted as ISO-8601 string
- **Dependencies:**
  - None (uses subprocess)
- **Completion Criteria:**
  - Audio metadata extraction works when ffprobe is installed
  - Graceful handling when ffprobe is not installed
  - Duration and creation_time extracted correctly
  - All fields returned in structured format

- [x] T3.7: Implement Remote Transcription Service
- **Description:** Create RemoteTranscriber class for API-based transcription
- **File:** `src/local_feed_generator/services/remote_transcriber.py` (new)
- **Actions:**
  - Implement `RemoteTranscriber` class with OpenAI-compatible client
  - Initialize client with TRANSCRIPTION_ENDPOINT and TRANSCRIPTION_API_KEY
  - Implement retry logic with exponential backoff and jitter
  - Implement `transcribe()` method that:
    - Uploads audio file to remote endpoint
    - Handles both text and JSON response formats (vLLM returns JSON)
    - Tracks processing in transcribe step map
    - Passes TRANSCRIPTION_REMOTE_MODEL if configured, else TRANSCRIPTION_MODEL
  - Handle various API errors (connection, authentication, rate limiting, etc.)
  - **NEW:** Use millisecond precision for timestamps in step maps
- **Dependencies:**
  - T2.1 (Config)
  - T2.4 (Error handling)
- **Completion Criteria:**
  - RemoteTranscriber class can transcribe audio files via API
  - Both text and JSON responses are parsed correctly
  - API errors handled gracefully
  - Millisecond precision timestamps in step maps
  - Retry logic works with exponential backoff

---

## Phase 4: Feed Generation

### Objective
Implement RSS feed generation for podcast and articles feeds with new features.

### Tasks

- [x] T4.1: Implement Feed Generator Base Class
- **Description:** Create feeds.py with base feed generation functionality
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Implement `FeedGenerator` class
  - Accept `DATA_DIR` and `BASE_URL` from configuration
  - Scan `DATA_DIR` recursively for all `*.json` files
  - Filter entries by `feed_eligible` computed property (which checks step map statuses: MP4 needs transcribe.status="complete" AND summarize.status="complete", PDF/HTML needs extract.status="complete")
  - Implement sorting: folder name (ascending), then filename (ascending)
  - Calculate absolute URLs: `BASE_URL` + `filepath`
  - **NEW:** Implement markdown to HTML conversion with XSS protection
  - **NEW:** Implement feed.json detection for per-folder feeds
  - **NEW:** Filter metadata by folder for per-folder feeds
  - **NEW:** Implement conditional feed generation: only generate feeds where feed.json exists AND applicable files present
- **Dependencies:**
  - T2.1 (Config)
  - T2.2 (Metadata model)
  - T2.3 (Scanner for JSON discovery)
- **Completion Criteria:**
  - Feed generator can discover all valid JSON metadata files
  - Error entries are excluded from feeds
  - Items are sorted correctly by folder then filename
  - URLs are computed correctly from BASE_URL + filepath
  - Markdown to HTML conversion works
  - Feed eligibility computed from step map statuses
  - Per-folder feed detection works
  - Conditional feed generation implemented

- [x] T4.2: Implement Podcast Feed Generation
- **Description:** Create podcast.xml feed generation
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Implement `generate_podcast_feed()` method
  - Use `feedgen` library with podcast extension
  - Include all MP4 files with `feed_eligible = true` from metadata
  - For each item:
    - Title from metadata title dict's "final" key
    - Description from `summary` field (LLM-generated key takeaways)
    - **NEW:** Convert markdown in summary to HTML
    - Link: absolute URL to source file
    - Enclosure with URL, type="audio/mp4", length (file size from `size` field)
    - Publication date from metadata `date` field (UNIX epoch to datetime)
    - GUID based on filepath
  - Write feed to `DATA_DIR/podcast.xml` (or custom output_dir)
  - **NEW:** Support per-folder feed generation when output_dir is subfolder
  - **NEW:** Accept custom title and description from feed.json
  - **NEW:** Only generate if MP4 files present in the folder
- **Dependencies:**
  - T4.1 (Feed generator base)
- **Completion Criteria:**
  - Podcast feed is valid RSS with podcast extensions
  - All MP4 files with complete step maps are included in feed
  - Enclosure tags have correct URLs, types, and lengths
  - Markdown in descriptions converted to HTML
  - Feed is written to correct location
  - Per-folder feeds work with custom titles/descriptions
  - Only generated if MP4 files present

- [x] T4.3: Implement Articles Feed Generation
- **Description:** Create articles.xml feed generation
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Implement `generate_articles_feed()` method
  - Use `feedgen` library for RSS
  - Include all PDF/HTML files with `feed_eligible = true` from metadata
  - For each item:
    - Title from metadata title dict's "final" key
    - **NEW:** Description from `content` field (if available), otherwise `description` field
    - **NEW:** Convert markdown in content/description to HTML
    - Link: absolute URL to source file
    - Publication date from metadata `date` field (UNIX epoch to datetime)
    - GUID based on filepath
  - Write feed to `DATA_DIR/articles.xml` (or custom output_dir)
  - **NEW:** Support per-folder feed generation when output_dir is subfolder
  - **NEW:** Accept custom title and description from feed.json
  - **NEW:** Only generate if PDF/HTML files present in the folder
- **Dependencies:**
  - T4.1 (Feed generator base)
- **Completion Criteria:**
  - Articles feed is valid RSS
  - All PDF/HTML files with complete step maps are included in feed
  - Content field used as description when available
  - Markdown in descriptions converted to HTML
  - Feed is written to correct location
  - Per-folder feeds work with custom titles/descriptions
  - Only generated if PDF/HTML files present

- [x] T4.4: Implement Feed Generation for Per-Folder Feeds
- **Description:** Create support for feed.json-based per-folder feeds
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Implement feed.json discovery (recursive)
  - Implement metadata filtering by folder
  - Implement `generate_all_feeds()` method that:
    - Generates root-level feeds (backward compatibility)
    - Discovers all feed.json files
    - For each feed.json, generates feeds in that folder with items from that folder tree
    - Uses custom title/description from feed.json
    - **NEW:** Only generates podcast.xml if MP4 files present
    - **NEW:** Only generates articles.xml if PDF/HTML files present
    - **NEW:** Does NOT generate root-level feeds unless feed.json exists in root
  - Ensure root-level feeds still generated when feed.json exists in root
- **Dependencies:**
  - T4.1 (Feed generator base)
  - T4.2 (Podcast feed)
  - T4.3 (Articles feed)
- **Completion Criteria:**
  - Root-level feeds generated only when feed.json exists in root
  - Per-folder feeds generated when feed.json exists in subfolder
  - Only items from folder tree included in per-folder feeds
  - Custom title/description from feed.json used
  - Both podcast.xml and articles.xml generated per folder (if applicable items exist)
  - podcast.xml only generated if MP4 files present
  - articles.xml only generated if PDF/HTML files present

- [x] T4.5: Implement Markdown to HTML Conversion
- **Description:** Create markdown to HTML converter with XSS protection
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Implement `_markdown_to_html()` function
  - Try to use `markdown` library if installed
  - Implement fallback simple converter for basic markdown:
    - Headers (# to ######)
    - Bold (**text**)
    - Italic (*text*, _text_)
    - Links [text](url) with safe URL scheme validation
    - Lists (ordered and unordered)
    - Paragraphs
  - Ensure all text is HTML-escaped to prevent XSS
  - Only allow safe URL schemes: http, https, mailto, relative paths, fragments
  - Handle None input gracefully (return empty string)
- **Completion Criteria:**
  - Markdown conversion works with markdown library
  - Fallback converter handles basic markdown
  - All output is XSS-safe
  - Unsafe URL schemes are rejected or textified
  - None input returns empty string

- [x] T4.6: Implement Feed Generation Method
- **Description:** Create main method to generate both feeds
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Implement `generate_all_feeds()` method
  - Generate root-level podcast and articles feeds (only if feed.json exists in root)
  - Discover and generate per-folder feeds (only where feed.json exists and applicable files present)
  - Handle errors for individual feed generation
  - Ensure feeds are always regenerated (idempotent)
  - **NEW:** Track which feeds were generated for index.html
- **Completion Criteria:**
  - Both root-level feeds generated with a single method call (when feed.json exists)
  - Per-folder feeds generated when feed.json exists AND applicable files present
  - Errors in one feed don't prevent others from being generated
  - Feed generation list available for index.html

- [x] T4.7: Implement Index.html Generation (NEW)
- **Description:** Create index.html generator for feed discovery
- **File:** `src/local_feed_generator/index.py` (new)
- **Actions:**
  - Implement `IndexGenerator` class
  - Scan for feed.json files recursively
  - For each folder with feed.json:
    - Check which feeds exist (podcast.xml, articles.xml) based on conditional generation rules
    - Generate index.html with feed titles, descriptions, and URLs from feed.json
    - Include hierarchical navigation to parent/child index pages
    - Use simple, clean HTML
  - Implement `generate_all_index_pages()` method
  - Handle errors gracefully
- **Dependencies:**
  - T4.6 (Feed generation)
- **Completion Criteria:**
  - index.html generated in each folder with feed.json
  - Only lists feeds that actually exist (based on conditional generation)
  - Includes feed titles and descriptions from feed.json
  - Provides hierarchical navigation
  - Simple HTML that works with any browser

---

## Phase 5: CLI Integration

### Objective
Implement command-line interface for all processing modes.

### Tasks

- [x] T5.1: Implement CLI Framework
- **Description:** Create cli.py with Click framework
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Create Click command group `local-feed-generator`
  - Implement shared options: `--data-dir`, `--config-file`
  - Validate options and load configuration
  - Handle errors and exit codes appropriately
  - **NEW:** Update help text to mention conditional feed generation and index.html
- **Dependencies:**
  - T2.1 (Config)
- **Completion Criteria:**
  - CLI command group is importable
  - Shared options work correctly
  - Configuration loading works from CLI

- [x] T5.2: Implement Process Command
- **Description:** Create `process` command for batch mode
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Implement `process` command with required `--data-dir` option
  - Orchestrate: Scanner \=> Producer \=> Feed Generator \=> Index Generator
  - Process all files without JSON metadata or with incomplete step maps
  - Generate both feeds (root-level and per-folder) after all files are processed, based on feed.json configuration
  - Generate index.html pages after feed generation
  - Output progress and summary
  - Exit with appropriate code (0 for success, 1 for errors)
- **Dependencies:**
  - T2.3 (Scanner)
  - T3.1 (Producer)
  - T4.6 (Feed generation)
  - T4.7 (Index generation)
- **Completion Criteria:**
  - Command processes all files in batch
  - Both root-level and per-folder feeds generated (where feed.json exists and files present)
  - Index.html pages generated
  - Progress is shown during processing
  - Exit code reflects success/failure

- [x] T5.3: Implement Watch Command
- **Description:** Create `watch` command for incremental mode
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Implement `watch` command with required `--data-dir` option
  - Start watchdog Observer on DATA_DIR
  - Register event handlers for:
    - File created (for .mp4, .pdf, .html)
    - File modified (for .mp4, .pdf, .html)
    - JSON file created/modified (check for orphaned)
  - On file event: Process file \=> Regenerate feeds \=> Regenerate index.html
  - On JSON event without source: Log warning
  - On other events: Ignore
  - **NEW:** Implement debouncing per file (1 second interval)
  - **NEW:** Implement periodic rescan fallback (every 5 minutes)
  - **NEW:** Implement error rate limiting in watch mode
  - Include graceful shutdown on SIGINT/SIGTERM
- **Dependencies:**
  - T2.3 (Scanner for initial scan)
  - T3.1 (Producer)
  - T4.6 (Feed generation)
  - T4.7 (Index generation)
- **Completion Criteria:**
  - Watch mode starts and monitors directory
  - New files are processed automatically
  - Both root-level and per-folder feeds regenerated after each file processing (where applicable)
  - Index.html regenerated after each file processing
  - Warnings are logged for orphaned JSON
  - Debouncing prevents duplicate processing
  - Periodic rescan works as fallback
  - Errors in one file don't stop watcher
  - Shutdown is graceful on Ctrl+C

- [x] T5.4: Implement Feeds Command
- **Description:** Create `feeds` command for feed-only generation
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Implement `feeds` command with required `--data-dir` option
  - Skip file processing
  - Read all existing JSON metadata files
  - Generate both root-level and per-folder feeds (only where feed.json exists and applicable files present)
  - Generate index.html pages
  - Finalize JSON metadata by computing feed eligibility from step maps
  - Output summary of feeds generated (counts for podcast and articles)
- **Dependencies:**
  - T2.2 (Metadata model)
  - T4.6 (Feed generation)
  - T4.7 (Index generation)
- **Completion Criteria:**
  - Command generates both root-level and per-folder feeds without processing files (where applicable)
  - Existing JSON metadata is used
  - Feed eligibility computed from step map statuses
  - Index.html pages generated
  - Summary shows items in each feed

- [x] T5.5: Implement Validate Command
- **Description:** Create `validate` command for metadata validation
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Implement `validate` command with required `--data-dir` option
  - Scan for all JSON metadata files
  - Validate each against the schema (including new fields: version, title dict, duration, creation_time, content, step maps)
  - Check for deprecated field usage (warn on transcript, status dict, transcribe_status, etc.)
  - Check for orphaned JSON files
  - Check for special characters in filenames
  - Finalize JSON metadata by setting step completion statuses and computing feed eligibility
  - Output validation report with errors and warnings
  - Exit code: 0 for all valid, 1 for errors found
- **Dependencies:**
  - T2.2 (Metadata model)
  - T2.3 (Scanner)
- **Completion Criteria:**
  - Command validates all JSON metadata files against new schema
  - Schema validation works correctly for version 2.0 with title dict
  - New fields validated (title dict, duration, creation_time, content, step maps with ms precision)
  - Deprecated field warnings are shown
  - Orphaned JSON files are detected
  - Special character warnings are shown
  - Feed eligibility computed from step map statuses
  - Exit code reflects validation status

- [x] T5.6: Package Entry Point Configuration
- **Description:** Configure package for CLI execution
- **File:** `pyproject.toml`
- **Actions:**
  - Add entry point for `local-feed-generator` CLI
  - Configure to call CLI main function
  - Ensure `uv run local-feed-generator` works
- **Completion Criteria:**
  - CLI can be invoked via `uv run local-feed-generator`
  - All subcommands are available
  - Help text is displayed correctly

---

## Phase 6: Incremental Mode (Watch)

### Objective
Enhance and test the incremental processing mode with robust file watching.

### Tasks

- [x] T6.1: Implement Initial Scan for Watch Mode
- **Description:** Scan directory on watch mode start
- **File:** `src/local_feed_generator/scanner.py` and `cli.py`
- **Actions:**
  - Implement initial scan in watch command
  - Process all files without JSON metadata on startup
  - Finalize JSON metadata by computing feed eligibility from step maps
  - Generate both root-level and per-folder feeds after initial scan (where feed.json exists and files present)
  - Generate index.html pages after initial scan
  - Track processed files to avoid re-processing on first event
- **Dependencies:**
  - T2.3 (Scanner)
  - T3.1 (Producer)
  - T4.6 (Feed generation)
  - T4.7 (Index generation)
- **Completion Criteria:**
  - Initial scan processes all unprocessed files
  - Both root-level and per-folder feeds generated after initial scan (where applicable)
  - Index.html pages generated
  - No duplicate processing on startup
  - Feed eligibility computed from step maps

- [x] T6.2: Implement Event Debouncing
- **Description:** Prevent rapid successive events from triggering multiple processes
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Implement debounce mechanism for file system events
  - Configurable debounce interval (1 second default)
  - Process file only after debounce period expires
  - Handle multiple events for same file
  - Track debounce state per file path
- **Completion Criteria:**
  - Multiple rapid events for same file are debounced
  - File is processed only once after debounce period
  - Different files are tracked independently

- [x] T6.3: Implement Periodic Rescan Fallback
- **Description:** Fallback directory scanning for missed watchdog events
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Implement periodic rescan (every 5 minutes by default)
  - Configurable via RESCAN_INTERVAL environment variable
  - Detect files that were added/modified since last scan
  - Process detected files
  - Regenerate feeds and index.html
  - Log rescan events
- **Dependencies:**
  - T2.3 (Scanner)
- **Completion Criteria:**
  - Periodic rescan runs at configured interval
  - Missed files are detected and processed
  - Rescan events are logged
  - Both root-level and per-folder feeds regenerated (where applicable)
  - Index.html regenerated

- [x] T6.4: Enhanced Error Handling in Watch Mode
- **Description:** Ensure watch mode handles errors gracefully
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Catch exceptions in event handlers
  - Log errors without crashing the watcher
  - Continue monitoring after errors
  - Implement error rate limiting (max 10 errors per minute)
  - Suppress detailed errors after rate limit reached
- **Completion Criteria:**
  - Single file processing error doesn't stop watch mode
  - Errors are logged with sufficient detail
  - Watcher continues monitoring after errors
  - Rate limiting prevents log flooding

---

## Phase 7: Validation and Error Handling

### Objective
Implement comprehensive validation and robust error handling throughout the system.

### Tasks

- [x] T7.1: Implement Metadata Schema Validation
- **Description:** Ensure all metadata files conform to schema
- **File:** `src/local_feed_generator/metadata.py`
- **Actions:**
  - Validate required fields for each filetype (version, title, filename, filepath, filetype, date)
  - Validate field types and formats (UNIX epoch float for date, float for duration, ISO8601 for creation_time)
  - Validate conditional fields (content/summary only for MP4, description for PDF/HTML)
  - Validate step map structure (transcribe, summarize, extract objects with proper fields)
  - **NEW:** Validate title dict structure (filename, extracted, generated, final keys)
  - **NEW:** Validate millisecond precision timestamps
  - Add validation for MP4: description should be blank, summary and content should be present for complete files
  - Add validation for PDF/HTML: description should be present, summary should be null (unless explicitly summarized), content should be present for complete files
- **Dependencies:**
  - T2.2 (Metadata model)
- **Completion Criteria:**
  - Invalid metadata files are rejected
  - Validation provides clear error messages
  - All required fields are validated
  - Title dict structure is validated
  - Filetype-specific field usage is validated
  - Millisecond precision timestamps are validated

- [x] T7.2: Implement XML Feed Validation
- **Description:** Validate generated RSS feeds
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Add validation that feeds are well-formed XML
  - Validate RSS compliance
  - Validate podcast enclosure tags
  - Check for proper XML escaping of special characters
  - Verify podcast feed uses summary field for descriptions
  - Verify articles feed uses content or description field for descriptions
  - **NEW:** Validate markdown to HTML conversion output
  - **NEW:** Validate safe URL scheme usage in links
  - **NEW:** Validate conditional feed generation (no empty feeds)
- **Completion Criteria:**
  - Generated feeds are valid XML
  - RSS compliance is verified
  - Special characters in filenames and content are properly escaped
  - Feed descriptions come from correct fields
  - Markdown to HTML conversion produces valid HTML
  - Safe URL schemes validated
  - No empty feeds generated

- [x] T7.3: Implement End-to-End Error Handling
- **Description:** Comprehensive error handling across all modules
- **Files:** All modules
- **Actions:**
  - Review all error paths in each module
  - Ensure errors are caught and logged appropriately
  - Ensure error metadata is written for all processing errors with step map status set to null
  - Ensure no unhandled exceptions reach CLI
  - Implement appropriate exit codes for CLI
  - **NEW:** Ensure step maps are preserved in error metadata
  - **NEW:** Ensure conditional feed generation handles errors gracefully
- **Completion Criteria:**
  - All errors are caught and logged
  - Error metadata is written for all processing failures with step map status = null
  - Step maps are preserved in error state
  - CLI returns appropriate exit codes
  - No unhandled exceptions in normal operation
  - Conditional feed generation errors handled gracefully

- [x] T7.4: Implement Integration Tests
- **Description:** Create end-to-end tests for critical workflows
- **File:** `tests/` (new directory)
- **Actions:**
  - Create test for batch processing workflow
  - Create test for watch mode startup
  - Create test for feed generation with valid metadata
  - Create test for error handling
  - Create test for special character handling
  - Create test for orphaned JSON detection
  - **NEW:** Create test for new schema fields (title dict, duration, creation_time, content, step maps with ms precision)
  - **NEW:** Create test for field usage by filetype (MP4 vs PDF/HTML)
  - **NEW:** Create test for markdown to HTML conversion
  - **NEW:** Create test for per-folder feed generation
  - **NEW:** Create test for conditional feed generation (no feeds without feed.json, no podcast without MP4, no articles without PDF/HTML)
  - **NEW:** Create test for index.html generation
- **Completion Criteria:**
  - Integration tests cover critical workflows
  - Tests can be run with `python -m pytest`
  - Tests verify correct behavior for all scenarios
  - Tests verify new schema fields are present and validated
  - Tests verify conditional feed generation
  - Tests verify index.html generation

- [x] T7.5: Create Validation Scripts
- **Description:** Scripts for manual validation and testing
- **File:** `scripts/` (new directory)
- **Actions:**
  - Create script to validate a sample data directory
  - Create script to test all CLI commands
  - Create script to verify feed validity
  - Create script to generate sample test files
  - **NEW:** Create script to verify new schema fields
  - **NEW:** Create script to test per-folder feeds
  - **NEW:** Create script to test conditional feed generation
  - **NEW:** Create script to test index.html generation
- **Completion Criteria:**
  - Validation scripts can be run manually
  - Scripts provide clear output for verification
  - Scripts verify new schema fields
  - Scripts test per-folder feed generation
  - Scripts test conditional feed generation
  - Scripts test index.html generation

---

## Phase 8: Remote Transcription Support

### Objective
Add support for remote OpenAI-compatible transcription API as an alternative to local faster-whisper transcription.

### Tasks

- [x] T8.1: Add remote transcription configuration variables
- **Description:** Add transcription_endpoint, transcription_api_key, transcription_remote_model config variables
- **File:** `src/local_feed_generator/config.py`
- **Actions:**
  - Add new configuration variables for remote transcription endpoint
  - Add API key configuration for authentication
  - Add remote model selection configuration
- **Completion Criteria:**
  - Configuration variables are defined and loadable
  - Default values are sensible (None for optional, defaults for others)
  - Variables are documented in config schema

- [x] T8.2: Create RemoteTranscriber class for API-based transcription
- **Description:** Implement RemoteTranscriber class that calls OpenAI-compatible API
- **File:** `src/local_feed_generator/services/remote_transcriber.py`
- **Actions:**
  - Implement `RemoteTranscriber` class with OpenAI-compatible client
  - Initialize client with TRANSCRIPTION_ENDPOINT and TRANSCRIPTION_API_KEY
  - Implement retry logic with exponential backoff and jitter
  - Implement `transcribe()` method that:
    - Uploads audio file to remote endpoint
    - Handles both text and JSON response formats (vLLM returns JSON)
    - Tracks processing in transcribe step map
    - Passes TRANSCRIPTION_REMOTE_MODEL if configured, else TRANSCRIPTION_MODEL
  - Handle various API errors (connection, authentication, rate limiting, etc.)
  - **NEW:** Use millisecond precision for timestamps in step maps
- **Completion Criteria:**
  - RemoteTranscriber class can transcribe audio files via API
  - Both text and JSON responses handled
  - API errors handled gracefully
  - Millisecond precision timestamps in step maps
  - Retry logic works with exponential backoff

- [x] T8.3: Update Transcriber to support dual-mode (local/remote) transcription
- **Description:** Modify Transcriber to select between local and remote transcription
- **File:** `src/local_feed_generator/services/transcriber.py`
- **Actions:**
  - Add mode detection based on TRANSCRIPTION_ENDPOINT configuration
  - Integrate RemoteTranscriber for remote mode
  - Maintain existing faster-whisper for local mode
  - Ensure proper error handling in both modes
  - **NEW:** Use millisecond precision for timestamps in both modes
- **Completion Criteria:**
  - Transcriber selects correct mode based on config
  - Both local and remote modes work
  - Millisecond precision timestamps in both modes
  - Error handling works in both modes

- [x] T8.4: Update Producer to track remote transcription in metadata
- **Description:** Ensure metadata tracks which transcription method was used
- **File:** `src/local_feed_generator/producer.py`
- **Actions:**
  - Metadata includes transcription model info in step map's model field
  - Track whether local or remote transcription was used
  - Include transcription model information in step map
- **Completion Criteria:**
  - Metadata includes transcription method info in model field
  - Both local and remote transcriptions are tracked
  - Model information is recorded in step map

- [x] T8.5: Document vLLM server configuration requirement
- **Description:** Document server-side VLLM_MAX_AUDIO_DECODE_DURATION_S requirement
- **Files:** README.md, design.md, specs.md, tasks.md
- **Actions:**
  - Add troubleshooting entry for "Audio exceeds maximum allowed duration" errors
  - Document configuration requirement in README Configuration section
  - Update design.md with server-side requirements
  - Update specs.md with server-side configuration needs
  - Mark remote transcription testing tasks as complete
- **Completion Criteria:**
  - All documentation reflects the remote transcription feature and its requirements
  - Users understand the vLLM server configuration needed for longer audio files
  - Testing status is updated in tasks.md

- [x] T8.6: Verify integration with existing pipeline
- **Description:** Ensure remote transcription works end-to-end
- **Files:** All pipeline modules
- **Actions:**
  - Test complete pipeline with remote transcription enabled
  - Verify metadata is correctly populated with new schema
  - Verify feeds are generated correctly
  - Test fallback behavior (no automatic fallback from remote to local)
- **Completion Criteria:**
  - End-to-end pipeline works with remote transcription
  - Metadata is correctly updated with new schema fields
  - Feeds are valid
  - No automatic fallback between modes

---

## Phase 9: New Features Implementation

### Objective
Implement all new features: title dict, millisecond timestamps, per-folder feeds, reflection support, markdown conversion, module prefixing, content extraction, conditional feed generation, index pages.

### Tasks

- [x] T9.1: Implement Title as Dict with Source Tracking
- **Description:** Change title field from string to dict
- **Files:** `metadata.py`, `producer.py`, `extractors.py`, `feeds.py`
- **Actions:**
  - Update Metadata model to accept title as dict or string
  - Implement title dict building in producer for all file types
  - Add final_title computed property to extract final title string
  - Update feed generator to use final_title from title dict
  - Ensure backward compatibility with string titles
- **Completion Criteria:**
  - Title dict with all sources (filename, extracted, generated, final) works
  - Final title extraction works correctly
  - Feed generation uses final title
  - Backward compatibility maintained

- [x] T9.2: Implement Millisecond Precision Timestamps
- **Description:** Change date timestamps from integer seconds to float with ms precision
- **Files:** `metadata.py`, `producer.py`, `scanner.py`, `extractors.py`
- **Actions:**
  - Update date field in Metadata model to accept float
  - Update step map timing fields to use float with ms precision
  - Update all timestamp generation to use datetime.now(UTC).timestamp()
  - Update ffprobe parsing to handle float duration
- **Completion Criteria:**
  - All timestamps have millisecond precision
  - Feed generation correctly converts ms timestamps to datetime
  - Backward compatibility with integer timestamps maintained

- [x] T9.3: Implement Per-Folder Feed Generation
- **Description:** Add support for feed.json files in subfolders
- **Files:** `feeds.py`, `cli.py`
- **Actions:**
  - Implement feed.json discovery (recursive)
  - Implement metadata filtering by folder
  - Update generate_all_feeds to create per-folder feeds
  - Ensure root-level feeds still generated when feed.json exists in root
  - Add title and description from feed.json to feeds
- **Completion Criteria:**
  - Per-folder feeds generated when feed.json exists
  - Only items from folder tree included
  - Custom title/description used
  - Root-level feeds generated when feed.json exists in root

- [x] T9.4: Implement Reflection Questions Support
- **Description:** Add support for reflection.md files
- **Files:** `producer.py`, `summarizer.py`
- **Actions:**
  - Implement reflection.md detection in producer
  - Pass reflection questions to summarizer
  - Use SUMMARY_REFLECTION_PROMPT when reflection questions available
  - Prepend reflection questions to transcript or extracted content
  - **NEW:** Works with both MP4 and PDF/HTML files
- **Completion Criteria:**
  - Reflection questions detected and loaded
  - Questions passed to summarizer
  - SUMMARY_REFLECTION_PROMPT used when available
  - Questions prepended to transcript
  - Works with MP4, PDF, and HTML files

- [x] T9.5: Implement Markdown to HTML Conversion
- **Description:** Add markdown to HTML conversion for feed descriptions
- **Files:** `feeds.py`
- **Actions:**
  - Implement _markdown_to_html function
  - Try to use markdown library if available
  - Implement fallback simple converter
  - Add XSS protection (HTML escaping, safe URL schemes)
  - Update feed generation to convert descriptions
- **Completion Criteria:**
  - Markdown to HTML conversion works
  - Fallback converter handles basic markdown
  - XSS protection implemented
  - Safe URL schemes validated

- [x] T9.6: Implement Module Name Prefixing
- **Description:** Add automatic module name prefixing for titles
- **Files:** `producer.py`
- **Actions:**
  - Implement _should_prepend_module function
  - Implement _apply_module_prefix function
  - Apply prefix to final title for all file types
  - Check if filename already starts with module name
  - Normalize module name and filename for comparison
- **Completion Criteria:**
  - Module prefixing works correctly
  - Doesn't double-prefix when filename already contains module name
  - Works with underscores in module names and filenames

- [x] T9.7: Implement Content Extraction for PDF/HTML
- **Description:** Extract full plaintext content from PDF and HTML files
- **Files:** `extractors.py`, `producer.py`
- **Actions:**
  - Extract text from all PDF pages
  - Extract text from HTML body
  - Store in content field
  - Remove control characters
  - Update producer to handle content field for PDF/HTML
- **Completion Criteria:**
  - Content extracted from PDF and HTML files
  - Stored in content field
  - Control characters removed
  - Feed generation uses content for descriptions

- [x] T9.8: Implement Control Character Removal
- **Description:** Remove control characters that are invalid in XML
- **Files:** `extractors.py`
- **Actions:**
  - Implement _sanitize_content function
  - Use regex to remove characters 0x00-0x1f (except tab, newline, CR) and 0x7f
  - Apply to all extracted content (PDF, HTML)
- **Completion Criteria:**
  - Control characters removed from extracted content
  - Tab, newline, CR preserved
  - XML generation doesn't fail due to control characters

- [x] T9.9: Implement Audio Metadata Extraction with ffprobe
- **Description:** Extract duration and creation_time from MP4 files
- **Files:** `extractors.py`, `producer.py`
- **Actions:**
  - Implement get_audio_metadata function
  - Use subprocess to call ffprobe
  - Parse JSON output
  - Extract duration, creation_time, bit_rate, format type
  - Handle ffprobe not installed gracefully
  - Store in metadata fields
- **Completion Criteria:**
  - Audio metadata extracted when ffprobe available
  - Graceful handling when ffprobe not available
  - Duration and creation_time fields populated correctly

- [x] T9.10: Add New Configuration Variables
- **Description:** Add SUMMARY_PROMPT, TITLE_PROMPT, SUMMARY_REFLECTION_PROMPT to config
- **Files:** `config.py`, `summarizer.py`
- **Actions:**
  - Add new fields to Settings class
  - Add SUMMARY_PROMPT as required (no default)
  - Add TITLE_PROMPT as required (no default)
  - Add SUMMARY_REFLECTION_PROMPT as optional (defaults to None)
  - Update summarizer to use new prompt variables
- **Completion Criteria:**
  - New config variables defined and validated
  - Required variables enforced
  - Summarizer uses new prompt variables

- [x] T9.11: Implement Conditional Feed Generation (NEW)
- **Description:** Only generate feeds where feed.json exists AND applicable files are present
- **Files:** `feeds.py`, `cli.py`
- **Actions:**
  - Update feed generation to check for feed.json before generating feeds
  - Check for MP4 files before generating podcast.xml
  - Check for PDF/HTML files before generating articles.xml
  - Skip feed generation if requirements not met
  - Update root-level feed generation: only if feed.json exists in root AND applicable files present
- **Completion Criteria:**
  - Feeds only generated where feed.json exists
  - podcast.xml only generated if MP4 files present
  - articles.xml only generated if PDF/HTML files present
  - Root-level feeds require feed.json in root

- [x] T9.12: Implement Index.html Generation (NEW)
- **Description:** Generate index.html pages in folders with feed.json
- **Files:** `index.py` (new), `feeds.py`
- **Actions:**
  - Create new index.py module
  - Implement IndexGenerator class
  - Scan for feed.json files
  - Generate index.html with feed links
  - Include hierarchical navigation
  - Call from feed generation workflow
- **Completion Criteria:**
  - index.html generated in each folder with feed.json
  - Lists all available feeds in the folder
  - Includes hierarchical navigation to subfolders
  - Simple HTML works with any browser

- [x] T9.13: Update Summarize Command for PDF/HTML (NEW)
- **Description:** Allow summarize command to process PDF/HTML files
- **Files:** `summarizer.py`, `producer.py`, `cli.py`
- **Actions:**
  - Update Summarizer to work with both transcripts and extracted content
  - Update Producer to support summarization for PDF/HTML files
  - Add check for extracted content in PDF/HTML files
  - Pass content to summarizer for summary generation
  - Update CLI to allow --file with PDF/HTML files
- **Completion Criteria:**
  - summarize command works with MP4 files
  - summarize command works with PDF/HTML files
  - Reflection questions work with PDF/HTML files
  - Summary stored in summary field for PDF/HTML files

---

## Dependencies Summary

### Phase Dependencies

```
Phase 1: Project Setup and Dependencies
  \u251c\u2500 No dependencies (foundation)
  
Phase 2: Core Modules (Config, Scanner, Metadata)
  \u251c\u2500 Depends on Phase 1
  
Phase 3: File Processing (Producer, Transcriber, Summarizer, Extractors)
  \u251c\u2500 Depends on Phase 1 (dependencies)
  \u251c\u2500 Depends on Phase 2 (config, metadata, scanner)
  
Phase 4: Feed Generation
  \u251c\u2500 Depends on Phase 2 (metadata model)
  \u251c\u2500 Depends on Phase 3 (producer for JSON files)
  
Phase 5: CLI Integration
  \u251c\u2500 Depends on Phase 2 (config)
  \u251c\u2500 Depends on Phase 3 (producer)
  \u251c\u2500 Depends on Phase 4 (feed generation)
  
Phase 6: Incremental Mode (Watch)
  \u251c\u2500 Depends on Phase 2 (scanner)
  \u251c\u2500 Depends on Phase 3 (producer)
  \u251c\u2500 Depends on Phase 5 (CLI)
  
Phase 7: Validation and Error Handling
  \u251c\u2500 Depends on all previous phases
  
Phase 8: Remote Transcription Support
  \u251c\u2500 Depends on Phase 2 (config)
  \u251c\u2500 Depends on Phase 3 (transcriber)
  
Phase 9: New Features Implementation
  \u251c\u2500 Depends on Phase 2 (metadata model)
  \u251c\u2500 Depends on Phase 3 (all processing modules)
  \u251c\u2500 Depends on Phase 4 (feed generation)
```

### Task Dependencies

The tasks are ordered within each phase to respect dependencies. Tasks within a phase may be worked on in parallel if they have no dependencies on each other.

### Parallelizable Tasks

The following tasks can be worked on concurrently:

**Phase 1:** T1.1, T1.2, T1.3 (minor dependency: T1.2 needs structure from T1.1)

**Phase 2:** T2.1, T2.2, T2.4 can be parallel; T2.3 depends on T2.1 and T2.2

**Phase 3:** T3.2 (Transcriber), T3.3 (Summarizer), T3.4 (PDF Extractor), T3.5 (HTML Extractor), T3.6 (Audio Metadata), T3.7 (RemoteTranscriber) can all be parallel; T3.1 depends on all of them

**Phase 4:** T4.2, T4.3, T4.5, T4.7 can be parallel; T4.1, T4.4, T4.6 are sequential

**Phase 5:** T5.2, T5.3, T5.4, T5.5 can be parallel; T5.1 and T5.6 are sequential

**Phase 6:** Sequential (depends on watch mode implementation)

**Phase 7:** T7.1, T7.2, T7.5 can be parallel; T7.3, T7.4 are sequential

**Phase 8:** T8.1, T8.2 can be parallel; T8.3 depends on T8.1 and T8.2; T8.4 depends on T8.3

**Phase 9:** T9.1-T9.13 can be worked on in parallel as they touch different files

---

## Execution Guidelines

### Prerequisites Discovery
- Verify Nvidia GPU and CUDA installation before Phase 3
- Verify LLM endpoint is accessible before testing summarization
- Verify all dependencies install correctly in Phase 1
- Verify ffprobe is installed (optional) for audio metadata extraction
- Verify markdown library is installed (optional) for enhanced conversion

### Quality Gates
- Each phase should be validated before moving to dependent phases
- All tasks in Phase 1 must be complete before starting Phase 2
- Core modules (Phase 2) must be validated before starting Phase 3
- Transcriber and Summarizer must work independently before integrating in Producer
- All new features (Phase 9) must be implemented and tested

### Testing Recommendations
- Test with sample MP4, PDF, and HTML files in Phase 3
- Validate feed output in Phase 4 with XML validator
- Test CLI commands in Phase 5 with various inputs
- Test watch mode in Phase 6 with file system events
- Run comprehensive validation in Phase 7
- Test remote transcription in Phase 8
- Test per-folder feeds, reflection, markdown conversion in Phase 9
- Test conditional feed generation and index.html in Phase 9

---

## Key Constraints Reminders

### Schema Requirements (Version 2.0)
1. **Title field**: Must be a dict with keys: `filename`, `extracted`, `generated`, `final` (or a plain string for backward compatibility)
2. **Content field**: Used for both MP4 (transcript) and PDF/HTML (extracted text)
3. **Step maps**: Processing tracked via `transcribe`, `summarize`, `extract` objects (not unified `status` dict)
4. **Description behavior**: MP4 files have blank `description`, PDF/HTML have extracted metadata in `description`
5. **Summary behavior**: MP4 files have key takeaways in `summary`, PDF/HTML have null `summary` (unless explicitly summarized via summarize command)
6. **extracted_title field**: NEW - stores source metadata title separately
7. **New fields**: `duration` (float, MP4 only), `creation_time` (ISO-8601, MP4 only)
8. **Millisecond precision**: All date/timestamp fields use float with 3 decimal places
9. **Backwards compatibility**: Warn and ignore deprecated fields
10. **Control characters**: Invalid XML characters removed from extracted content

### New Features Requirements
1. **Per-folder feeds**: Require feed.json file in folder
2. **Reflection questions**: Require reflection.md file in folder
3. **Module prefixing**: Automatic based on parent folder name
4. **Markdown conversion**: Automatic for feed descriptions with XSS protection
5. **Content extraction**: PDF/HTML files have content extracted to content field
6. **Conditional feed generation**: Feeds only generated where feed.json exists AND applicable files present
7. **Index.html generation**: Auto-generated in folders with feed.json
8. **No root feeds without feed.json**: Root-level feeds require explicit feed.json configuration
9. **summarize command**: Works with MP4 (transcripts), PDF (extracted content), and HTML (extracted content)

### Other Constraints
1. **Dependencies:** Use uv for all dependency management
2. **Optional dependencies:** faster-whisper, av in separate group; markdown library optional
3. **Python:** Target Python 3.14 or later
4. **Transcription:** Use faster-whisper with large-v3-turbo, CUDA, float16 (local); or remote API (when configured)
5. **No quantization:** Best quality transcription, no quantization
6. **LLM:** OpenAI-compatible with 128K context
7. **Prompts:** SUMMARY_PROMPT and TITLE_PROMPT required; SUMMARY_REFLECTION_PROMPT optional
8. **System Prompt:** LLM_SYSTEM_PROMPT still used for general LLM configuration
9. **Feed Format:** RSS (not Atom) by default
10. **URLs:** Absolute URLs = BASE_URL + filepath
11. **Errors:** Error state written to JSON with step map status = null, excluded from feeds
12. **Directory Structure:** Follow design doc section 5.1

---

## Success Checklist

- [x] All phases completed
- [x] All tasks within each phase completed
- [x] All dependencies between tasks respected
- [x] All completion criteria met for each task
- [x] Error handling implemented for all error paths
- [x] XML feeds are valid and properly formatted
- [x] All CLI commands work as specified
- [x] Watch mode processes files incrementally
- [x] Feeds are regenerated after each file processing
- [x] Index.html pages are regenerated after each file processing
- [x] Special characters handled correctly (warnings only, no modification)
- [x] Orphaned JSON files detected and logged
- [x] All environment variables from design doc are supported
- [x] New schema fields (title dict, duration, creation_time, content, step maps with ms precision) implemented and validated
- [x] Field usage by filetype validated (MP4 vs PDF/HTML)
- [x] Feed eligibility computed from step map statuses
- [x] Backwards compatibility for deprecated fields implemented with warnings
- [x] New features implemented: per-folder feeds, reflection support, markdown conversion, module prefixing, content extraction
- [x] New features implemented: conditional feed generation, index.html generation, summarize for PDF/HTML
- [x] Control characters removed from extracted content
- [x] Audio metadata extracted via ffprobe when available
- [x] Millisecond precision timestamps throughout
- [x] Debouncing and periodic rescan in watch mode
- [x] No root feeds without feed.json
- [x] Conditional feed generation (podcast only with MP4, articles only with PDF/HTML)

---

## Completion Report

### Artifacts Updated
1. **README.md** - Updated with all new features, configuration, usage examples, conditional feed generation, index.html
2. **proposal.md** - Updated problem statement, scope, success criteria, conditional feed generation, index.html
3. **specs.md** - Updated with new functional requirements, conditional feed generation, index.html, summarize for PDF/HTML
4. **design.md** - Updated with new architecture, interfaces, data flow, conditional feed generation, index.html, summarize for PDF/HTML
5. **tasks.md** - Updated with new tasks for conditional feed generation, index.html, summarize for PDF/HTML, all marked complete

### New Features Implemented
1. \u2705 Title field as dict with source tracking
2. \u2705 New config variables (SUMMARY_PROMPT, TITLE_PROMPT, SUMMARY_REFLECTION_PROMPT)
3. \u2705 Reflection.md support
4. \u2705 Per-folder feed generation with feed.json
5. \u2705 Markdown to HTML conversion
6. \u2705 Millisecond precision timestamps
7. \u2705 Module name prefixing for titles
8. \u2705 Whitespace/control character normalization
9. \u2705 Content extraction for PDF/HTML files
10. \u2705 Audio metadata extraction via ffprobe
11. \u2705 Optional dependencies (faster-whisper, av, markdown)
12. \u2705 Conditional feed generation (feeds only where feed.json exists AND applicable files present)
13. \u2705 Index.html generation for feed discovery
14. \u2705 NO root feeds without feed.json
15. \u2705 podcast.xml only if MP4 files present
16. \u2705 articles.xml only if PDF/HTML files present
17. \u2705 summarize command processes PDF/HTML files

### Implementation Status
- **All required features:** \u2705 COMPLETED
- **All documentation:** \u2705 UPDATED
- **All tests:** \u2705 PASSED (as per integration tests)
- **Backward compatibility:** \u2705 MAINTAINED
- **Error handling:** \u2705 COMPREHENSIVE

### Notes
- Remote transcription is fully implemented and tested
- Per-folder feeds work with feed.json configuration
- Reflection questions are supported via reflection.md files
- Markdown to HTML conversion has both enhanced (with markdown library) and fallback implementations
- Control characters are automatically removed from extracted content to prevent XML errors
- All timestamps use millisecond precision
- Module name prefixing is automatic and intelligent (doesn't double-prefix)
- Audio metadata extraction via ffprobe is optional (graceful fallback if not installed)
- Conditional feed generation: feeds only generated where feed.json exists AND applicable file types present
- Index.html auto-generated in each folder with feed.json for easy feed discovery
- Root-level feeds require explicit feed.json configuration
- summarize command works with both MP4 transcripts and PDF/HTML extracted content
