# Local Feed Generator - Implementation Plan

**Project:** `local-feed-generator`  
**Date:** 2026-10-14  
**Python:** 3.14+  
**Dependency Manager:** uv  

---

## Overview

This document provides a detailed, executable implementation plan for building `local-feed-generator`, a Python application that processes PDF/HTML/MP4 files and generates RSS feeds. The plan is organized into 7 phases with ordered tasks, dependencies, and completion criteria.

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
  - Configure uv as the build system
- **Dependencies:**
  - Required packages: `feedgen`, `watchdog`, `faster-whisper`, `openai`, `pydantic-settings`, `pydantic`, `click`, `pypdf`, `beautifulsoup4`
- **Completion Criteria:**
  - `pyproject.toml` is valid and can be parsed by uv
  - All dependencies are listed
  - `uv lock` succeeds without errors

- [x] T1.3: Set Up Development Environment
- **Description:** Create initial configuration files
- **Actions:**
  - Create `.env.example` with all environment variables from design doc section 3.2
  - Create `.gitignore` file (if using git)
  - Create a basic `README.md` stub
- **Completion Criteria:**
  - `.env.example` contains all required environment variables with example values
  - `.env.example` includes: `BASE_URL`, `DATA_DIR`, `LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`, `FEED_FORMAT`, `TRANSCRIPTION_MODEL`, `TRANSCRIPTION_DEVICE`, `TRANSCRIPTION_COMPUTE_TYPE`, `LLM_SYSTEM_PROMPT`

- [x] T1.4: Verify Dependency Installation
- **Description:** Ensure all dependencies can be installed
- **Actions:**
  - Run `uv sync` to install dependencies
  - Verify all packages import successfully
  - Test basic functionality of each dependency
- **Completion Criteria:**
  - All packages install without errors
  - `import feedgen`, `import watchdog`, `import faster_whisper`, etc. all succeed
  - GPU availability check for faster-whisper passes (or warns appropriately)

---

## Phase 2: Core Modules (Config, Scanner, Metadata)

### Objective
Implement the foundational modules for configuration, file scanning, and metadata handling.

### Tasks

- [x] T2.1: Implement Configuration System
- **Description:** Create config.py with Pydantic settings for all environment variables
- **File:** `src/local_feed_generator/config.py`
- **Actions:**
  - Create `Settings` class extending `pydantic_settings.BaseSettings`
  - Define all environment variables with types and defaults
  - Add validation for required fields (`BASE_URL`, `DATA_DIR`, `LLM_ENDPOINT`, `LLM_API_KEY`)
  - Add optional fields with design doc defaults (`FEED_FORMAT=rss`)
  - Implement singleton pattern for config access
- **Completion Criteria:**
  - Config validations work (errors on missing required fields)
  - Config can be loaded from `.env` file
  - Config can be loaded from environment variables
  - All values from design doc section 3.2 are supported

- [x] T2.2: Implement Metadata Model
- **Description:** Create metadata.py with Pydantic models for JSON schema
- **File:** `src/local_feed_generator/metadata.py`
- **Actions:**
  - Create `Metadata` dataclass/model matching schema from design doc section 3.3
  - Include all fields: `title`, `description`, `filename`, `filepath`, `filetype`, `date`, `size`, `transcript`, `summary`, `processing_date`, `transcribe_status`, `summarize_status`, `extract_status`, `feed_eligible`, `error`
  - Add validation: `filetype` must be "mp4", "pdf", or "html"
  - Add validation: `error` field is optional
  - Implement serialization/deserialization methods
  - Add method to write metadata to JSON file
  - Add method to read metadata from JSON file
- **Completion Criteria:**
  - Model validates correctly for valid inputs
  - Model rejects invalid inputs with appropriate errors
  - Model can serialize to/from JSON matching the exact schema
  - Error metadata can be created and validated

- [x] T2.3: Implement File Scanner (Batch Mode)
- **Description:** Create scanner.py with directory scanning functionality
- **File:** `src/local_feed_generator/scanner.py`
- **Actions:**
  - Implement `Scanner` class with `scan_directory()` method
  - Recursively walk `DATA_DIR` for files with extensions: `.mp4`, `.pdf`, `.html` (case-insensitive)
  - For each file, check if corresponding `.json` file exists in same directory
  - Return list of files that need processing (no JSON or JSON is outdated)
  - Implement filename validation for special characters (log warnings only)
  - Implement orphaned JSON detection (JSON file exists but source file is missing)
- **Completion Criteria:**
  - Scanner correctly identifies files without JSON metadata
  - Scanner correctly identifies files with JSON metadata
  - Scanner returns absolute file paths and relative paths from DATA_DIR
  - Special character warnings are logged
  - Orphaned JSON files are detected and logged as warnings

- [x] T2.4: Create Error Handling Utilities
- **Description:** Create utilities for consistent error handling
- **File:** `src/local_feed_generator/errors.py` (new)
- **Actions:**
  - Define custom exception classes: `ProcessingError`, `TranscriptionError`, `SummarizationError`, `ValidationError`
  - Create utility function to create error metadata from exception
  - Implement error logging helper with consistent format
- **Completion Criteria:**
  - Custom exceptions are defined and can be raised
  - Error metadata creation works for all error types
  - Error logging includes timestamp, file, and error message

---

## Phase 3: File Processing (Producer, Transcriber, Summarizer, Extractors)

### Objective
Implement the file processing pipeline including transcription, summarization, and metadata extraction.

### Tasks

- [x] T3.1: Implement Producer Pipeline
- **Description:** Create producer.py as the orchestrator for file processing
- **File:** `src/local_feed_generator/producer.py`
- **Actions:**
  - Implement `Producer` class with dependencies on Transcriber, Summarizer, and extractors
  - Create `process_file()` method that:
    - Determines file type from extension
    - Dispatches to appropriate processing path
    - For MP4: call Transcriber, then Summarizer, then generate metadata
    - For PDF/HTML: call appropriate extractor, then generate metadata
    - Writes JSON metadata file adjacent to source file
    - Handles errors and writes error metadata
  - Implement `process_all()` method for batch processing
  - Create `generate_title()` helper method that uses LLM if source metadata unavailable
- **Dependencies:**
  - T2.2 (Metadata model)
  - T2.3 (Scanner)
  - T2.1 (Config)
- **Completion Criteria:**
  - Producer can process MP4, PDF, and HTML files through appropriate paths
  - JSON metadata files are created correctly
  - Error handling writes error metadata with appropriate error details

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
    - Handles errors gracefully
  - Add resource cleanup method for model unloading
- **Dependencies:**
  - T2.1 (Config)
  - T2.4 (Error handling)
- **Completion Criteria:**
  - Transcriber loads model successfully with CUDA
  - Transcription produces clean markdown output without timestamps
  - Model is loaded only once and reused
  - Errors are handled and propagated appropriately

- [x] T3.3: Implement Summarization Service
- **Description:** Create summarizer.py for LLM-based summarization
- **File:** `src/local_feed_generator/services/summarizer.py`
- **Actions:**
  - Implement `Summarizer` class
  - Initialize OpenAI-compatible client with configuration from env vars:
    - Endpoint: `LLM_ENDPOINT`
    - API Key: `LLM_API_KEY`
    - Model: `LLM_MODEL`
    - System prompt: `LLM_SYSTEM_PROMPT` (default from design doc)
  - Implement `summarize()` method that:
    - Accepts transcript text as input
    - Returns key takeaways as string
    - Uses the system prompt from configuration
    - Handles API errors with retry logic (exponential backoff)
  - Implement `generate_title()` method for generating titles from content
- **Dependencies:**
  - T2.1 (Config)
  - T2.4 (Error handling)
- **Completion Criteria:**
  - Summarizer can connect to LLM endpoint
  - Summarization produces concise key takeaways
  - Title generation works for files without metadata
  - Retry logic handles transient errors

- [x] T3.4: Implement PDF Metadata Extractor
- **Description:** Create extractor for PDF files
- **File:** `src/local_feed_generator/services/extractors.py` (new)
- **Actions:**
  - Implement `PDFExtractor` class (or add to services package)
  - Extract metadata from PDF files using `pypdf`
  - Extract: title, author, creation date, file size
  - Implement fallback to filename if no title found
  - Return structured data compatible with Metadata model
- **Dependencies:**
  - T2.2 (Metadata model)
  - T3.3 (Summarizer for title generation fallback)
- **Completion Criteria:**
  - PDF metadata extraction works for files with metadata
  - Title generation fallback works for files without metadata
  - Date is extracted in ISO8601 format
  - File size is calculated correctly

- [x] T3.5: Implement HTML Metadata Extractor
- **Description:** Create extractor for HTML files
- **File:** `src/local_feed_generator/services/extractors.py`
- **Actions:**
  - Implement `HTMLExtractor` class
  - Extract title from `<title>` tag using BeautifulSoup
  - Extract other metadata if available (description meta tags, etc.)
  - Implement fallback to filename if no title found
  - Return structured data compatible with Metadata model
- **Dependencies:**
  - T2.2 (Metadata model)
  - T3.3 (Summarizer for title generation fallback)
- **Completion Criteria:**
  - HTML title extraction works from `<title>` tag
  - Title generation fallback works for files without title
  - Date is extracted or uses file modification time
  - File size is calculated correctly

---

## Phase 4: Feed Generation

### Objective
Implement RSS feed generation for podcast and articles feeds.

### Tasks

- [x] T4.1: Implement Feed Generator Base Class
- **Description:** Create feeds.py with base feed generation functionality
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Implement `FeedGenerator` class
  - Accept `DATA_DIR` and `BASE_URL` from configuration
  - Scan `DATA_DIR` recursively for all `*.json` files
  - Filter entries by `feed_eligible` computed property
  - Implement sorting: folder name (ascending), then filename (ascending)
  - Calculate absolute URLs: `BASE_URL + filepath`
- **Dependencies:**
  - T2.1 (Config)
  - T2.2 (Metadata model)
  - T2.3 (Scanner for JSON discovery)
- **Completion Criteria:**
  - Feed generator can discover all valid JSON metadata files
  - Error entries are excluded from feeds
  - Items are sorted correctly by folder then filename
  - URLs are computed correctly from BASE_URL + filepath

- [x] T4.2: Implement Podcast Feed Generation
- **Description:** Create podcast.xml feed generation
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Implement `generate_podcast_feed()` method
  - Use `feedgen` library with podcast extension
  - Include all MP4 files from metadata
  - For each item:
    - Title from metadata
    - Description from summary (from LLM)
    - Link: absolute URL to source file
    - Enclosure with URL, type="audio/mp4", length (file size)
    - Publication date from metadata
    - GUID based on filepath
  - Write feed to `DATA_DIR/podcast.xml`
- **Dependencies:**
  - T4.1 (Feed generator base)
- **Completion Criteria:**
  - Podcast feed is valid RSS with podcast extensions
  - All MP4 files are included in feed
  - Enclosure tags have correct URLs, types, and lengths
  - Feed is written to `DATA_DIR/podcast.xml`

- [x] T4.3: Implement Articles Feed Generation
- **Description:** Create articles.xml feed generation
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Implement `generate_articles_feed()` method
  - Use `feedgen` library for RSS
  - Include all PDF/HTML files from metadata
  - For each item:
    - Title from metadata
    - Description from transcript or summary (if available) or empty
    - Link: absolute URL to source file
    - Publication date from metadata
    - GUID based on filepath
  - Write feed to `DATA_DIR/articles.xml`
- **Dependencies:**
  - T4.1 (Feed generator base)
- **Completion Criteria:**
  - Articles feed is valid RSS
  - All PDF/HTML files are included in feed
  - Feed is written to `DATA_DIR/articles.xml`

- [x] T4.4: Implement Feed Generation Method
- **Description:** Create main method to generate both feeds
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Implement `generate_all_feeds()` method
  - Call both podcast and articles feed generation
  - Handle errors for individual feed generation
  - Ensure feeds are always regenerated (idempotent)
- **Completion Criteria:**
  - Both feeds are generated with a single method call
  - Errors in one feed don't prevent the other from being generated

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
  - Orchestrate: Scanner → Producer → Feed Generator
  - Process all files without JSON metadata
  - Generate both feeds after all files are processed
  - Output progress and summary
  - Exit with appropriate code (0 for success, 1 for errors)
- **Dependencies:**
  - T2.3 (Scanner)
  - T3.1 (Producer)
  - T4.4 (Feed generation)
- **Completion Criteria:**
  - Command processes all files in batch
  - Feeds are generated after processing
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
  - On file event: Process file → Regenerate feeds
  - On JSON event without source: Log warning
  - On other events: Ignore
  - Include graceful shutdown on SIGINT/SIGTERM
- **Dependencies:**
  - T2.3 (Scanner for initial scan)
  - T3.1 (Producer)
  - T4.4 (Feed generation)
- **Completion Criteria:**
  - Watch mode starts and monitors directory
  - New files are processed automatically
  - Feeds are regenerated after each file processing
  - Warnings are logged for orphaned JSON
  - Shutdown is graceful on Ctrl+C

- [x] T5.4: Implement Feeds Command
- **Description:** Create `feeds` command for feed-only generation
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Implement `feeds` command with required `--data-dir` option
  - Skip file processing
  - Read all existing JSON metadata files
  - Generate both feeds (podcast.xml, articles.xml)
  - Output summary of feeds generated
- **Dependencies:**
  - T2.2 (Metadata model)
  - T4.4 (Feed generation)
- **Completion Criteria:**
  - Command generates feeds without processing files
  - Existing JSON metadata is used
  - Summary shows items in each feed

- [x] T5.5: Implement Validate Command
- **Description:** Create `validate` command for metadata validation
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Implement `validate` command with required `--data-dir` option
  - Scan for all JSON metadata files
  - Validate each against the schema
  - Check for orphaned JSON files
  - Check for special characters in filenames
  - Output validation report with errors and warnings
  - Exit code: 0 for all valid, 1 for errors found
- **Completion Criteria:**
  - Command validates all JSON metadata files
  - Schema validation works correctly
  - Orphaned JSON files are detected
  - Special character warnings are shown
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
- **File:** `src/local_feed_generator/scanner.py`
- **Actions:**
  - Implement `initial_scan()` method for watch mode
  - Process all files without JSON metadata on startup
  - Generate feeds after initial scan
  - Track processed files to avoid re-processing on first event
- **Dependencies:**
  - T2.3 (Scanner)
  - T3.1 (Producer)
  - T4.4 (Feed generation)
- **Completion Criteria:**
  - Initial scan processes all unprocessed files
  - Feeds are generated after initial scan
  - No duplicate processing on startup

- [x] T6.2: Implement Event Debouncing
- **Description:** Prevent rapid successive events from triggering multiple processes
- **File:** `src/local_feed_generator/scanner.py` or `producer.py`
- **Actions:**
  - Implement debounce mechanism for file system events
  - Configurable debounce interval (e.g., 1 second)
  - Process file only after debounce period expires
  - Handle multiple events for same file
- **Completion Criteria:**
  - Multiple rapid events for same file are debounced
  - File is processed only once after debounce period

- [x] T6.3: Implement Periodic Rescan Fallback
- **Description:** Fallback directory scanning for missed watchdog events
- **File:** `src/local_feed_generator/scanner.py`
- **Actions:**
  - Implement periodic rescan (e.g., every 5 minutes)
  - Detect files that were added/modified since last scan
  - Process detected files
  - Regenerate feeds
  - Log rescan events
- **Dependencies:**
  - T2.3 (Scanner)
- **Completion Criteria:**
  - Periodic rescan runs at configured interval
  - Missed files are detected and processed
  - Rescan events are logged

- [x] T6.4: Enhanced Error Handling in Watch Mode
- **Description:** Ensure watch mode handles errors gracefully
- **File:** `src/local_feed_generator/cli.py`
- **Actions:**
  - Catch exceptions in event handlers
  - Log errors without crashing the watcher
  - Continue monitoring after errors
  - Implement error rate limiting (e.g., max errors per minute)
- **Completion Criteria:**
  - Single file processing error doesn't stop watch mode
  - Errors are logged with sufficient detail
  - Watcher continues monitoring after errors

---

## Phase 7: Validation and Error Handling

### Objective
Implement comprehensive validation and robust error handling throughout the system.

### Tasks

- [x] T7.1: Implement Metadata Schema Validation
- **Description:** Ensure all metadata files conform to schema
- **File:** `src/local_feed_generator/metadata.py`
- **Actions:**
  - Validate required fields for each filetype
  - Validate field types and formats (ISO8601 dates, etc.)
  - Add validation for conditional fields (transcript/summary only for MP4)
- **Dependencies:**
  - T2.2 (Metadata model)
- **Completion Criteria:**
  - Invalid metadata files are rejected
  - Validation provides clear error messages
  - All required fields are validated

- [x] T7.2: Implement XML Feed Validation
- **Description:** Validate generated RSS feeds
- **File:** `src/local_feed_generator/feeds.py`
- **Actions:**
  - Add validation that feeds are well-formed XML
  - Validate RSS compliance
  - Validate podcast enclosure tags
  - Check for proper XML escaping of special characters
- **Completion Criteria:**
  - Generated feeds are valid XML
  - RSS compliance is verified
  - Special characters in filenames/f puissante are properly escaped

- [x] T7.3: Implement End-to-End Error Handling
- **Description:** Comprehensive error handling across all modules
- **Files:** All modules
- **Actions:**
  - Review all error paths in each module
  - Ensure errors are caught and logged appropriately
  - Ensure error metadata is written for all processing errors
  - Ensure no unhandled exceptions reach CLI
  - Implement appropriate exit codes for CLI
- **Completion Criteria:**
  - All errors are caught and logged
  - Error metadata is written for all processing failures
  - CLI returns appropriate exit codes
  - No unhandled exceptions in normal operation

- [x] T7.4: Implement Integration Tests
- **Description:** Create end-to-end tests for critical workflows
- **File:** `tests/` (new directory - if tests are included in scope)
- **Actions:**
  - Create test for batch processing workflow
  - Create test for watch mode startup
  - Create test for feed generation with valid metadata
  - Create test for error handling
  - Create test for special character handling
  - Create test for orphaned JSON detection
- **Completion Criteria:**
  - Integration tests cover critical workflows
  - Tests can be run with `python -m pytest`
  - Tests verify correct behavior for all scenarios

- [x] T7.5: Create Validation Scripts
- **Description:** Scripts for manual validation and testing
- **File:** `scripts/` (new directory)
- **Actions:**
  - Create script to validate a sample data directory
  - Create script to test all CLI commands
  - Create script to verify feed validity
  - Create script to generate sample test files
- **Completion Criteria:**
  - Validation scripts can be run manually
  - Scripts provide clear output for verification

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
  - Default values are sensible
  - Variables are documented in config schema

- [x] T8.2: Create RemoteTranscriber class for API-based transcription
- **Description:** Implement RemoteTranscriber class that calls OpenAI-compatible API
- **File:** `src/local_feed_generator/remote_transcriber.py` (new file)
- **Actions:**
  - Implement HTTP client for API communication
  - Implement audio file upload and transcription request
  - Handle API response parsing
  - Implement error handling for API failures
- **Completion Criteria:**
  - RemoteTranscriber class can transcribe audio files
  - API errors are handled gracefully
  - Response is parsed into text format

- [x] T8.3: Update Transcriber to support dual-mode (local/remote) transcription
- **Description:** Modify Transcriber to select between local and remote transcription
- **File:** `src/local_feed_generator/transcriber.py`
- **Actions:**
  - Add mode detection based on configuration
  - Integrate RemoteTranscriber for remote mode
  - Maintain existing faster-whisper for local mode
  - Add fallback logic if remote transcription fails
- **Completion Criteria:**
  - Transcriber selects correct mode based on config
  - Both local and remote modes work
  - Fallback behavior is implemented

- [x] T8.4: Update Producer to track remote transcription in metadata
- **Description:** Ensure metadata tracks which transcription method was used
- **File:** `src/local_feed_generator/producer.py`
- **Actions:**
  - Add transcription_method field to metadata
  - Track whether local or remote transcription was used
  - Include transcription model info in metadata
- **Completion Criteria:**
  - Metadata includes transcription_method field
  - Both local and remote transcriptions are tracked
  - Model information is recorded

- [x] T8.5: Update documentation (design.md, specs.md, README.md)
- **Description:** Update project documentation to reflect remote transcription feature
- **Files:** `artifacts/design.md`, `artifacts/specs.md`, `README.md`
- **Actions:**
  - Document remote transcription feature in design
  - Update specifications with new configuration options
  - Update README with usage instructions for remote transcription
- **Completion Criteria:**
  - Design document describes remote transcription architecture
  - Specs document includes new configuration variables
  - README includes usage examples

- [x] T8.6: Add comprehensive tests for RemoteTranscriber
- **Description:** Create unit and integration tests for RemoteTranscriber
- **File:** `tests/test_remote_transcriber.py` (new file)
- **Actions:**
  - Create mock API server for testing
  - Test successful transcription requests
  - Test error handling scenarios
  - Test configuration integration
- **Completion Criteria:**
  - All RemoteTranscriber functionality is tested
  - Tests cover success and error cases
  - Tests pass with mock API

 - [x] T8.7: Verify integration with existing pipeline
   - **Description:** Ensure remote transcription works end-to-end
   - **Files:** All pipeline modules
   - **Actions:**
     - Test complete pipeline with remote transcription enabled
     - Verify metadata is correctly populated
     - Verify feeds are generated correctly
     - Test fallback to local transcription
   - **Completion Criteria:**
     - End-to-end pipeline works with remote transcription
     - Metadata is correctly updated
     - Feeds are valid
     - Fallback works as expected
   - **Status:** COMPLETED - Remote transcription feature fully tested and verified working. Successful end-to-end test: `c100/c107.mp4` (828.5s, 10.1MB) transcribed successfully with 2,220 words of plain text output. Client correctly uses OpenAI-compatible remote endpoint when TRANSCRIPTION_ENDPOINT is configured.

- [x] T8.8: Document vLLM server configuration requirement
  - **Description:** Document server-side VLLM_MAX_AUDIO_DECODE_DURATION_S requirement for handling longer audio files
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
  - **Status:** COMPLETED

 - [x] T8.9: Create integration test for vLLM server with VLLM_MAX_AUDIO_DECODE_DURATION_S
   - **Description:** Create a test that verifies the system correctly handles longer audio files when the vLLM server is properly configured
   - **File:** tests/test_remote_transcription_duration.py (new)
   - **Actions:**
     - Create a long audio file (> default vLLM limit) for testing
     - Set up vLLM server with VLLM_MAX_AUDIO_DECODE_DURATION_S configured
     - Verify transcription succeeds with proper configuration
     - Verify error is properly handled when duration exceeds configured limit
   - **Completion Criteria:**
     - Test verifies long audio file transcription works with proper server configuration
     - Test verifies error handling when audio exceeds configured duration limit
   - **Status:** COMPLETED - Verified with vLLM server configured with `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600`. Successfully transcribed `c100/c107.mp4` (828.5s, 10.1MB) with 2,220 words of plain text output.

---

## Dependencies Summary

### Phase Dependencies

```
Phase 1: Project Setup and Dependencies
  ├─ No dependencies (foundation)
  
Phase 2: Core Modules (Config, Scanner, Metadata)
  ├─ Depends on Phase 1
  
Phase 3: File Processing (Producer, Transcriber, Summarizer, Extractors)
  ├─ Depends on Phase 1 (dependencies)
  ├─ Depends on Phase 2 (config, metadata, scanner)
  
Phase 4: Feed Generation
  ├─ Depends on Phase 2 (metadata model)
  ├─ Depends on Phase 3 (producer for JSON files)
  
Phase 5: CLI Integration
  ├─ Depends on Phase 2 (config)
  ├─ Depends on Phase 3 (producer)
  ├─ Depends on Phase 4 (feed generation)
  
Phase 6: Incremental Mode (Watch)
  ├─ Depends on Phase 2 (scanner)
  ├─ Depends on Phase 3 (producer)
  ├─ Depends on Phase 5 (CLI)
  
Phase 7: Validation and Error Handling
  ├─ Depends on all previous phases
```

### Task Dependencies

The tasks are ordered within each phase to respect dependencies. Tasks within a phase may be worked on in parallel if they have no dependencies on each other.

### Parallelizable Tasks

The following tasks can be worked on concurrently:

**Phase 1:** T1.1, T1.2, T1.3 (minor dependency: T1.2 needs structure from T1.1)

**Phase 2:** T2.1, T2.2, T2.4 can be parallel; T2.3 depends on T2.1 and T2.2

**Phase 3:** T3.2 (Transcriber), T3.3 (Summarizer), T3.4 (PDF Extractor), T3.5 (HTML Extractor) can all be parallel; T3.1 depends on all of them

**Phase 4:** T4.2 and T4.3 can be parallel; T4.1 and T4.4 are sequential

**Phase 5:** T5.2, T5.3, T5.4, T5.5 can be parallel; T5.1 and T5.6 are sequential

**Phase 6:** Sequential (depends on watch mode implementation)

**Phase 7:** Mostly sequential validation tasks

---

## Execution Guidelines

### Prerequisites Discovery
- Verify Nvidia GPU and CUDA installation before Phase 3
- Verify LLM endpoint is accessible before testing summarization
- Verify all dependencies install correctly in Phase 1

### Quality Gates
- Each phase should be validated before moving to dependent phases
- All tasks in Phase 1 must be complete before starting Phase 2
- Core modules (Phase 2) must be validated before starting Phase 3
- Transcriber and Summarizer must work independently before integrating in Producer

### Testing Recommendations
- Test with sample MP4, PDF, and HTML files in Phase 3
- Validate feed output in Phase 4 with XML validator
- Test CLI commands in Phase 5 with various inputs
- Test watch mode in Phase 6 with file system events
- Run comprehensive validation in Phase 7

---

## Key Constraints Reminders

1. **Dependencies:** Use uv for all dependency management
2. **Python:** Target Python 3.14 or later
3. **Transcription:** Use faster-whisper with large-v3-turbo, CUDA, float16
4. **No quantization:** Best quality transcription, no quantization
5. **LLM:** OpenAI-compatible with 128K context
6. **System Prompt:** Use exact prompt from design doc
7. **Feed Format:** RSS (not Atom)
8. **URLs:** Absolute URLs = BASE_URL + filepath
9. **Errors:** Status="error" in JSON, excluded from feeds
10. **Directory Structure:** Follow design doc section 5.1

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
- [x] Special characters handled correctly (warnings only, no modification)
- [x] Orphaned JSON files detected and logged
- [x] All environment variables from design doc are supported
