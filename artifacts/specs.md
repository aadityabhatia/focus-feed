# Education Content Feed System - Specifications

## Overview

See [Proposal](./proposal.md) for problem statement, user intent, and scope. This document specifies detailed functional and non-functional requirements.

---

## Functional Requirements

### Input & File Management

**FR-001:** System SHALL process files from `data` directory and subfolders

**FR-002:** System SHALL support batch mode (all files) and incremental mode (watch for new/updated files)

**FR-003:** System SHALL skip files with existing JSON metadata where all applicable processing steps are complete

**FR-004:** System SHALL treat files without corresponding JSON as new files requiring processing

**FR-005:** System SHALL generate warnings for orphaned JSON files and filenames with special characters

**FR-006:** System SHALL accept input files: PDF, HTML, MP4

### JSON Metadata Schema

**FR-007:** For each input file, generate JSON metadata file with same name + `.json` extension in same directory

**FR-008:** JSON metadata SHALL contain:
- `title`: dict with keys `filename`, `extracted`, `generated`
- `extracted_title`: source metadata title (optional)
- `description`: blank for MP4; extracted metadata for PDF/HTML
- `summary`: LLM-generated key takeaways for MP4; null for PDF/HTML unless summarized
- `content`: full transcript for MP4; extracted plaintext for PDF/HTML
- `filename`, `filepath`, `filetype`
- `date`: UNIX epoch float (millisecond precision)
- `publication_date`: UNIX epoch float (optional, PDF/HTML metadata)
- `duration`: float seconds (MP4 only)
- `creation_time`: ISO8601 string (MP4 only)
- `size`: bytes
- `hash`: SHA256 (auto-populated during feed generation if missing)
- Step maps: `transcribe`, `summarize`, `extract` with status, timing, metrics
- `error`: string (optional, if processing failed)

**FR-009:** Title precedence: filename < extracted < generated (generated highest)

**FR-010:** System SHALL sanitize filename key in title dict: remove extension, replace underscores with spaces

**FR-011:** System SHALL automatically prepend parent folder name to final title when it doesn't already start with it (case-insensitive, underscores normalized)

**FR-012:** System SHALL remove control characters (0x00-0x1f except tab/newline/CR, and 0x7f) from extracted PDF/HTML content

**FR-013:** System SHALL compute SHA256 hash and store in `hash` field when: (1) first processing a file, OR (2) generating feeds for feed-eligible items missing hash

**FR-014:** For MP4: description blank, summary = key takeaways, content = full transcript

**FR-015:** For PDF/HTML: description = extracted metadata, content = extracted plaintext, summary = null unless explicitly summarized

### Feed Eligibility

**FR-016:** feed_eligible = (transcribe.status == "complete" AND summarize.status == "complete") for MP4

**FR-017:** feed_eligible = (extract.status == "complete") for PDF/HTML

**FR-018:** Items with feed_eligible = false SHALL be excluded from feeds

**FR-019:** Deprecated fields (`transcribe_status`, `summarize_status`, `extract_status`, `stage`, `status` dict, `transcript`, `processing_date`, `completed_at`, `started_at`) SHALL be silently ignored when loaded

### Audio Processing

**FR-020:** System SHALL transcribe MP4 to markdown using local faster-whisper OR remote OpenAI-compatible API when `TRANSCRIPTION_ENDPOINT` configured

**FR-021:** System SHALL generate key takeaway summaries from audio transcripts using LLM

**FR-022:** System SHALL feed entire transcript to LLM for summarization

**FR-023:** System SHALL extract audio metadata (duration, creation_time) using ffprobe when available

**FR-024:** When `TRANSCRIPTION_ENDPOINT` configured, use remote API; otherwise use local faster-whisper

**FR-025:** If `TRANSCRIPTION_REMOTE_MODEL` configured, pass as model parameter to remote endpoint

**FR-026:** System SHALL handle remote API errors and record in JSON error field

### Document Processing

**FR-027:** System SHALL extract metadata AND content from PDF/HTML

**FR-028:** System SHALL generate summaries for PDF/HTML when `summarize` command explicitly invoked

**FR-029:** For PDF/HTML, use `publication_date` from document metadata for feed pubDate when available, falling back to `date` field

### Reflection Support

**FR-030:** System SHALL check for `reflection.md` in source file's directory and use content as reflection questions for summarization

**FR-031:** `reflection.md` is optional; files can be summarized without it

**FR-032:** System SHALL prepend reflection questions to transcript when summarizing

**FR-033:** System SHALL use filetype-based reflection prompts when reflection.md exists:
- MP4: `PROMPT_PODCAST_REFLECTION` (fallback: `PROMPT_PODCAST`)
- PDF/HTML: `PROMPT_ARTICLE_REFLECTION` (fallback: `PROMPT_ARTICLE`)

### Feed Generation

**FR-034:** System SHALL generate `podcast.xml` (MP4) and `articles.xml` (PDF/HTML) based on feed.json configuration

**FR-035:** Feed items SHALL include: title, description (MP4: summary; PDF/HTML: summary or description), link (BASE_URL + filepath), pubDate (MP4: date; PDF/HTML: publication_date or date), guid (last 12 chars of hash or filepath)

**FR-036:** Feed items SHALL be sorted by folder name, then filename (alphabetical)

**FR-037:** System SHALL convert markdown in descriptions to HTML with XSS protection (safe URL schemes only)

**FR-038:** Per-folder feeds: when folder contains `feed.json`, generate `podcast.xml` and `articles.xml` in that folder only for items in subtree

**FR-039:** Conditional generation: `podcast.xml` only if feed.json exists AND at least one MP4 with feed_eligible=true exists in subtree

**FR-040:** Conditional generation: `articles.xml` only if feed.json exists AND at least one PDF/HTML with feed_eligible=true exists in subtree

**FR-041:** System SHALL generate root-level `index.html` listing all feeds from all folders containing feed.json

**FR-042:** Root-level feeds SHALL NOT be generated unless feed.json exists in root of data/ directory

**FR-043:** System SHALL ensure all feed-eligible items have hash field before generating feeds

**FR-044:** feed.json SHALL support: title (required), description, language, author, link, image (optional)

### Configuration & Environment

**FR-045:** System SHALL use environment variables for configuration

**FR-046:** Required environment variables:
- `BASE_URL`: Root URL for feed enclosure links
- `DATA_DIR`: Path to data directory
- `LLM_ENDPOINT`: LLM API endpoint URL
- `LLM_API_KEY`: LLM API key
- `LLM_MODEL`: Model identifier for summarization
- `PROMPT_ARTICLE`: Article summarization prompt
- `PROMPT_PODCAST`: Podcast summarization prompt
- `PROMPT_TITLE`: Title generation prompt

**FR-047:** Optional environment variables:
- `PROMPT_ARTICLE_REFLECTION`: Article summarization with reflection (fallback: PROMPT_ARTICLE)
- `PROMPT_PODCAST_REFLECTION`: Podcast summarization with reflection (fallback: PROMPT_PODCAST)
- `TRANSCRIPTION_ENDPOINT`: Remote transcription endpoint
- `TRANSCRIPTION_API_KEY`: Remote transcription API key
- `TRANSCRIPTION_MODEL`: Local transcription model (default: large-v3-turbo)
- `TRANSCRIPTION_DEVICE`: Device for transcription (default: cuda)
- `TRANSCRIPTION_COMPUTE_TYPE`: Compute type (default: float16)
- `TRANSCRIPTION_REMOTE_MODEL`: Remote transcription model override

### CLI Interface

**FR-048:** System SHALL provide CLI commands:
- `process`: Process all files (batch mode)
- `watch`: Watch directory for changes (incremental mode)
- `transcribe`: Transcribe audio files
- `summarize`: Generate summaries from transcripts or extracted content
- `extract`: Extract metadata from PDF/HTML files
- `feeds`: Generate RSS feeds and index.html
- `validate`: Validate metadata files
- `create-feed`: Create template feed.json

**FR-049:** All commands SHALL accept:
- `--data-dir <path>`: Path to data directory (default: DATA_DIR from config)
- `--config-file <path>`: Path to .env-style configuration file
- `--file <path>`: Process specific file (overrides --data-dir)
- `--force`: Force execution even if step appears complete

**FR-050:** `transcribe`, `summarize`, `extract` commands SHALL support `--dry-run` flag

**FR-051:** `--force` flag behavior:
- `transcribe`: Re-transcribe even if complete
- `summarize`: Re-summarize even if complete
- `extract`: Re-extract even if complete
- `process`: Re-process all steps
- `feeds`: Always regenerate feeds and index.html
- `validate`: Always perform validation

**FR-052:** `summarize` command SHALL work with MP4 (transcripts) and PDF/HTML (extracted content)

---

## Non-Functional Requirements

**NFR-001:** All transcription and summarization SHALL be performed locally by default

**NFR-002:** System SHALL NOT use cloud-based services by default

**NFR-003:** System SHALL leverage Nvidia GPUs for efficient model inference when available

**NFR-004:** System SHALL use open-weight models for transcription

**NFR-005:** System SHALL use OpenAI-compatible LLMs for summarization

**NFR-006:** Summaries SHALL be concise key takeaways

**NFR-007:** Transcripts SHALL be in human-readable markdown format

**NFR-008:** In incremental mode, system SHALL detect and process new files within reasonable timeframe

**NFR-009:** System SHALL handle file system events gracefully

**NFR-010:** System SHALL implement debouncing in watch mode

**NFR-011:** System SHALL implement periodic rescan as fallback for unreliable file system events

---

## Constraints

**C-001:** All processing local by default; remote transcription via OpenAI-compatible API supported when configured

**C-002:** Open-weight models required for local processing; remote endpoint may use any model

**C-003:** Feeds placed in `data` directory; only generated where feed.json exists and applicable files present

**C-004:** Input files and JSON metadata must maintain existing directory structure

**C-005:** Existing web server serves data directory; no deployment needed

**C-006:** Special characters in filenames preserved (warnings only, no modifications)

**C-007:** All endpoint examples MUST use example.com domain (RFC 2606)

**C-008:** Title field SHALL be dict with keys: filename, extracted, generated

**C-009:** Title precedence: filename < extracted < generated

**C-010:** Dates SHALL be stored as UNIX epoch floats with millisecond precision

**C-011:** Per-folder feeds require feed.json in the folder

**C-012:** Root feeds require feed.json in root data directory

**C-013:** Feeds only generated if applicable files exist (podcast.xml: MP4; articles.xml: PDF/HTML)

**C-014:** `index.html` maintained at root level only, listing all feeds from all folders

---

## Assumptions

**A-001:** `data` directory exists and is accessible for read/write operations

**A-002:** Existing web server configured to serve files from `data` directory

**A-003:** Open-weight transcription models and OpenAI-compatible LLMs available and configured

**A-004:** Nvidia GPUs available for the system to use

**A-005:** Source files have sufficient quality for accurate transcription

**A-006:** vLLM server configured with `VLLM_MAX_AUDIO_DECODE_DURATION_S` when using remote transcription

---

## Open Questions

None - all resolved.

---

## Risks

| ID | Risk | Mitigation |
|----|------|------------|
| R-001 | Transcription quality may vary with open-weight models | Use established models; accept basic quality for personal use |
| R-002 | LLM summarization may produce inconsistent results | Use tested system prompts; allow config updates |
| R-003 | File watching may miss events on some systems | Implement periodic directory scanning as fallback |
| R-004 | Special characters in filenames may cause parsing issues | Properly escape all output; warn user |
| R-005 | Large audio files may take significant time to transcribe | No time constraints; acceptable for personal use |
| R-006 | Control characters in PDF/HTML may cause XML failure | Automatically remove control characters during extraction |

---

## Metadata Schema (Current)

```json
{
  "title": {
    "filename": "lecture",
    "extracted": "Lecture 1",
    "generated": "LLM Title"
  },
  "extracted_title": "Lecture 1",
  "description": "",
  "summary": "Key takeaways from LLM...",
  "content": "Full transcript or extracted text...",
  "filename": "lecture.mp4",
  "filepath": "c100/lecture.mp4",
  "filetype": "mp4",
  "date": 1710000000.123,
  "publication_date": 1710000000.0,
  "duration": 45.5,
  "creation_time": "2024-01-01T00:00:00.000000Z",
  "size": 12345678,
  "hash": "a1b2c3d4e5f6...",
  "transcribe": {
    "status": "complete",
    "start_time": 1792096505.123,
    "end_time": 1792096515.456,
    "input_size": 12345678,
    "output_size": 456789,
    "words": 452,
    "model": "large-v3-turbo",
    "input_tokens": null,
    "output_tokens": null
  },
  "summarize": {
    "status": "complete",
    "start_time": 1792096515.123,
    "end_time": 1792096520.456,
    "input_size": 456789,
    "output_size": 1997,
    "words": 345,
    "model": "mistralai/Mistral-7B-Instruct",
    "input_tokens": 890,
    "output_tokens": 123
  },
  "extract": null,
  "error": "string"  // Optional
}
```

**Feed Inclusion:**
- MP4: `feed_eligible = (transcribe.status == "complete" AND summarize.status == "complete")`
- PDF/HTML: `feed_eligible = (extract.status == "complete")`

**Field Usage:**
- MP4: `description = ""`, `summary` = key takeaways, `content` = transcript
- PDF/HTML: `description` = extracted metadata, `content` = plaintext, `summary` = null unless summarized

---

## Feed Generation Rules

- **Conditional:** podcast.xml only if feed.json exists + MP4 feed_eligible=true files present
- **Conditional:** articles.xml only if feed.json exists + PDF/HTML feed_eligible=true files present
- **Root index:** index.html generated at root listing all feeds from all folders with feed.json
- **No root feeds without feed.json** in data directory root

---

## feed.json Format

```json
{
  "title": "Required: Feed title",
  "description": "Optional: Feed description",
  "language": "Optional: en-us",
  "author": "Optional: Author name",
  "link": "Optional: Feed URL",
  "image": {
    "url": "Optional: Image URL",
    "title": "Optional: Image title",
    "link": "Optional: Image link"
  }
}
```

---

## Environment Variables Summary

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| BASE_URL | Yes | - | Root URL for feed links |
| DATA_DIR | Yes | - | Path to data directory |
| LLM_ENDPOINT | Yes | - | LLM API endpoint |
| LLM_API_KEY | Yes | - | LLM API key |
| LLM_MODEL | Yes | - | LLM model for summarization |
| PROMPT_ARTICLE | Yes | - | Article summarization prompt |
| PROMPT_PODCAST | Yes | - | Podcast summarization prompt |
| PROMPT_TITLE | Yes | - | Title generation prompt |
| PROMPT_ARTICLE_REFLECTION | No | PROMPT_ARTICLE | Article with reflection |
| PROMPT_PODCAST_REFLECTION | No | PROMPT_PODCAST | Podcast with reflection |
| TRANSCRIPTION_ENDPOINT | No | - | Remote transcription endpoint |
| TRANSCRIPTION_API_KEY | No | - | Remote transcription API key |
| TRANSCRIPTION_MODEL | No | large-v3-turbo | Transcription model |
| TRANSCRIPTION_DEVICE | No | cuda | Transcription device |
| TRANSCRIPTION_COMPUTE_TYPE | No | float16 | Transcription compute type |
| TRANSCRIPTION_REMOTE_MODEL | No | - | Remote transcription model override |
