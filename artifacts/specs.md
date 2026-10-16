# Education Content Feed System - Specifications

## Problem Statement

Education content exists as disconnected files (PDF, HTML, MP4) in a `data` directory with no unified access method. Users cannot efficiently consume this content across devices through standard applications. No coherent processing pipeline exists to transform these files into consumable formats.

## Scope

This system processes education content files in the `data` directory and generates RSS feeds placed alongside them for personal consumption via mobile apps on an existing internal web server. The system supports per-folder feed generation, enhanced metadata extraction, conditional feed generation, and a root-level index page for feed discovery.

---

## Functional Requirements

### Input & File Management

**FR-001**: The system SHALL process files from a `data` directory, including subfolders

**FR-002**: The system SHALL support two processing modes:
- **Batch mode**: Process all files in `data` directory
- **Incremental mode**: Watch for new files or updates to existing JSON files

**FR-003**: The system SHALL skip files that have corresponding JSON metadata files already present with all applicable processing steps complete (feed_eligible=true)

**FR-004**: The system SHALL treat files without corresponding JSON files as new files requiring processing

**FR-005**: The system SHALL generate warnings for:
- Orphaned JSON files (JSON exists but source file is missing)
- Filenames containing special characters (no changes to filenames, just warnings)

**FR-006**: The system SHALL accept input files of the following types:
- PDF documents
- HTML documents
- MP4 audio files

### JSON Metadata

**FR-007**: For each input file, the system SHALL generate a JSON metadata file with the same name and `.json` extension appended (e.g., `document.pdf` \=> `document.pdf.json`)

**FR-008**: JSON metadata files SHALL be placed in the same directory as their source files

**FR-009**: JSON metadata SHALL include fields required to generate RSS feed items. All date fields (date) SHALL be stored as UNIX epoch floats (or integers) with **millisecond precision** (float with 3 decimal places). The `duration` field SHALL be stored as a float or integer representing seconds.

**FR-010**: JSON metadata SHALL contain at minimum:
```json
{
  "title": {
    "filename": "lecture",      // sanitized filename with no extension, no underscores
    "extracted": "Lecture 1",   // from source metadata (PDF/HTML/MP4 metadata where available)
    "generated": "LLM Title"   // LLM-generated, prompt includes filename and extracted title before content
  },
  "extracted_title": "Lecture 1",  // extracted title stored separately (optional)
  "description": "string",     // For MP4: blank; For PDF/HTML: extracted metadata
  "summary": "string",         // For MP4: Key takeaways (populated by LLM); For PDF/HTML: null or LLM-generated
  "content": "string",         // For MP4: Full transcript; For PDF/HTML: extracted plaintext
  "filename": "lecture.mp4",
  "filepath": "videos/lecture.mp4",
  "filetype": "pdf|html|mp4",
  "date": 1728931200.123,       // UNIX epoch (float or int, millisecond precision)
  "publication_date": 1728931200.0,  // UNIX epoch (float, millisecond precision, optional - extracted from PDF/HTML metadata)
  "duration": 3600.5,          // For MP4: duration in seconds (float, optional)
  "creation_time": "2024-01-15T10:30:00.000000Z", // ISO-8601 (MP4 only, optional)
  "size": 12345678,
  "hash": "string",            // OPTIONAL: SHA256 hash of source file
  
  // Step maps tracking processing steps
  "transcribe": {
    "status": null | "in_progress" | "complete",
    "start_time": 1792096505.123,  // UNIX epoch (millisecond precision)
    "end_time": 1792096515.456,    // UNIX epoch (millisecond precision)
    "input_size": 12345678,
    "output_size": 456789,
    "words": 452,
    "model": "large-v3-turbo",
    "input_tokens": null,        // optional, for API-based transcription
    "output_tokens": null        // optional, for API-based transcription
  },
  
  "summarize": {
    "status": null | "in_progress" | "complete",
    "start_time": 1792096515.123,
    "end_time": 1792096520.456,
    "input_size": 456789,
    "output_size": 1997,
    "words": 345,
    "model": "mistralai/Mistral-7B-Instruct",
    "input_tokens": 890,
    "output_tokens": 123
  },
  
  "extract": null | {
    "status": null | "in_progress" | "complete",
    "start_time": 1792096500.123,
    "end_time": 1792096505.456,
    "input_size": 1790891,
    "output_size": 12345,
    "words": 543
  },
  
  // Computed field for feed inclusion (not stored in JSON)
  // "feed_eligible": bool,
  
  // NEW: Error field is optional (only present when processing failed)
  "error": "string"  // OPTIONAL: Error message if any step failed
}
```

**Feed Eligibility Rules:**
- **MP4 files**: `feed_eligible = (transcribe.status == "complete" AND summarize.status == "complete")`
- **PDF/HTML files**: `feed_eligible = (extract.status == "complete")`
- Items with `feed_eligible = false` are excluded from feeds
- Items with processing errors will have the respective step map entry as `null` and `feed_eligible = false`

**FR-010a**: The system SHALL use step maps (transcribe, summarize, extract) to track processing steps instead of a unified status map, enabling detailed progress visibility and better error recovery

**FR-010b**: When loading JSON metadata containing deprecated fields (`transcribe_status`, `summarize_status`, `extract_status`, `stage`, `status` dict, `transcript`, `processing_date`, `completed_at`, `started_at`), the system SHALL silently ignore them. Deprecated fields SHALL be ignored (not migrated).

**FR-010c**: The system SHALL replace underscores in extracted titles with spaces (e.g., "my_document" \=> "my document") during metadata finalization. For the filename key in the title dict, the system SHALL sanitize the filename by removing the extension and replacing underscores with spaces.

**FR-010d**: The system SHALL ensure MP4 files have blank `description` field and PDF/HTML files have extracted metadata in `description` field. The `summary` field SHALL contain key takeaways for MP4 files and be null for PDF/HTML files (unless explicitly summarized).

**FR-010e**: The system SHALL extract plaintext content from PDF and HTML files and store it in the `content` field (in addition to metadata in `description`).

**FR-010f**: The system SHALL automatically prepend the parent folder name (module name) to the final title (after precedence resolution) when the final title doesn't already start with the folder name (case-insensitive, underscores normalized to spaces).

**FR-010g**: The system SHALL check for `reflection.md` files in the source file's directory and use their content as reflection questions for summarization when available. `reflection.md` is NOT a prerequisite for summarization - files can be summarized without reflection questions.

**FR-010h**: The system SHALL remove control characters (0x00-0x1f except tab/newline/CR, and 0x7f) from extracted PDF/HTML content to prevent XML validation errors.

**FR-010i**: The system SHALL generate summaries for PDF/HTML files when the `summarize` command is explicitly invoked on those files.

**FR-010j**: When a file is processed for the first time (no existing JSON or incomplete processing), the system SHALL compute the SHA256 hash of the source file and store it in the `hash` field of the JSON metadata.

**FR-010k**: For PDF/HTML files, the system SHALL use publication_date from document metadata for feed pubDate when available, falling back to the date field.

### Audio Processing

**FR-011**: The system SHALL transcribe MP4 audio files to markdown format using a local open-weight model OR a remote OpenAI-compatible API endpoint. When `TRANSCRIPTION_ENDPOINT` is configured, remote mode is automatically enabled as the default transcription method.

**FR-012**: The system SHALL generate key takeaway summaries from audio transcripts using a local OpenAI-compatible LLM

**FR-013**: The system SHALL feed the **entire transcript** to the LLM for summarization

**FR-014**: Audio file summaries SHALL be used as the `summary` field in the JSON metadata (not as description for MP4 files)

**FR-015**: Audio transcripts SHALL NOT include timestamps

**FR-016**: The system SHALL use configurable system prompts for LLM summarization and title generation, stored in environment variables

**FR-016a**: The system SHALL support separate prompts for:
- `SUMMARY_PROMPT`: Regular summarization
- `SUMMARY_REFLECTION_PROMPT`: Summarization with reflection questions (optional, falls back to SUMMARY_PROMPT)
- `TITLE_PROMPT`: Title generation from content

**FR-028**: When reflection questions are available (via `reflection.md`), the system SHALL use `SUMMARY_REFLECTION_PROMPT` for summarization and prepend the reflection questions to the transcript.

**FR-028**: When `TRANSCRIPTION_ENDPOINT` is configured, the system SHALL use the remote OpenAI-compatible API for transcription

**FR-029**: When `TRANSCRIPTION_ENDPOINT` is NOT configured, the system SHALL use local faster-whisper for transcription

**FR-030**: The system SHALL NOT pass a model parameter to the remote transcription endpoint by default (let the endpoint decide)

**FR-031**: If `TRANSCRIPTION_REMOTE_MODEL` is configured, the system SHALL pass it as the model parameter to the remote transcription endpoint

**FR-032**: The system SHALL handle remote API errors appropriately and record them in the JSON metadata error field. This includes errors such as "Audio exceeds maximum allowed duration" which may occur if the remote vLLM server does not have `VLLM_MAX_AUDIO_DECODE_DURATION_S` configured.

**FR-033**: The system SHALL extract audio metadata (duration, creation_time) using ffprobe when available, falling back to file metadata when ffprobe is not installed.

### Feed Generation

**FR-017**: The system SHALL generate two distinct RSS feeds based on feed.json configuration:
1. `podcast.xml` - Contains MP4 audio files
2. `articles.xml` - Contains PDF/HTML files

**FR-018**: Podcast feed SHALL follow podcast RSS specifications (enclosure tags for audio files)

**FR-019**: Articles feed SHALL follow standard RSS specifications

**FR-020**: Each feed item SHALL include:
- Title (from JSON metadata title dict's final title computation)
- Description (from JSON metadata: for MP4 use summary field, for PDF/HTML use summary field or description)
- Link to the source file (computed as BASE_URL + filepath)
- Publication date (from JSON metadata: for MP4 use date field, for PDF/HTML use publication_date field when available, falling back to date field)
- Unique identifier (guid): When the `hash` field is present in JSON metadata, use the last 12 characters of the hash value. Otherwise, use the file path as fallback.

**FR-021**: Feed items SHALL be sorted by folder name, then filename (alphabetical)

**FR-022**: Feed items SHALL only include entries where `feed_eligible = true`

**FR-023**: The system SHALL convert markdown in descriptions to HTML for feed output, with proper XSS protection (safe URL schemes only).

**FR-024**: Per-folder feed generation: When a folder contains a `feed.json` file, the system SHALL generate `podcast.xml` and `articles.xml` in that folder containing only items from that folder's subtree, using the title and description from `feed.json`. The `feed.json` SHALL support standard RSS metadata fields including title (required), description (optional), language (optional), author (optional), link (optional), and image (optional).

**FR-025**: Root-level feeds SHALL **NOT** be generated unless a `feed.json` file exists in the root of the `data/` directory. Feeds require explicit configuration via feed.json.

**FR-025a**: **Conditional feed generation**: The system SHALL only generate `podcast.xml` if both: (1) a `feed.json` file exists in that folder, AND (2) at least one MP4 file with `feed_eligible=true` exists in that folder's subtree.

**FR-025b**: **Conditional feed generation**: The system SHALL only generate `articles.xml` if both: (1) a `feed.json` file exists in that folder, AND (2) at least one PDF/HTML file with `feed_eligible=true` exists in that folder's subtree.

**FR-025c**: The system SHALL generate a **root-level** `index.html` file listing all feeds available from all folders containing `feed.json` with their titles, descriptions, and URLs.

**FR-025d**: The system SHALL include links in **root-level** `index.html` only to feeds that exist (based on conditional generation rules above).

### Configuration

**FR-026**: The system SHALL use environment variables for configuration including:
- Required LLM settings
- Optional runtime parameters
- Endpoint configurations
- Optional remote transcription settings

**FR-027**: The system SHALL require the following environment variables to be provided:
- `LLM_MODEL`: Model identifier for LLM summarization (REQUIRED - no default)
- `LLM_ENDPOINT`: Base URL for LLM API endpoint
- `LLM_API_KEY`: API key for LLM authentication
- `BASE_URL`: Root URL for feed enclosure links
- `DATA_DIR`: Path to data directory
- `SUMMARY_PROMPT`: System prompt for regular summarization (default: "Summarize key takeaways from the given content. Output just the summary concisely.")
- `TITLE_PROMPT`: System prompt for title generation (default: "Generate a concise, descriptive title (5-10 words) for the given content. Do not include any formatting, quotes, or extra text.")

**FR-028**: The system SHALL support the following optional environment variables for remote transcription:
- `TRANSCRIPTION_ENDPOINT`: OpenAI-compatible API endpoint for transcription (optional)
- `TRANSCRIPTION_API_KEY`: API key for transcription endpoint authentication (optional)
- `TRANSCRIPTION_REMOTE_MODEL`: Model override for remote transcription (optional)

**FR-033**: The system SHALL support the following optional environment variables for enhanced features:
- `SUMMARY_REFLECTION_PROMPT`: System prompt for summarization with reflection questions (optional, falls back to SUMMARY_PROMPT)

**FR-035**: When using remote transcription with a vLLM server, the remote server SHALL have the `VLLM_MAX_AUDIO_DECODE_DURATION_S` environment variable configured to handle audio files of the expected duration. This is a server-side requirement, not a client-side configuration. Without this configuration, the vLLM server may reject longer audio files with "Audio exceeds maximum allowed duration" errors.

### CLI Interface

**FR-029**: The system SHALL provide CLI commands for all processing operations:
- `process`: Process all files in data directory
- `watch`: Watch data directory for changes
- `transcribe`: Transcribe audio files
- `summarize`: Generate summaries from transcripts or extracted content
- `extract`: Extract metadata from PDF/HTML files
- `feeds`: Generate RSS feeds and finalize JSON metadata by computing feed eligibility, and generate root-level index.html
- `validate`: Validate all metadata files and finalize JSON by setting step completion statuses
- `create-feed`: Create a template `feed.json` file in the specified directory

**FR-030**: All CLI subcommands SHALL accept the following flags:
- `--data-dir <path>`: Path to the data directory. Defaults to `DATA_DIR` from configuration.
- `--config-file <path>`: Path to a `.env`-style configuration file.
- `--file <path>`: Process a single specific file instead of scanning the entire directory. Takes precedence over `--data-dir`. Enables targeted processing for debugging, retries, and manual intervention.
- `--force`: Force execution even if step appears complete. Overrides step completion checks to enable reprocessing. Applies to all processing steps.

**FR-030a**: The `--dry-run` flag SHALL be available on `transcribe`, `summarize`, and `extract` commands. It SHALL show which files would be processed without making any changes.

**FR-031**: The `--force` flag behavior SHALL be:
- For `transcribe`: Re-transcribe even if `transcribe.status == "complete"`
- For `summarize`: Re-summarize even if `summarize.status == "complete"`
- For `extract`: Re-extract even if `extract.status == "complete"`
- For `process`: Re-process all steps even if `feed_eligible == true`
- For `feeds`: Always regenerate feeds and index.html regardless of state
- For `validate`: Perform validation regardless of previous validation state

**FR-031a**: The `summarize` command SHALL process both MP4 files (with transcripts) and PDF/HTML files (with extracted content).

---

## Non-Functional Requirements

### Local Processing

**NFR-001**: All transcription and summarization SHALL be performed locally by default

**NFR-002**: The system SHALL NOT use any cloud-based services by default

**NFR-003**: The system SHALL leverage Nvidia GPUs for efficient model inference when available

**NFR-004**: The system SHALL use open-weight models for transcription

**NFR-005**: The system SHALL use OpenAI-compatible LLMs for summarization

### Content Quality

**NFR-006**: Summaries SHALL be concise key takeaways

**NFR-007**: Transcripts SHALL be in human-readable markdown format

**NFR-008**: Extracted content from PDF/HTML SHALL have control characters removed to prevent XML validation errors

**NFR-009**: Feed descriptions SHALL be converted from markdown to HTML for proper rendering in feed readers

### Processing Behavior

**NFR-010**: In incremental mode, the system SHALL detect and process new files within a reasonable timeframe

**NFR-011**: The system SHALL handle file system events gracefully (new, modified, deleted files)

**NFR-012**: The system SHALL support targeted reprocessing of individual files via the `--file` flag

**NFR-013**: The system SHALL respect the `--force` flag to bypass completion checks when debugging or testing different parameters

**NFR-014**: The system SHALL implement debouncing in watch mode to prevent rapid successive events from triggering multiple processes for the same file

**NFR-015**: The system SHALL implement periodic rescan as a fallback for file systems that may not reliably trigger watchdog events

---

## Constraints

**C-001**: By default, no cloud-based services are used; transcription and summarization are performed locally. However, remote transcription via OpenAI-compatible API endpoint is supported as an alternative when configured.

**C-002**: By default, open-weight models must be used for all AI processing. When using remote transcription via `TRANSCRIPTION_ENDPOINT`, the remote endpoint may use any model, but local processing still requires open-weight models.

**C-003**: Output feeds must be placed in `data` directory root (and optionally in subfolders with feed.json). **Feeds are only generated where feed.json exists and applicable files are present.** Each feed aggregates content from its entire directory subtree, so per-folder feeds are subsets of any parent folder feeds.

**C-004**: Input files and their JSON metadata must maintain existing directory structure

**C-005**: Existing web server will serve the `data` directory - no deployment needed

**C-006**: Special characters in filenames must be preserved (warnings only, no modifications)

**C-007**: All endpoint examples and placeholders MUST use the `example.com` domain as per RFC 2606

**C-008**: The title field in JSON metadata SHALL be a dict with keys: filename (sanitized filename with no extension, no underscores), extracted (from metadata), generated (LLM-generated with filename and extracted title in prompt)

**C-008a**: Title precedence for final title SHALL be: filename < extracted < generated (generated takes highest precedence, filename lowest)

**C-009**: Dates SHALL be stored as UNIX epoch floats with millisecond precision

**C-010**: Control characters that are invalid in XML SHALL be removed from extracted content

**C-011**: Per-folder feeds require a feed.json file in the folder

**C-012**: **Root feeds require a feed.json file in the root data directory**

**C-013**: **Feeds are only generated if applicable files exist**: podcast.xml only if MP4 files present, articles.xml only if PDF/HTML files present

**C-014**: **`index.html` is maintained at root level only** and is NOT generated in subfolders. It lists all feeds from all folders.

---

## Assumptions

**A-001**: The `data` directory exists and is accessible for read/write operations

**A-002**: The existing web server is configured to serve files from the `data` directory

**A-003**: Open-weight transcription models and OpenAI-compatible LLMs are available and configured locally

**A-004**: Nvidia GPUs are available for the system to use

**A-005**: Source files have sufficient quality for accurate transcription

**A-006**: The architect will select:
- Specific open-weight transcription model
- Specific OpenAI-compatible LLM for summarization

---

## Open Questions

**None** - All open questions have been resolved.

---

## Risks

**R-001**: Transcription quality may vary with open-weight models
- **Mitigation**: Use well-established models; accept basic quality as sufficient for personal use

**R-002**: LLM summarization may produce inconsistent results
- **Mitigation**: Use a tested system prompt; allow config file updates for improvement

**R-003**: File watching in incremental mode may miss events on some systems
- **Mitigation**: Implement periodic directory scanning as fallback

**R-004**: Special characters in filenames may cause parsing issues in RSS feeds
- **Mitigation**: Properly escape all output; warn user during processing

**R-005**: Large audio files may take significant time to transcribe
- **Mitigation**: No time constraints; acceptable for personal use

**R-006**: Control characters in PDF/HTML content may cause XML generation to fail
- **Mitigation**: Automatically remove control characters during extraction

---

## Configuration

### Environment Variables

The system uses the following environment variables for configuration:

#### Required Variables

These variables **MUST** be provided via `.env` file or environment variables. The system will fail to start if they are missing.

- **`BASE_URL`**: Root URL for feed enclosure links. Uses RFC 2606 compliant example domain. (Example: `https://example.com/feeds`)
- **`DATA_DIR`**: Path to the data directory containing source files. (Example: `./data`)
- **`LLM_ENDPOINT`**: Base URL for the OpenAI-compatible LLM API. Uses RFC 2606 compliant example domain. (Example: `https://llm.example.com/v1`)
- **`LLM_API_KEY`**: API key for authenticating with the LLM endpoint. (Example: `your-api-key-here`)
- **`LLM_MODEL`**: Model identifier for LLM summarization. Required because summarization requires a specific model; cannot default to None. (Example: `mistralai/Mistral-7B-Instruct`)
- **`SUMMARY_PROMPT`**: System prompt for regular LLM summarization. Users need explicit control as no sensible universal default exists. (Example: `"Summarize key takeaways from the given content. Output just the summary concisely."`)
- **`TITLE_PROMPT`**: System prompt for LLM title generation. Required for generating titles from content. (Example: `"Generate a concise, descriptive title (5-10 words) for the given content. Do not include any formatting, quotes, or extra text."`)

#### Optional Variables

These variables have sensible defaults and may be omitted.

- **`SUMMARY_REFLECTION_PROMPT`**: System prompt for summarization with reflection questions. Optional; if not provided, falls back to SUMMARY_PROMPT. (Example: `"Summarize key takeaways focusing on: frameworks, analytical tools, critical thinking. Output just the summary concisely."`)
- **`TRANSCRIPTION_MODEL`**: Transcription model to use (Default: `large-v3-turbo`)
- **`TRANSCRIPTION_DEVICE`**: Device for transcription (`cpu` or `cuda`) (Default: `cuda`)
- **`TRANSCRIPTION_COMPUTE_TYPE`**: Compute type for transcription (Default: `float16`)
- **`TRANSCRIPTION_ENDPOINT`**: OpenAI-compatible API endpoint for remote transcription. When configured, enables remote transcription mode instead of local faster-whisper. Uses RFC 2606 compliant example domain. (Example: `https://transcription.example.com/v1`)
- **`TRANSCRIPTION_API_KEY`**: API key for authenticating with the remote transcription endpoint. Required if `TRANSCRIPTION_ENDPOINT` is configured. (Example: `your-transcription-api-key-here`)
- **`TRANSCRIPTION_REMOTE_MODEL`**: Model identifier override for remote transcription. Optional; if not provided, the remote endpoint decides the model. (Example: `whisper-1`)

### Example `.env` File

```bash
# =============================================================================
# REQUIRED VARIABLES - MUST be configured
# =============================================================================

# Feed Configuration
BASE_URL=https://example.com/feeds
DATA_DIR=./data

# LLM Configuration (REQUIRED - no defaults provided)
LLM_ENDPOINT=https://llm.example.com/v1
LLM_API_KEY=your-llm-api-key-here
LLM_MODEL=mistralai/Mistral-7B-Instruct

# NEW: Required prompt configuration
SUMMARY_PROMPT="Summarize key takeaways from the given content. Output just the summary concisely."
TITLE_PROMPT="Generate a concise, descriptive title (5-10 words) for the given content. Do not include any formatting, quotes, or extra text."

# Optional: Reflection prompt for summarization with reflection questions
SUMMARY_REFLECTION_PROMPT="Summarize key takeaways focusing on: frameworks, analytical tools, critical thinking, creative thinking, systems thinking. Output just the summary concisely."

# =============================================================================
# OPTIONAL VARIABLES - Defaults shown
# =============================================================================

# Transcription Configuration
TRANSCRIPTION_MODEL=large-v3-turbo
TRANSCRIPTION_DEVICE=cuda
TRANSCRIPTION_COMPUTE_TYPE=float16

# Remote Transcription Configuration (Optional - enables remote API mode when configured)
# When TRANSCRIPTION_ENDPOINT is set, remote mode is used instead of local faster-whisper
# TRANSCRIPTION_ENDPOINT=https://transcription.example.com/v1
# TRANSCRIPTION_API_KEY=your-transcription-api-key-here
# TRANSCRIPTION_REMOTE_MODEL=whisper-1
```

---

## CLI Commands and Flags

### Command Overview

Note: Transcription commands use local faster-whisper by default, or a remote OpenAI-compatible API endpoint when `TRANSCRIPTION_ENDPOINT` is configured.

```bash
# Process all files in data/ directory (batch mode)
local-feed-generator process --data-dir ./data

# Process a specific file (takes precedence over --data-dir)
local-feed-generator process --file ./data/videos/lecture.mp4

# Process a specific file with force flag (reprocess even if complete)
local-feed-generator process --file ./data/videos/lecture.mp4 --force

# Watch data/ directory for changes (incremental mode)
local-feed-generator watch --data-dir ./data

# Watch a specific file
local-feed-generator watch --file ./data/videos/lecture.mp4

# Transcribe all audio files
local-feed-generator transcribe --data-dir ./data

# Transcribe a specific file
local-feed-generator transcribe --file ./data/videos/lecture.mp4

# Force re-transcription
local-feed-generator transcribe --file ./data/videos/lecture.mp4 --force

# Summarize all transcripts and extracted content
local-feed-generator summarize --data-dir ./data

# Summarize a specific file (MP4, PDF, or HTML)
local-feed-generator summarize --file ./data/videos/lecture.mp4 --force

# Summarize a PDF/HTML file
local-feed-generator summarize --file ./data/documents/report.pdf

# Extract metadata and content from PDF/HTML files
local-feed-generator extract --data-dir ./data

# Extract metadata from a specific file
local-feed-generator extract --file ./data/documents/report.pdf --force

# Generate feeds only (no processing); finalizes JSON metadata; generates index.html
local-feed-generator feeds --data-dir ./data

# Create feed.json template in a directory
local-feed-generator create-feed --dir ./data/my-course

# Validate all metadata files and finalize JSON
local-feed-generator validate --data-dir ./data
```

### Flag Behavior

| Flag | Applies To | Description | Priority |
|------|------------|-------------|----------|
| `--file <path>` | All commands | Process only the specified file | Highest (overrides --data-dir) |
| `--config-file <path>` | All commands | Path to a .env-style configuration file | Medium |
| `--data-dir <path>` | All commands | Path to the data directory. Defaults to `DATA_DIR` from configuration. | Medium |
| `--force` | process, watch, transcribe, summarize, extract, feeds, validate | Force execution even if step appears complete | Medium |
| `--dry-run` | transcribe, summarize, extract | Show which files would be processed without making changes | Low |

### Precedence Rules

1. `--file` takes precedence over `--data-dir` when both are provided
2. `--force` overrides step completion checks for the specified command
3. For commands processing a single file with `--file`, the `--force` flag applies to that file only

---

## Appendix

### Appendix A: Metadata Schema Evolution

#### Overview

This section documents the evolution of the JSON metadata schema through multiple versions to the current step map-based format.

#### Legacy Schema (Deprecated)

```json
{
  "status": "success" | "error" | "in_progress" | "complete",
  "title": "string",
  "description": "string",
  "filename": "string",
  "filepath": "string",
  "filetype": "pdf|html|mp4",
  "date": "ISO8601",
  "transcript": "string",
  "summary": "string",
  "processing_date": "ISO8601",
  "error": "string (optional)"
}
```

**Note:** This schema is deprecated. Files with this schema will trigger warnings when loaded.

#### Intermediate Schema (Deprecated)

```json
{
  "title": "string",
  "description": "string",
  "filename": "string",
  "filepath": "string",
  "filetype": "pdf|html|mp4",
  "date": "ISO8601",
  "size": number,
  "transcript": "string",
  "summary": "string",
  "processing_date": "ISO8601",
  
  "transcribe_status": null | "in_progress" | "complete",
  "summarize_status": null | "in_progress" | "complete",
  "extract_status": null | "in_progress" | "complete",
  "feed_eligible": bool,
  "error": "string (optional)"
}
```

**Note:** This schema is deprecated. Files with individual status fields will trigger warnings when loaded.

#### Previous Schema with Unified Status Map (Deprecated)

```json
{
  "title": "string",
  "description": "string",
  "filename": "string",
  "filepath": "string",
  "filetype": "pdf|html|mp4",
  "date": 1710000000,                  // UNIX epoch seconds
  "size": number,
  "transcript": "string",
  "summary": "string",
  "processing_date": 1710000000,       // UNIX epoch seconds
  "completed_at": 1710000000,          // UNIX epoch seconds (OPTIONAL)
  "started_at": 1710000000,            // UNIX epoch seconds (OPTIONAL)
  
  "status": {
    "transcribe": null | "in_progress" | "complete",
    "summarize": null | "in_progress" | "complete",
    "extract": null | "in_progress" | "complete"
  },
  
  "feed_eligible": bool,               // computed
  "error": "string (optional)"
}
```

**Note:** This schema is deprecated. Files with unified status map will trigger warnings when loaded.

#### Current Schema (Active)

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
  "publication_date": 1710000000.0,  // Extracted from PDF/HTML metadata (optional)
  "duration": 45.5,
  "creation_time": "2024-01-01T00:00:00.000000Z",
  "size": 12345678,
  "hash": "a1b2c3d4e5f6...",  // OPTIONAL: SHA256 hash of source file
  
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
  
  "extract": null
}
```

**Feed Inclusion Rule:**
- **MP4 files**: `feed_eligible = (transcribe.status == "complete" AND summarize.status == "complete")`
- **PDF/HTML files**: `feed_eligible = (extract.status == "complete")`

**Field Usage by Filetype:**
- **MP4**: `description = ""`, `summary` = LLM-generated key takeaways, `content` = full transcript
- **PDF/HTML**: `title` = dict with sources (filename, extracted, generated with precedence filename < extracted < generated), `extracted_title` = source metadata title, `description` = extracted metadata, `content` = extracted plaintext, `summary` = null (unless summarized), `publication_date` = extracted from document metadata when available
**Key Changes in Current Schema:**

- `title` is now a dict with sources: filename, extracted, generated
- `extracted_title` field added for storing source metadata titles separately
- `content` field now used for both MP4 (transcript) and PDF/HTML (extracted text)
- Timestamps now have millisecond precision (float with 3 decimal places)
- `publication_date` field added for PDF/HTML files (extracted from document metadata)
- `duration` and `creation_time` fields added for MP4 files
- Step maps include `start_time` and `end_time` with millisecond precision
- `hash` field added for storing SHA256 hash of source file (optional)

#### Schema Changes Summary

| Version | Status Fields | Date Format | Title Format | Content Field | Timestamps | Hash Field |
|---------|---------------|-------------|--------------|---------------|------------|------------|
| Legacy | Single `status` string | ISO8601 strings | String | `transcript` | Basic | N/A |
| Intermediate | Individual fields (`transcribe_status`, etc.) | ISO8601 strings | String | `transcript` | Basic | N/A |
| Previous | Unified `status` map | UNIX epoch integers | String | `transcript` | Second precision | N/A |
| **Current** | **Step maps** | **UNIX epoch floats (ms)** | **Dict with sources** | **`content` (all types)** | **Millisecond precision** | **SHA256 hash (optional)** |

#### Migration and Compatibility

- **No backwards migration**: Old fields are ignored, not converted
- **Deprecated fields**: Loading deprecated fields is silently ignored
- **Title cleanup**: Underscores in titles automatically replaced with spaces. Filename key in title dict is sanitized (no extension, no underscores).
- **Field mapping**: `transcript` field maps to `content` field for backwards compatibility (with warning)
- **summarize for PDF/HTML**: Optional; summary field remains null unless explicitly requested

### Appendix B: Environment Variable Standardization

All endpoint examples and placeholders now use the `example.com` domain as per RFC 2606, which reserves the `example.com` domain for documentation and examples.

| Variable | Old Placeholder | New Placeholder | RFC 2606 Compliance |
|----------|-----------------|-----------------|---------------------|
| `LLM_ENDPOINT` | `https://your-llm-endpoint.com/v1` | `https://llm.example.com/v1` | \u2713 Compliant |
| `BASE_URL` | `https://your-domain.com/feeds` | `https://example.com/feeds` | \u2713 Compliant |

This standardization:
- Follows internet standards for documentation
- Makes examples clearly identifiable as placeholders
- Prevents accidental use of real domains in documentation
- Improves consistency across all documentation

### Appendix C: CLI Granular Flags Rationale

The introduction of `--file` and `--force` flags addresses several use cases:

#### Use Case 1: Targeted Processing
```bash
# User wants to reprocess a single file that had an error
local-feed-generator process --file ./data/videos/problematic.mp4
```
Benefit: No need to scan entire directory; faster debugging

#### Use Case 2: Parameter Testing
```bash
# User wants to test different LLM parameters on a single file
local-feed-generator transcribe --file ./data/videos/test.mp4 --force
local-feed-generator summarize --file ./data/videos/test.mp4 --force
```
Benefit: Can iterate on model parameters without reprocessing all files

#### Use Case 3: Manual Intervention
```bash
# User fixed a source file and wants to retry just that one
local-feed-generator process --file ./data/documents/fixed.pdf --force
```
Benefit: Only the corrected file is reprocessed; existing metadata for other files is preserved

#### Use Case 4: Debugging Step-by-Step
```bash
# User wants to check transcription without summarization
local-feed-generator transcribe --file ./data/videos/test.mp4
# Then check the transcript before summarizing
local-feed-generator summarize --file ./data/videos/test.mp4
```
Benefit: Granular control over each processing step

#### Use Case 5: PDF/HTML Summarization
```bash
# User wants to generate summaries for PDF/HTML files
local-feed-generator summarize --file ./data/documents/report.pdf
```
Benefit: Generate summaries for document content, not just audio

#### Flag Interaction Matrix

| Command | `--file` | `--force` | Behavior |
|---------|----------|-----------|----------|
| process | \u2713 | \u2713 | Process single file, bypassing all completion checks |
| process | \u2713 | \u2717 | Process single file, respecting step completion |
| process | \u2717 | \u2713 | Process all files, bypassing all completion checks |
| process | \u2717 | \u2717 | Process all incomplete files |
| transcribe | \u2713 | \u2713 | Re-transcribe single file |
| transcribe | \u2713 | \u2717 | Transcribe single file if not already complete |
| watch | \u2713 | \u2713 | Watch single file, bypassing completion checks |
| watch | \u2713 | \u2717 | Watch single file, respecting completion |

### Appendix D: Feed Eligibility Logic

The `feed_eligible` field is a computed boolean that determines whether an item should appear in the generated feeds. This field is **not** stored in the JSON file but is computed at feed generation time based on the step map fields.

**Computation Algorithm:**

```
def compute_feed_eligible(filetype, transcribe, summarize, extract, error):
    if error is not None:
        return False
    if filetype == "mp4":
        return (transcribe is not None and transcribe.status == "complete" and
                summarize is not None and summarize.status == "complete")
    elif filetype in ["pdf", "html"]:
        return (extract is not None and extract.status == "complete")
    return False
```

**State Matrix for MP4 Files:**

| transcribe.status | summarize.status | feed_eligible | Feed Inclusion |
|-------------------|------------------|---------------|----------------|
| null | null | false | \u274c Excluded |
| in_progress | null | false | \u274c Excluded |
| complete | null | false | \u274c Excluded |
| complete | in_progress | false | \u274c Excluded |
| complete | complete | true | \u2705 Included |

**State Matrix for PDF/HTML Files:**

| extract.status | feed_eligible | Feed Inclusion |
|----------------|---------------|----------------|
| null | false | \u274c Excluded |
| in_progress | false | \u274c Excluded |
| complete | true | \u2705 Included |

**Error Handling:**
- If any step fails (status remains `null` or error is set), `feed_eligible` will be `false`
- Items with errors are **excluded** from feeds to prevent partial content from appearing
- Users must fix the error and reprocess (using `--force` if needed) to make the item eligible

### Appendix E: Per-Folder Feed Generation

Per-folder feed generation allows users to organize content into separate feeds for different courses, modules, or topics.

**Mechanism:**
1. Create a `feed.json` file in any subfolder of `data/`
2. The `feed.json` should contain at least `title` and optionally `description`
3. When feeds are generated, the system will:
   - Generate `podcast.xml` in that folder **only if MP4 files are present**
   - Generate `articles.xml` in that folder **only if PDF/HTML files are present**
   - Generate `index.html` in that folder with links to available feeds
   - Include only items from that folder's subtree
   - Use the custom title and description from `feed.json`

**Example:**
```
data/
\u251c\u2500\u2500 feed.json                    # Root-level feed config (REQUIRED for root feeds)
   \u251c\u2500\u2500 podcast.xml                  # Root-level podcast feed (only if MP4 files present)
   \u251c\u2500\u2500 articles.xml                 # Root-level articles feed (only if PDF/HTML files present)
   \u251c\u2500\u2500 index.html                    # Root-level index page
   \u251c\u2500\u2500 Course1/
\u2502   \u251c\u2500\u2500 feed.json               # Per-folder feed config
\u2502   \u251c\u2500\u2500 podcast.xml             # Course1 podcast feed (only if MP4 files present)
\u2502   \u251c\u2500\u2500 articles.xml            # Course1 articles feed (only if PDF/HTML files present)
\u2502   \u251c\u2500\u2500 lecture1.mp4
\u2502   \u2502\u2500 lecture1.mp4.json
\u2514\u2500 Course2/
     \u251c\u2500\u2500 feed.json
     \u251c\u2500\u2500 podcast.xml              # NOT generated if no MP4 files
     \u251c\u2500\u2500 articles.xml             # Generated if PDF/HTML files present
    \u251c\u2500\u2500 lecture1.mp4
    \u2502\u2500 lecture1.mp4.json
```

**feed.json format:**
```json
{
  "title": "Course 1: Introduction to AI",
  "description": "All materials for Course 1 covering AI fundamentals",
  "language": "en-us",
  "author": "Course Author",
  "link": "https://example.com/course1",
  "image": {
    "url": "https://example.com/course1-logo.png",
    "title": "Course 1 Logo",
    "link": "https://example.com/course1"
  }
}
```

**Required fields:** `title`
**Optional fields:** `description`, `language`, `author`, `link` (optional, defaults to BASE_URL if not provided), `image` (with `url`, `title`, `link` subfields)

**Folder Subtree Aggregation:**
- Each feed includes all content files (MP4, PDF, HTML) from its own folder and all subfolders within it
- This means per-folder feeds are subsets of their parent folder feeds
- Example: A feed.json in `Course1/` will include files from `Course1/`, `Course1/Week1/`, `Course1/Week2/`, etc.

**Benefits:**
- Better organization in feed readers
- Custom branding per course/module
- Conditional feed generation based on file types
- Explicit configuration required (no accidental feeds)
- Maintains predictability (feeds only where configured)
- Nested feed hierarchy allows for flexible organization

### Appendix F: Reflection Questions Support

Reflection questions allow users to focus summarization on specific aspects of the content. **`reflection.md` is optional and helpful but NOT a prerequisite for summarization.**

**Mechanism:**
1. Create a `reflection.md` file in any folder containing MP4 or PDF/HTML files
2. The content of `reflection.md` should be reflection questions
3. When processing files in that folder:
   - The reflection questions are read from `reflection.md`
   - They are prepended to the transcript or extracted content
   - The system uses `SUMMARY_REFLECTION_PROMPT` (or falls back to `SUMMARY_PROMPT`)
   - The LLM focuses its summary on answering the reflection questions
4. **If no `reflection.md` exists**, the system uses `SUMMARY_PROMPT` with just the content

**Example reflection.md:**
```markdown
# Reflection Questions

1. What are the key frameworks or mental models presented?
2. What analytical tools can help develop critical thinking?
3. How does this content relate to systems thinking?
4. What are the practical applications of these concepts?
```

**Key Points:**
- PDF/HTML files CAN be summarized without `reflection.md`
- MP4 files CAN be transcribed and summarized without `reflection.md`
- `reflection.md` is purely optional for all file types
- The `summarize` command works with MP4, PDF, and HTML files regardless of `reflection.md` presence

**Benefits:**
- More focused summaries when reflection questions are present
- Customizable per course/module
- Encourages deeper engagement with content
- Can be used for educational purposes
- Works with both MP4 and PDF/HTML files via the summarize command

### Appendix G: Module Name Prefixing

The system automatically prepends the parent folder name (module name) to titles when appropriate.

**Logic:**
1. Extract the parent folder name from the filepath
2. Normalize both the folder name and filename (replace underscores with spaces, lowercase)
3. Check if the filename (without extension) starts with the normalized folder name
4. If not, prepend the normalized folder name with ": " prefix

**Examples:**
- Folder: `Course1/`, File: `lecture.mp4` \=> Title: `Course1: lecture`
- Folder: `Course1/`, File: `Course1_lecture.mp4` \=> Title: `Course1 lecture` (no prefix, already contains)
- Folder: `AI_Fundamentals/`, File: `intro.mp4` \=> Title: `AI Fundamentals: intro`
- Folder: `math/`, File: `math101.mp4` \=> Title: `math101` (no prefix, starts with "math")

**Benefits:**
- Better organization in feed readers
- Clearer context for each item
- Automatic categorization

### Appendix H: Conditional Feed Generation

Feeds are only generated based on explicit configuration and file availability.

**Requirements:**
1. `feed.json` must exist in the folder
2. At least one file of the appropriate type must exist with `feed_eligible=true`

**Rules:**
- `podcast.xml` is generated ONLY IF: (feed.json exists) AND (at least one MP4 file with `feed_eligible=true` exists in the folder subtree)
- `articles.xml` is generated ONLY IF: (feed.json exists) AND (at least one PDF/HTML file with `feed_eligible=true` exists in the folder subtree)
- **Root-level `index.html`** is generated ONLY IF: **at least one feed.json exists anywhere in the data directory** (regardless of file types). Note: index.html is maintained at the root level only, not in subfolders.

**Example Scenarios:**

| feed.json | MP4 files | PDF/HTML files | podcast.xml | articles.xml | index.html |
|-----------|------------|----------------|-------------|--------------|------------|
| Yes | Yes | Yes | Yes | Yes | Yes |
| Yes | Yes | No | Yes | No | Yes |
| Yes | No | Yes | No | Yes | Yes |
| Yes | No | No | No | No | Yes |
| No | Yes | Yes | No | No | No |
| No | Yes | No | No | No | No |
| No | No | Yes | No | No | No |

**Benefits:**
- Explicit configuration (no accidental feeds)
- Resource efficiency (no empty feeds)
- Clear intent (feeds only where explicitly requested)
- Predictable behavior

### Appendix I: Index.html Generation

The system automatically generates a **single root-level** `index.html` file to provide a discovery mechanism for all available feeds. **No index.html files are generated in subfolders.**

**Generation Rules:**
- Root-level `index.html` is generated **only in the data directory root**
- Lists all feeds from all folders containing `feed.json` (including nested folders)
- Contains links only to feeds that exist (based on conditional generation rules)
- Includes feed titles and descriptions from each `feed.json`
- Lists all available podcast.xml and articles.xml feeds from all folders
- Automatically updated each time feeds are regenerated
- Generated whenever at least one feed.json exists in the data directory tree

**Content:**
```html
<!DOCTYPE html>
<html>
<head><title>Available Feeds</title></head>
<body>
  <h1>Available Feeds</h1>
  <h2>Podcast Feeds</h2>
  <ul>
    <li><a href="course1/podcast.xml">Course 1 Podcast</a></li>
    <li><a href="course2/podcast.xml">Course 2 Podcast</a></li>
  </ul>
  <h2>Articles Feeds</h2>
  <ul>
    <li><a href="course1/articles.xml">Course 1 Articles</a></li>
    <li><a href="course3/articles.xml">Course 3 Articles</a></li>
  </ul>
</body>
</html>
```

**Benefits:**
- Easy discovery of all feeds from a single page
- Lists all available podcast and articles feeds
- Works with any web browser
