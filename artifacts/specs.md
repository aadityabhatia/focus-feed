# Education Content Feed System - Specifications

## Problem Statement

Education content exists as disconnected files (PDF, HTML, MP4) in a `data` directory with no unified access method. Users cannot efficiently consume this content across devices through standard applications. No coherent processing pipeline exists to transform these files into consumable formats.

## Scope

This system processes education content files in the `data` directory and generates RSS/Atom feeds placed alongside them for personal consumption via mobile apps on an existing internal web server.

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

**FR-007**: For each input file, the system SHALL generate a JSON metadata file with the same name and `.json` extension appended (e.g., `document.pdf` → `document.pdf.json`)

**FR-008**: JSON metadata files SHALL be placed in the same directory as their source files

**FR-009**: JSON metadata SHALL include fields required to generate RSS/Atom feed items. All date fields (date, creation_time) SHALL be stored as UNIX epoch timestamps (integer seconds since 1970-01-01T00:00:00Z). The `duration` field SHALL be stored as a float or integer representing seconds.

**FR-010**: JSON metadata SHALL contain at minimum:
```json
{
  "version": "2.0",
  "title": "string",           // Human-readable title (derived or from file)
  "description": "string",    // For MP4: blank; For PDF/HTML: extracted metadata
  "summary": "string",        // For MP4: Key takeaways (populated by LLM); For PDF/HTML: null
  "content": "string",        // For MP4: Full transcript; For PDF/HTML: null
  "filename": "string",       // Original filename with extension
  "filepath": "string",       // Relative path from data directory root
  "filetype": "pdf|html|mp4", // Source file type
  "date": 1710000000,          // Publication/creation date (UNIX epoch seconds)
  "duration": 45.5,          // For MP4: duration in seconds (float)
  "creation_time": "2024-01-01T00:00:00Z", // ISO8601 creation timestamp
  "size": 12345678,          // File size in bytes (for podcast enclosure length)
  
  // Step maps tracking processing steps
  "transcribe": {
    "status": null | "in_progress" | "complete",
    "start_time": 1710000000,    // UNIX epoch seconds
    "end_time": 1710000005,      // UNIX epoch seconds
    "input_size": 12345678,     // Input file size in bytes
    "output_size": 456789,       // Output transcript size in bytes
    "words": 452,                // Number of words in transcript
    "model": "large-v3-turbo",  // Model used for transcription
    "input_tokens": null,        // Input tokens (optional, for API-based transcription)
    "output_tokens": null        // Output tokens (optional, for API-based transcription)
  },
  
  "summarize": {
    "status": null | "in_progress" | "complete",
    "start_time": 1710000005,    // UNIX epoch seconds
    "end_time": 1710000010,      // UNIX epoch seconds
    "input_size": 456789,        // Input transcript size in bytes
    "output_size": 1997,         // Output summary size in bytes
    "words": 345,                // Number of words in summary
    "model": "mistralai/Mistral-7B-Instruct", // Model used for summarization
    "input_tokens": 890,         // Input tokens
    "output_tokens": 123         // Output tokens
  },
  
  "extract": null | {
    "status": null | "in_progress" | "complete",
    "start_time": 1710000000,    // UNIX epoch seconds
    "end_time": 1710000005,      // UNIX epoch seconds
    "input_size": 12345678,     // Input file size in bytes
    "output_size": 456789,       // Output metadata size in bytes
    "model": "pypdf"            // Extraction method used
  },
  
  // Computed field for feed inclusion
  "feed_eligible": bool,      // Determines if item appears in feeds
  
  "error": "string"           // OPTIONAL: Error message if any step failed
}
```

**Feed Eligibility Rules:**
- **MP4 files**: `feed_eligible = (transcribe.status == "complete" AND summarize.status == "complete")`
- **PDF/HTML files**: `feed_eligible = (extract.status == "complete")`
- Items with `feed_eligible = false` are excluded from feeds
- Items with processing errors will have the respective step map entry as `null` and `feed_eligible = false`

**FR-010a**: The system SHALL use step maps (transcribe, summarize, extract) to track processing steps instead of a unified status map, enabling detailed progress visibility and better error recovery

**FR-010b**: When loading JSON metadata containing deprecated fields (`transcribe_status`, `summarize_status`, `extract_status`, `stage`, `status` dict, `transcript`, `processing_date`, `completed_at`, `started_at`), the system SHALL print a warning to stderr. Deprecated fields SHALL be ignored (not migrated).

**FR-010c**: The system SHALL replace underscores in extracted titles with spaces (e.g., "my_document" → "my document") during metadata finalization.

**FR-010d**: The system SHALL ensure MP4 files have blank `description` field and PDF/HTML files have extracted metadata in `description` field. The `summary` field SHALL contain key takeaways for MP4 files and be null for PDF/HTML files.

### Audio Processing

**FR-011**: The system SHALL transcribe MP4 audio files to markdown format using a local open-weight model OR a remote OpenAI-compatible API endpoint. When `TRANSCRIPTION_ENDPOINT` is configured, remote mode is automatically enabled as the default transcription method.

**FR-012**: The system SHALL generate key takeaway summaries from audio transcripts using a local OpenAI-compatible LLM

**FR-013**: The system SHALL feed the **entire transcript** to the LLM for summarization

**FR-014**: Audio file summaries SHALL be used as the `summary` field in the JSON metadata (not as description for MP4 files)

**FR-015**: Audio transcripts SHALL NOT include timestamps

**FR-016**: The system SHALL use a configurable system prompt for LLM summarization, stored in environment variables

**FR-028**: When `TRANSCRIPTION_ENDPOINT` is configured, the system SHALL use the remote OpenAI-compatible API for transcription

**FR-029**: When `TRANSCRIPTION_ENDPOINT` is NOT configured, the system SHALL use local faster-whisper for transcription

**FR-030**: The system SHALL NOT pass a model parameter to the remote transcription endpoint by default (let the endpoint decide)

**FR-031**: If `TRANSCRIPTION_REMOTE_MODEL` is configured, the system SHALL pass it as the model parameter to the remote transcription endpoint

**FR-032**: The system SHALL handle remote API errors appropriately and record them in the JSON metadata error field. This includes errors such as "Audio exceeds maximum allowed duration" which may occur if the remote vLLM server does not have `VLLM_MAX_AUDIO_DECODE_DURATION_S` configured.

### Feed Generation

**FR-017**: The system SHALL generate two distinct RSS/Atom feeds placed in the `data` directory root:
1. `podcast.xml` (or `podcast.atom`) - Contains MP4 audio files
2. `articles.xml` (or `articles.atom`) - Contains PDF/HTML files

**FR-018**: Podcast feed SHALL follow podcast RSS specifications (enclosure tags for audio files)

**FR-019**: Articles feed SHALL follow standard RSS or Atom specifications

**FR-020**: Each feed item SHALL include:
- Title (from JSON metadata)
- Description (from JSON metadata: blank for MP4, extracted metadata for PDF/HTML)
- Link to the source file (computed as BASE_URL + filepath)
- Publication date (from JSON metadata)
- Unique identifier (file path)

**FR-021**: Feed items SHALL be sorted by folder name, then filename (alphabetical)

**FR-022**: Feed items SHALL only include entries where `feed_eligible = true`

### Configuration

**FR-023**: The system SHALL use environment variables for configuration including:
- Required LLM settings
- Optional runtime parameters
- Endpoint configurations
- Optional remote transcription settings

**FR-024**: The system SHALL require the following environment variables to be provided:
- `LLM_SYSTEM_PROMPT`: System prompt for LLM summarization (REQUIRED - no default)
- `LLM_MODEL`: Model identifier for LLM summarization (REQUIRED - no default)
- `LLM_ENDPOINT`: Base URL for LLM API endpoint
- `LLM_API_KEY`: API key for LLM authentication
- `BASE_URL`: Root URL for feed enclosure links
- `DATA_DIR`: Path to data directory

**FR-033**: The system SHALL support the following optional environment variables for remote transcription:
- `TRANSCRIPTION_ENDPOINT`: OpenAI-compatible API endpoint for transcription (optional)
- `TRANSCRIPTION_API_KEY`: API key for transcription endpoint authentication (optional)
- `TRANSCRIPTION_REMOTE_MODEL`: Model override for remote transcription (optional)

**FR-034**: When using remote transcription with a vLLM server, the remote server SHALL have the `VLLM_MAX_AUDIO_DECODE_DURATION_S` environment variable configured to handle audio files of the expected duration. This is a server-side requirement, not a client-side configuration. Without this configuration, the vLLM server may reject longer audio files with "Audio exceeds maximum allowed duration" errors.

### CLI Interface

**FR-025**: The system SHALL provide CLI commands for all processing operations:
- `process`: Process all files in data directory
- `watch`: Watch data directory for changes
- `transcribe`: Transcribe audio files
- `summarize`: Generate summaries
- `extract`: Extract metadata from PDF/HTML files
- `feeds`: Generate RSS/Atom feeds and finalize JSON metadata by computing feed eligibility
- `validate`: Validate all metadata files and finalize JSON by setting step completion statuses

**FR-026**: All CLI subcommands SHALL accept the following flags:
- `--data-dir <path>`: Path to the data directory. Defaults to `DATA_DIR` from configuration.
- `--config-file <path>`: Path to a `.env`-style configuration file.
- `--file <path>`: Process a single specific file instead of scanning the entire directory. Takes precedence over `--data-dir`. Enables targeted processing for debugging, retries, and manual intervention.
- `--force`: Force execution even if step appears complete. Overrides step completion checks to enable reprocessing. Applies to all processing steps.

**FR-026a**: The `--dry-run` flag SHALL be available on `transcribe`, `summarize`, and `extract` commands. It SHALL show which files would be processed without making any changes.

**FR-027**: The `--force` flag behavior SHALL be:
- For `transcribe`: Re-transcribe even if `transcribe.status == "complete"`
- For `summarize`: Re-summarize even if `summarize.status == "complete"`
- For `extract`: Re-extract even if `extract.status == "complete"`
- For `process`: Re-process all steps even if `feed_eligible == true`
- For `feeds`: Always regenerate feeds regardless of state
- For `validate`: Perform validation regardless of previous validation state

---

## Non-Functional Requirements

### Local Processing

**NFR-001**: All transcription and summarization SHALL be performed locally

**NFR-002**: The system SHALL NOT use any cloud-based services

**NFR-003**: The system SHALL leverage Nvidia GPUs for efficient model inference when available

**NFR-004**: The system SHALL use open-weight models for transcription

**NFR-005**: The system SHALL use OpenAI-compatible LLMs for summarization

### Content Quality

**NFR-006**: Summaries SHALL be concise key takeaways

**NFR-007**: Transcripts SHALL be in human-readable markdown format

### Processing Behavior

**NFR-008**: In incremental mode, the system SHALL detect and process new files within a reasonable timeframe

**NFR-009**: The system SHALL handle file system events gracefully (new, modified, deleted files)

**NFR-010**: The system SHALL support targeted reprocessing of individual files via the `--file` flag

**NFR-011**: The system SHALL respect the `--force` flag to bypass completion checks when debugging or testing different parameters

---

## Constraints

**C-001**: By default, no cloud-based services are used; transcription and summarization are performed locally. However, remote transcription via OpenAI-compatible API endpoint is supported as an alternative when configured.

**C-002**: By default, open-weight models must be used for all AI processing. When using remote transcription via `TRANSCRIPTION_ENDPOINT`, the remote endpoint may use any model, but local processing still requires open-weight models.

**C-003**: Output feeds must be placed in `data` directory root

**C-004**: Input files and their JSON metadata must maintain existing directory structure

**C-005**: Existing web server will serve the `data` directory - no deployment needed

**C-006**: Special characters in filenames must be preserved (warnings only, no modifications)

**C-007**: All endpoint examples and placeholders MUST use the `example.com` domain as per RFC 2606

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
  - Feed format (RSS, Atom, or both)

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

---

## Configuration

### Environment Variables

The system uses the following environment variables for configuration:

#### Required Variables

These variables **MUST** be provided via `.env` file or environment variables. The system will fail to start if they are missing.

- **`LLM_SYSTEM_PROMPT`**: System prompt for LLM summarization. Users need explicit control as no sensible universal default exists. (Example: `"Summarize key takeaways from this document..."`)
- **`LLM_MODEL`**: Model identifier for LLM summarization. Required because summarization requires a specific model; cannot default to None. (Example: `mistralai/Mistral-7B-Instruct`)
- **`LLM_ENDPOINT`**: Base URL for the OpenAI-compatible LLM API. Uses RFC 2606 compliant example domain. (Example: `https://llm.example.com/v1`)
- **`LLM_API_KEY`**: API key for authenticating with the LLM endpoint. (Example: `your-api-key-here`)
- **`BASE_URL`**: Root URL for feed enclosure links. Uses RFC 2606 compliant example domain. (Example: `https://example.com/feeds`)
- **`DATA_DIR`**: Path to the data directory containing source files. (Example: `./data`)

#### Optional Variables

These variables have sensible defaults and may be omitted.

- **`FEED_FORMAT`**: Feed output format: `rss` or `atom` (Default: `rss`)
- **`TRANSCRIPTION_MODEL`**: Transcription model to use (Default: `large-v3-turbo`)
- **`TRANSCRIPTION_DEVICE`**: Device for transcription (`cpu` or `cuda`) (Default: `cuda`)
- **`TRANSCRIPTION_COMPUTE_TYPE`**: Compute type for transcription (Default: `float16`)
- **`TRANSCRIPTION_ENDPOINT`**: OpenAI-compatible API endpoint for remote transcription. When configured, enables remote transcription mode. Uses RFC 2606 compliant example domain. (Example: `https://transcription.example.com/v1`)
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
LLM_SYSTEM_PROMPT="Summarize key takeaways from this document. Focus on frameworks or analytical tools to develop critical, creative, and systems thinking."

# =============================================================================
# OPTIONAL VARIABLES - Defaults shown
# =============================================================================

# Feed Configuration
FEED_FORMAT=rss                           # rss or atom (default: rss)

# Transcription Configuration
TRANSCRIPTION_MODEL=large-v3-turbo
TRANSCRIPTION_DEVICE=cuda
TRANSCRIPTION_COMPUTE_TYPE=float16

# Remote Transcription Configuration (Optional - enables remote API mode when configured)
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

# Summarize all transcripts
local-feed-generator summarize --data-dir ./data

# Summarize a specific file
local-feed-generator summarize --file ./data/videos/lecture.mp4 --force

# Extract metadata from PDF/HTML files
local-feed-generator extract --data-dir ./data

# Extract metadata from a specific file
local-feed-generator extract --file ./data/documents/report.pdf --force

# Generate feeds only (no processing); finalizes JSON metadata
local-feed-generator feeds --data-dir ./data

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

#### Current Schema (Active - Version 2.0)

```json
{
  "version": "2.0",
  "title": "My Audio File",
  "description": "",
  "summary": "Key takeaways from LLM...", 
  "content": "Full transcript...",
  "filename": "lecture.mp4",
  "filepath": "c100/lecture.mp4",
  "filetype": "mp4",
  "date": 1710000000,
  "duration": 45.5,
  "creation_time": "2024-01-01T00:00:00Z",
  "size": 12345678,
  "transcribe": {
    "status": "complete",
    "start_time": 1710000000,
    "end_time": 1710000005,
    "input_size": 12345678,
    "output_size": 456789,
    "words": 452,
    "model": "large-v3-turbo",
    "input_tokens": null,
    "output_tokens": null
  },
  "summarize": {
    "status": "complete",
    "start_time": 1710000005,
    "end_time": 1710000010,
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
- **PDF/HTML**: `description` = extracted metadata, `summary` = null, `content` = null

#### Schema Changes Summary

| Version | Status Fields | Date Format | Fields | Title Cleanup |
|---------|---------------|-------------|--------|---------------|
| Legacy | Single `status` string | ISO8601 strings | Basic | No |
| Intermediate | Individual fields (`transcribe_status`, etc.) | ISO8601 strings | Basic | No |
| Previous | Unified `status` map | UNIX epoch integers | Basic + timestamps | Yes |
| **Current** | **Step maps (transcribe, summarize, extract objects)** | **UNIX epoch + ISO8601 creation_time** | **version, duration, creation_time, content** | **Yes** |

#### Migration and Compatibility

- **No backwards migration**: Old fields are ignored, not converted
- **Warning system**: Loading deprecated fields triggers stderr warnings
- **Title cleanup**: Underscores in titles automatically replaced with spaces
- **Field mapping**: `transcript` field maps to `content` field for backwards compatibility (with warning)

### Appendix B: Environment Variable Standardization

All endpoint examples and placeholders now use the `example.com` domain as per RFC 2606, which reserves the `example.com` domain for documentation and examples.

| Variable | Old Placeholder | New Placeholder | RFC 2606 Compliance |
|----------|-----------------|-----------------|---------------------|
| `LLM_ENDPOINT` | `https://your-llm-endpoint.com/v1` | `https://llm.example.com/v1` | ✓ Compliant |
| `BASE_URL` | `https://your-domain.com/feeds` | `https://example.com/feeds` | ✓ Compliant |

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

#### Flag Interaction Matrix

| Command | `--file` | `--force` | Behavior |
|---------|----------|-----------|----------|
| process | ✓ | ✓ | Process single file, bypassing all completion checks |
| process | ✓ | ✗ | Process single file, respecting step completion |
| process | ✗ | ✓ | Process all files, bypassing all completion checks |
| process | ✗ | ✗ | Process all incomplete files |
| transcribe | ✓ | ✓ | Re-transcribe single file |
| transcribe | ✓ | ✗ | Transcribe single file if not already complete |
| watch | ✓ | ✓ | Watch single file, bypassing completion checks |
| watch | ✓ | ✗ | Watch single file, respecting completion |

### Appendix D: Feed Eligibility Logic

The `feed_eligible` field is a computed boolean that determines whether an item should appear in the generated feeds. This field is **not** stored in the JSON file but is computed at feed generation time based on the step map fields.

**Computation Algorithm:**

```
def compute_feed_eligible(filetype, transcribe, summarize, extract):
    if filetype == "mp4":
        return (transcribe.status == "complete" and summarize.status == "complete")
    elif filetype in ["pdf", "html"]:
        return (extract.status == "complete")
    return False
```

**State Matrix for MP4 Files:**

| transcribe.status | summarize.status | feed_eligible | Feed Inclusion |
|-------------------|------------------|---------------|----------------|
| null | null | false | ❌ Excluded |
| in_progress | null | false | ❌ Excluded |
| complete | null | false | ❌ Excluded |
| complete | in_progress | false | ❌ Excluded |
| complete | complete | true | ✅ Included |
| complete | complete | true | ✅ Included |

**State Matrix for PDF/HTML Files:**

| extract.status | feed_eligible | Feed Inclusion |
|----------------|---------------|----------------|
| null | false | ❌ Excluded |
| in_progress | false | ❌ Excluded |
| complete | true | ✅ Included |

  **Error Handling:**
- If any step fails (status remains `null` or error is set), `feed_eligible` will be `false`
- Items with errors are **excluded** from feeds to prevent partial content from appearing
- Users must fix the error and reprocess (using `--force` if needed) to make the item eligible
