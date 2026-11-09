# Focus Feed

**Transform education content into RSS feeds with AI summarization powered by your reflection questions.**

## Installation & First Run

Install the project dependencies, configure the required environment variables, and process your data:

```bash
uv sync
cp example.env .env
# Edit .env and set all 9 required environment variables.
uv run focus-feed process ./data
```

The nine required variables are `BASE_URL`, `LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`, `PROMPT_ARTICLE`, `PROMPT_PODCAST`, `PROMPT_TITLE`, `TRANSCRIPTION_ENDPOINT`, and `TRANSCRIPTION_API_KEY`.

## What It Does

Focus Feed turns your educational content (PDF, DOCX, HTML, and other supported document formats, plus audio files) into private RSS feeds that you can consume in any podcast or RSS reader app.

To guide summaries toward your learning goals, create a `reflection.md` file containing a list of reflection questions in a folder. Focus Feed provides these questions to the AI when summarizing materials in that folder. The resulting summary appears in the feed item description.

```mermaid
flowchart TD
    subgraph Inputs
        A[Media files (MP4/MP3/MKV/...)]
        B[Documents (PDF/HTML/DOCX/...)]
        C[reflection.md]
        D[feed.json]
    end
    
    subgraph Processing
        Scanner[Scanner
Discovers files]
        
        Transcribe[Transcribe
media to markdown transcript
remote API]
        Extract[Extract
documents to plaintext + metadata]
        
        Summarize[Summarize
text + reflection.md to key takeaways]
        
        Title[Generate Title
content to title]
    end
    
    subgraph Outputs
        Meta[filename.json
 Contains: title, summary, content, hash, mtime]
        FeedOut[podcast.xml / articles.xml / index.html]
    end
    
    A --> Scanner
    B --> Scanner
    C --> Summarize
    D --> FeedOut
    
    Scanner -->|media| Transcribe
    Scanner -->|documents| Extract
    
    Transcribe --> Summarize
    Extract --> Summarize
    
    Summarize --> Title
    Transcribe --> Title
    Extract --> Title
    
    Transcribe --> Meta
    Extract --> Meta
    Summarize --> Meta
    Title --> Meta
    
    Meta -->|eligible| FeedOut
```

## Quick Start

```bash
# Process supported files
uv run focus-feed process ./data

# Access your feeds at {BASE_URL}/index.html in your browser
```

## Project Layout

- `src/focus_feed/` - Application package. `producer/` contains base, media, and document strategies; `feeds/` contains feed generation, markdown, and validation; `cli/` contains command helpers, watch, and options. Shared modules include `filetypes.py`, `retry.py`, `utils.py`, and `metadata_io.py`; `services/` contains extractors, the transcriber, summarizer, and audio extraction service.
- `tests/` - Automated tests.
- `data/` - Example input data and generated files; this directory is gitignored.
- `scripts/` - Project scripts (gitignored).
- `artifacts/` - Project proposal, specifications, and design documents.
- `example.env` - Environment-variable template to copy to `.env`.

## Development

```bash
uv sync
uv run pytest
uv run ruff check
uv run ruff format
uv run mypy .
```

## Environment Variables

Copy `example.env` to `.env` and set all nine required variables below. Optional variables can be added as needed.

Required configuration:
- BASE_URL - base URL for your feeds
- LLM_ENDPOINT - OpenAI-compatible API endpoint
- LLM_API_KEY - API key for the LLM service
- LLM_MODEL - model identifier for the LLM service
- PROMPT_ARTICLE - article summarization prompt
- PROMPT_PODCAST - podcast/audio summarization prompt
- PROMPT_TITLE - title generation prompt
- TRANSCRIPTION_ENDPOINT - transcription API endpoint
- TRANSCRIPTION_API_KEY - API key for transcription service

Optional:
- DOCLING_API - endpoint for the Docling API service used for document extraction
- MAX_WORKERS - maximum number of files processed concurrently (default: `4`)
- MAX_SUMMARIZE_CONCURRENCY - maximum number of concurrent summarization calls (default: `4`); equivalent to `--max-summarize-jobs`
- PROMPT_ARTICLE_REFLECTION - reflection-aware article prompt
- PROMPT_PODCAST_REFLECTION - reflection-aware podcast prompt
- TRANSCRIPTION_REMOTE_MODEL - transcription model override

## Processing Modes

### Batch Mode
Process all files at once in a directory.

### Incremental Mode
Watch for new or updated files:
```bash
uv run focus-feed watch ./data
```

### Parallel Processing
Batch and watch mode process independent files concurrently with a bounded in-process thread pool. Steps within a single file stay sequential, so the metadata JSON and step ordering are unchanged. Concurrent summarization is capped at 4 by default; adjust it with `--max-summarize-jobs` / `MAX_SUMMARIZE_CONCURRENCY` or the overall file concurrency with `--max-workers`. The primary use case is a small number of new files per run, so no external task queue is used.

## Granular Processing

Target specific steps or files:

- `transcribe` - Transcribe audio files to markdown (remote-only)
- `summarize` - Generate summaries from transcripts or document content
- `extract` - Extract metadata from PDF/HTML and Docling-supported document formats
- `title` - Generate titles for files that lack proper titles
- `feeds` - Generate RSS feeds and index.html from existing metadata (read-only; skips items without a summarized JSON sidecar)
- `validate` - Validate all metadata files
- `create-feed` - Create a template `feed.json` in a directory
- `cleanup` - Remove unknown keys from JSON metadata files

## Feeds & Discovery

Each folder can contain a `feed.json` configuration with `title`, `description`, `language`, `author`, `link`, and `image` fields. `title` is optional; when omitted, the feed uses the default title. Feeds are generated only for folders containing `feed.json` and when eligible files are present. An `index.html` is generated in every directory containing `feed.json`; the root index lists feeds across the whole data tree. Use `{BASE_URL}/index.html` as the browser entry point.

## Reflection Questions

Place a `reflection.md` file in a folder containing a list of questions. Those questions are passed to the summarizer for files in that folder. When set, `PROMPT_ARTICLE_REFLECTION` overrides `PROMPT_ARTICLE`, and `PROMPT_PODCAST_REFLECTION` overrides `PROMPT_PODCAST`.

## Metadata JSON Schema

Each source file has an adjacent metadata file named `<source_filename>.json` (for example, `lecture.mp4.json`). The source `filepath` is not stored; it is inferred from the JSON file's location.

| Field | Description |
| --- | --- |
| `title` | Normally a dict with `filename`, `extracted`, and `generated`; the final title uses `generated` > `extracted` > `filename` (legacy strings are also accepted). |
| `description`, `summary`, `content` | Extracted description, generated summary, and plaintext source content/transcript. |
| `filename`, `filetype` | Original filename and type (supported media formats, PDF, HTML, and Docling-supported document formats). |
| `date` | UNIX epoch seconds with millisecond precision. |
| `mtime`, `publication_date` | Source modification time and optional publication date from document metadata, as UNIX epoch seconds. |
| `size`, `duration`, `creation_time` | Source size in bytes; optional media duration in seconds and ISO-8601 media creation time. |
| `hash`, `error` | Optional SHA-256 source hash and processing error message. |
| `transcribe`, `summarize`, `extract`, `generate_title` | Optional step maps; see fields below. |

Each step map can contain `status` (`null`, `"in_progress"`, or `"complete"`), `start_time`, `end_time` (UNIX epoch seconds with millisecond precision), `input_size`, `output_size`, `words`, `model`, `input_tokens`, `output_tokens`, and `error`. Null-valued fields are preserved by the canonical writer (`write_metadata_json`), including null values nested in the title dict; `Metadata.to_dict()`/`to_json()` omit None values. `feed_eligible` is computed and is not stored.

Media files are feed-eligible when `transcribe` and `summarize` are complete. PDF, HTML, and Docling-supported document files are eligible when `extract` and `summarize` are complete. Any error makes a file ineligible. The `transcribe`, `summarize`, `extract`, and `title` commands run only their respective processing step (and update its related output fields), so other steps may remain incomplete for later processing. `process` and `watch` run the full pipeline. `validate` finalizes incomplete metadata before validating. `feeds` is read-only: it only uses existing JSON sidecars that contain a non-empty `summary`, notes and skips items with no sidecar or no summary, and never creates or modifies JSON metadata.

```json
{
  "title": {"filename": "lecture", "extracted": "Lecture 1", "generated": "Lecture 1: Introduction"},
  "description": "",
  "summary": "Key takeaways",
  "content": "Transcript or extracted text",
  "filename": "lecture.mp4",
  "filetype": "mp4",
  "date": 1728931200.123,
  "mtime": 1728931200.123,
  "size": 12345678,
  "transcribe": {"status": "complete", "input_size": 12345678, "output_size": 456789},
  "summarize": {"status": "complete", "input_size": 456789, "output_size": 1997}
}
```

### Common Flags

| Flag | Scope | Description |
| --- | --- | --- |
| `--config-file <path>` | Commands that accept configuration | Use a custom configuration file instead of `.env`. |
| `--force` | Processing commands | Force re-processing even if complete or unchanged. |
| `--dry-run` | `transcribe`, `summarize`, `extract`, `title` | Preview without making changes. |
| `--max-workers <n>` | `process`, `watch` | Maximum number of files processed concurrently (default: `4`). |
| `--max-summarize-jobs <n>` / `MAX_SUMMARIZE_CONCURRENCY` | `process`, `watch` | Maximum number of concurrent summarization calls (default: `4`). |
| `--log-level <level>` / `LOG_LEVEL` | Global `focus-feed` command group | Set the logging level (default: `INFO`). |

### Examples

```bash
# Process a single file
uv run focus-feed process ./videos/lecture.mp4

# Force re-transcription
uv run focus-feed transcribe ./videos/lecture.mp4 --force

# Summarize a PDF
uv run focus-feed summarize ./documents/report.pdf

# See what would be transcribed
uv run focus-feed transcribe --dry-run
```
