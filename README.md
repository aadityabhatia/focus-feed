# Focus Feed

**Transform your education content into mobile-friendly RSS feeds for podcast and article consumption.**

## Project Summary

Turn disorganized PDF, HTML, and MP4 audio files into RSS feeds for mobile apps.

- Podcast feed - For MP4 audio (Antennapod and other podcast apps)
- Articles feed - For PDF/HTML documents (any RSS reader)
- Per-folder feeds - Organize with `feed.json` in subfolders
- Discovery - Auto-generated `index.html` lists all available feeds

All processing uses your configured endpoints. Transcription is remote-only; audio metadata extracted via PyAV.

**Documentation:**
- [Proposal](./artifacts/proposal.md) - Problem and scope
- [Specifications](./artifacts/specs.md) - Detailed requirements
- [Design Document](./artifacts/design.md) - Architecture

---

## Quick Start

1. **Configure** via environment variables or `.env` file

2. **Process your content** (targeting specific files or directories):
   ```bash
   uv run focus-feed process ./videos/lecture.mp4
   uv run focus-feed process ./documents/ ./videos/
   ```

3. **Access your feeds** at `{BASE_URL}/index.html`

---
To process all files, pass your data directory:
   ```bash
   uv run focus-feed process ./data
   ```

---

## Configuration

### Required Environment Variables

- `BASE_URL` - Root URL where data directory is served (e.g., `https://example.com`)
- `LLM_ENDPOINT` - OpenAI-compatible API endpoint for summarization (can be remote or local)
- `LLM_API_KEY` - API key for LLM endpoint
- `LLM_MODEL` - Model identifier for summarization
- `PROMPT_ARTICLE` - Article summarization prompt (PDF/HTML)
- `PROMPT_PODCAST` - Podcast summarization prompt (MP4)
- `PROMPT_TITLE` - Title generation prompt
- `TRANSCRIPTION_ENDPOINT` - Remote transcription API endpoint (required, no local transcription)
- `TRANSCRIPTION_API_KEY` - Remote transcription API key

### Optional Environment Variables

- `PROMPT_ARTICLE_REFLECTION` - Article summarization with reflection questions (default: `PROMPT_ARTICLE`)
- `PROMPT_PODCAST_REFLECTION` - Podcast summarization with reflection questions (default: `PROMPT_PODCAST`)
- `TRANSCRIPTION_REMOTE_MODEL` - Remote transcription model override

### Example `.env` File

```bash
# Required
BASE_URL=https://example.com
LLM_ENDPOINT=https://llm.example.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=mistralai/Mistral-7B-Instruct
TRANSCRIPTION_ENDPOINT=https://transcription.example.com/v1
TRANSCRIPTION_API_KEY=your-transcription-api-key

# Prompts
PROMPT_ARTICLE="Summarize key takeaways from this document concisely in a few paragraphs."
PROMPT_PODCAST="Summarize key takeaways from this podcast transcript concisely in a few paragraphs."
PROMPT_TITLE="Generate a concise, descriptive title (5-10 words) for the given content."

# Optional
PROMPT_ARTICLE_REFLECTION="Summarize with reflection questions: ..."
PROMPT_PODCAST_REFLECTION="Summarize with reflection questions: ..."
TRANSCRIPTION_REMOTE_MODEL=whisper-1
```

**Note:** Transcription is remote-only. `TRANSCRIPTION_ENDPOINT` and `TRANSCRIPTION_API_KEY` are required.

---

## Processing Modes

### Batch Mode

Process all files at once:
```bash
uv run focus-feed process ./data
```

### Incremental Mode

Watch for new/updated files:
```bash
uv run focus-feed watch ./data
```

File modification times are stored in JSON and checked on each process. Files are skipped if already complete and unchanged. Use `--force` to override.

---

## Granular Processing

Target specific steps or files by passing files and directories as positional arguments:

- `transcribe` - Transcribe MP4 files to markdown (remote-only)
- `summarize` - Generate summaries from transcripts or document content
- `extract` - Extract metadata from PDF/HTML files
- `title` - Generate titles for files that lack proper titles
- `feeds` - Generate RSS feeds and index.html
- `validate` - Validate all metadata files
- `create-feed` - Create a template `feed.json` in a directory
- `cleanup` - Remove unknown keys from JSON metadata files

### Common Flags

- `--force` - Force re-processing even if complete or unchanged
- `--dry-run` - Preview without making changes (transcribe, summarize, extract)
- `--config-file <path>` - Use custom `.env` file

### Examples

```bash
# Process a single file
uv run focus-feed process ./videos/lecture.mp4

# Force re-transcription of a specific file
uv run focus-feed transcribe ./videos/lecture.mp4 --force

# Summarize a PDF
uv run focus-feed summarize ./documents/report.pdf

# Process all files in a directory
uv run focus-feed process ./documents/

# Process multiple files and directories at once
uv run focus-feed process ./videos/lecture.mp4 ./documents/report.pdf ./course1/

# See what would be transcribed
uv run focus-feed transcribe --dry-run
```

---

## File Organization

### Directory Structure

```
data/
├── index.html                 # Auto-generated: lists all feeds
├── feed.json                  # REQUIRED for root-level feeds
├── podcast.xml                # Generated if MP4 files present
├── articles.xml               # Generated if PDF/HTML files present
└── Course1/
    ├── feed.json              # REQUIRED for per-folder feeds
    ├── podcast.xml            # Course1 audio feed
    ├── articles.xml           # Course1 articles feed
    ├── lecture.mp4
    ├── lecture.mp4.json       # Auto-generated metadata
    └── reflection.md          # Optional: reflection questions
```

### feed.json Format

```json
{
  "title": "Course 1: Introduction to AI",
  "description": "All materials for Course 1",
  "language": "en-us",
  "author": "Course Author"
}
```

**Required:** `title`
**Optional:** `description`, `language`, `author`, `link`, `image`

### reflection.md (Optional)

Place a `reflection.md` file in any folder to focus summarization on specific questions:

```markdown
# Reflection Questions

1. What are the key frameworks or mental models presented?
2. What analytical tools can help develop critical thinking?
3. How does this content relate to systems thinking?
```

**Note:** Works with both MP4 and PDF/HTML files. Completely optional.

---

## Feed Generation Rules

- **Podcast feed** (`podcast.xml`): Generated only if `feed.json` exists AND at least one MP4 file is processed with transcribe+summarize complete
- **Articles feed** (`articles.xml`): Generated only if `feed.json` exists AND at least one PDF/HTML file is processed with extract+summarize complete
- **Root-level feeds**: Require `feed.json` in the data directory root
- **Per-folder feeds**: Require `feed.json` in the subfolder
- **Index page** (`index.html`): Auto-generated at root, lists all feeds from all folders

---

## Supported File Types

- **MP4 (audio)** - Transcribed (remote-only) + summarized, Podcast feed, `description=""`, `summary`=key takeaways, `content`=transcript
- **PDF** - Metadata + content extracted, Articles feed, `description`=metadata, `content`=text, `summary`=null (unless summarized)
- **HTML** - Metadata + content extracted, Articles feed, `description`=metadata, `content`=text, `summary`=null (unless summarized)

### JSON Metadata

Metadata JSON files are stored alongside their source files. Titles use a `title` object with `filename`, `extracted`, and `generated` values; generated titles take precedence over extracted titles, which take precedence over filenames. **Note: filepath is NOT stored in JSON metadata; it is inferred from the JSON file's location.**

---

## Requirements

### Software
- Python 3.14+
- uv - For dependency management
- LLM Endpoint - OpenAI-compatible API for summarization (can be remote or local)
- Transcription Endpoint - Remote OpenAI-compatible API for transcription (required, no local transcription)
- PyAV (`av` package) - For audio metadata extraction

### Models
- Summarization: Any OpenAI-compatible model with 128K+ context window recommended
- Transcription: Any OpenAI-compatible whisper model on remote endpoint

---

## Troubleshooting

### Transcription Issues

- **Remote endpoint unavailable**: Verify `TRANSCRIPTION_ENDPOINT` and `TRANSCRIPTION_API_KEY`
- **Audio exceeds maximum duration**: Configure vLLM server with `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600` (or higher)

### Feed Issues

- **Missing feeds**: Ensure `feed.json` exists and files are processed (check `.json` metadata)
- **No root feeds**: Root-level `feed.json` is required
- **Invalid characters**: Control characters in PDF/HTML are automatically removed

### File Processing Issues

- **Special characters**: Warnings are generated; filenames are preserved
- **Orphaned JSON**: Delete `.json` file to re-process
- **File modified after processing**: File modification time is stored and checked; use `--force` to reprocess
- **Force re-processing**: Use `--force` flag

---

## Project Artifacts

- [Proposal](./artifacts/proposal.md) - Problem and scope
- [Specifications](./artifacts/specs.md) - Detailed requirements
- [Design Document](./artifacts/design.md) - Architecture
- [Tasks](./artifacts/tasks.md) - Implementation plan
