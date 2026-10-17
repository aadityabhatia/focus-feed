# Local Feed Generator

**Transform your education content into mobile-friendly RSS feeds for podcast and article consumption.**

## Project Summary

Turn disorganized PDF, HTML, and MP4 audio files into RSS feeds for mobile apps. 

- **Podcast feed** - For MP4 audio (Antennapod and other podcast apps)
- **Articles feed** - For PDF/HTML documents (any RSS reader)
- **Per-folder feeds** - Organize with `feed.json` in subfolders
- **Discovery** - Auto-generated `index.html` lists all available feeds

All processing happens locally on your infrastructure with optional remote transcription.

**Documentation:**
- [Proposal](./artifacts/proposal.md) - Problem and scope
- [Specifications](./artifacts/specs.md) - Detailed requirements
- [Design Document](./artifacts/design.md) - Architecture

---

## Quick Start

1. **Place your files** in the `data/` directory (supports subfolders)

2. **Configure** via environment variables or `.env` file

3. **Process your content:**
   ```bash
   uv run local-feed-generator process
   ```

4. **Access your feeds** at `{BASE_URL}/index.html`

---

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `BASE_URL` | Root URL where data directory is served (e.g., `https://example.com`) |
| `DATA_DIR` | Path to your content directory (e.g., `./data`) |
| `LLM_ENDPOINT` | OpenAI-compatible API endpoint for summarization |
| `LLM_API_KEY` | API key for LLM endpoint |
| `LLM_MODEL` | Model identifier for summarization |
| `PROMPT_ARTICLE` | Article summarization prompt (PDF/HTML) |
| `PROMPT_PODCAST` | Podcast summarization prompt (MP4) |
| `PROMPT_TITLE` | Title generation prompt |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMPT_ARTICLE_REFLECTION` | `PROMPT_ARTICLE` | Article summarization with reflection questions |
| `PROMPT_PODCAST_REFLECTION` | `PROMPT_PODCAST` | Podcast summarization with reflection questions |
| `TRANSCRIPTION_ENDPOINT` | - | Remote transcription API endpoint (enables remote mode) |
| `TRANSCRIPTION_API_KEY` | - | Remote transcription API key |
| `TRANSCRIPTION_REMOTE_MODEL` | - | Remote transcription model override |
| `TRANSCRIPTION_MODEL` | `large-v3-turbo` | Local transcription model |
| `TRANSCRIPTION_DEVICE` | `cuda` | Transcription device (cuda/cpu) |
| `TRANSCRIPTION_COMPUTE_TYPE` | `float16` | Compute type for transcription |

### Example `.env` File

```bash
# Required
BASE_URL=https://example.com
DATA_DIR=./data
LLM_ENDPOINT=https://llm.example.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=mistralai/Mistral-7B-Instruct

# Prompts
PROMPT_ARTICLE="Summarize key takeaways from this document concisely in a few paragraphs."
PROMPT_PODCAST="Summarize key takeaways from this podcast transcript concisely in a few paragraphs."
PROMPT_TITLE="Generate a concise, descriptive title (5-10 words) for the given content."

# Optional
PROMPT_ARTICLE_REFLECTION="Summarize with reflection questions: ..."
PROMPT_PODCAST_REFLECTION="Summarize with reflection questions: ..."

# Transcription (default: local faster-whisper on GPU)
TRANSCRIPTION_MODEL=large-v3-turbo
TRANSCRIPTION_DEVICE=cuda
TRANSCRIPTION_COMPUTE_TYPE=float16

# Remote transcription (optional - overrides local when set)
# TRANSCRIPTION_ENDPOINT=https://transcription.example.com/v1
# TRANSCRIPTION_API_KEY=your-transcription-api-key
```

**Note:** When `TRANSCRIPTION_ENDPOINT` is configured, remote transcription mode is used instead of local faster-whisper.

---

## Processing Modes

### Batch Mode

Process all files at once:
```bash
uv run local-feed-generator process
```

### Incremental Mode

Watch for new/updated files:
```bash
uv run local-feed-generator watch
```

---

## Granular Processing

Target specific steps or files:

| Command | Description |
|---------|-------------|
| `transcribe` | Transcribe MP4 files to markdown |
| `summarize` | Generate summaries from transcripts or document content |
| `extract` | Extract metadata from PDF/HTML files |
| `feeds` | Generate RSS feeds and index.html |
| `validate` | Validate all metadata files |
| `create-feed` | Create a template `feed.json` in a directory |

### Common Flags

| Flag | Description |
|------|-------------|
| `--data-dir <path>` | Path to data directory (default: `DATA_DIR`) |
| `--file <path>` | Process only this specific file |
| `--force` | Force re-processing even if complete |
| `--dry-run` | Preview without making changes (transcribe, summarize, extract) |
| `--config-file <path>` | Use custom `.env` file |

### Examples

```bash
# Process a single file
uv run local-feed-generator process --file videos/lecture.mp4

# Force re-transcription
uv run local-feed-generator transcribe --file videos/lecture.mp4 --force

# Summarize a PDF
uv run local-feed-generator summarize --file documents/report.pdf

# See what would be transcribed
uv run local-feed-generator transcribe --dry-run
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

- **Podcast feed** (`podcast.xml`): Generated only if `feed.json` exists AND at least one MP4 file is processed
- **Articles feed** (`articles.xml`): Generated only if `feed.json` exists AND at least one PDF/HTML file is processed
- **Root-level feeds**: Require `feed.json` in the data directory root
- **Per-folder feeds**: Require `feed.json` in the subfolder
- **Index page** (`index.html`): Auto-generated at root, lists all feeds from all folders

---

## Supported File Types

| File Type | Processing | Feed | Fields |
|-----------|------------|------|--------|
| MP4 (audio) | Transcribed + summarized | Podcast | `description=""`, `summary`=key takeaways, `content`=transcript |
| PDF | Metadata + content extracted | Articles | `description`=metadata, `content`=text, `summary`=null (unless summarized) |
| HTML | Metadata + content extracted | Articles | `description`=metadata, `content`=text, `summary`=null (unless summarized) |

---

## Requirements

### Hardware
- **Nvidia GPU with CUDA** - Required for efficient local transcription
- **CUDA Toolkit** - Must be installed

### Software
- **Python 3.14+**
- **uv** - For dependency management
- **Local LLM Endpoint** - Running OpenAI-compatible API (vLLM, Ollama, etc.)
- **Optional**: `ffprobe` - For audio metadata extraction (falls back to file metadata)

### Models
- **Transcription**: faster-whisper models (downloaded automatically, ~2-6 GB)
- **Summarization**: Any OpenAI-compatible model with 128K+ context window recommended

---

## Optional Dependencies

### Local Transcription (GPU-accelerated)

```bash
uv sync --optional-group local-transcription
```

Installs `faster-whisper` and `av` for local GPU-accelerated transcription.

### Markdown Conversion

For enhanced markdown to HTML conversion:
```bash
uv pip install markdown
```

---

## Troubleshooting

### Transcription Issues

- **Large files take time**: No time constraints; acceptable for personal use
- **CUDA out of memory**: Use smaller model (`medium`, `small`, `base`) or `int8` compute type
- **Model not found**: Verify model name or installation

### Remote Transcription Issues

- **Audio exceeds maximum duration**: Configure vLLM server with `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600` (or higher)
- **Connection errors**: Verify `TRANSCRIPTION_ENDPOINT` and `TRANSCRIPTION_API_KEY`

### Feed Issues

- **Missing feeds**: Ensure `feed.json` exists and files are processed (check `.json` metadata)
- **No root feeds**: Root-level `feed.json` is required
- **Invalid characters**: Control characters in PDF/HTML are automatically removed

### File Processing Issues

- **Special characters**: Warnings are generated; filenames are preserved
- **Orphaned JSON**: Delete `.json` file to re-process
- **Force re-processing**: Use `--force` flag

---

## Project Artifacts

- [Proposal](./artifacts/proposal.md) - Problem and scope
- [Specifications](./artifacts/specs.md) - Detailed requirements
- [Design Document](./artifacts/design.md) - Architecture
- [Tasks](./artifacts/tasks.md) - Implementation plan
