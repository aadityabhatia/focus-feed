# Local Feed Generator

**Transform your education content into mobile-friendly RSS feeds for podcast and article consumption.**

This is a personal feed generator that processes your education content files (PDF, HTML, MP4 audio) and creates standardized RSS/Atom feeds. These feeds can be consumed through mobile applications like **Antennapod** for audio content and **any RSS reader** for text content.

## Synopsis

You have education content as a disorganized collection of PDF documents, HTML files, and MP4 audio recordings. There's no easy way to browse, track, or consume this content across your devices. The feed generator solves this by:

1. **Processing your files** - Scans your `data` directory (including subfolders) for content
2. **Transcribing audio** - Converts MP4 files to markdown text using local AI models on your Nvidia GPU, or optionally via a remote OpenAI-compatible API
3. **Generating summaries** - Creates key takeaways from audio content using a local OpenAI-compatible LLM
4. **Extracting metadata** - Pulls titles and descriptions from PDF/HTML files
5. **Creating feeds** - Generates two RSS/Atom feeds that work with mobile apps:
   - `podcast.xml` - For audio content (Antennapod and other podcast apps)
   - `articles.xml` - For text content (any RSS reader app)

Your existing web server serves the `data` directory, making the feeds accessible to your mobile devices. All processing happens locally on your infrastructure - no cloud services required.

## Quick Start

1. **Place your education content** in the `data/` directory (supports subfolders)
2. **Configure environment** variables (see Configuration below)
3. **Process your files**:
   ```bash
   cd local-feed-generator
   uv run local-feed-generator process --data-dir ../data
   ```
4. **Access your feeds** at `{BASE_URL}/podcast.xml` and `{BASE_URL}/articles.xml`

### Quick Examples

```bash
# Process a single file
uv run local-feed-generator process --data-dir ./data --file videos/lecture.mp4

# Force reprocessing of a file (even if already complete)
uv run local-feed-generator process --data-dir ./data --file videos/lecture.mp4 --force

# Transcribe a specific MP4 file
uv run local-feed-generator transcribe --data-dir ./data --file videos/lecture.mp4

# Generate feeds only (no processing)
uv run local-feed-generator feeds --data-dir ./data
```

## Configuration

The feed generator is configured through environment variables. Create a `.env` file or set environment variables directly.

### Required Variables

- **BASE_URL**: Root URL where your data directory is served (Example: `https://your-domain.example.com`)
- **DATA_DIR**: Path to your education content directory (Example: `./data`)
- **LLM_ENDPOINT**: OpenAI-compatible API endpoint for summarization (Example: `https://your-llm-endpoint.example.com/v1`)
- **LLM_API_KEY**: API key for your LLM endpoint (Example: `your-api-key-here`)
- **LLM_MODEL**: Model identifier to use for LLM summarization (Example: `mistralai/Mistral-7B-Instruct`)
- **LLM_SYSTEM_PROMPT**: System prompt for LLM summarization (Example: `"Summarize key takeaways..."`)

### Optional Variables

- **FEED_FORMAT**: Default: `rss`. Feed format: `rss` or `atom`
- **TRANSCRIPTION_MODEL**: Default: `large-v3-turbo`. Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3-turbo`
- **TRANSCRIPTION_DEVICE**: Default: `cuda`. Device for transcription: `cuda` (recommended) or `cpu`
- **TRANSCRIPTION_COMPUTE_TYPE**: Default: `float16`. Compute type: `float16` (best quality), `int8` (lower memory)
- **TRANSCRIPTION_ENDPOINT**: OpenAI-compatible API endpoint for remote transcription (Example: `https://your-transcription-endpoint.example.com/v1`)
- **TRANSCRIPTION_API_KEY**: API key for your remote transcription endpoint (Example: `your-transcription-api-key-here`)
- **TRANSCRIPTION_REMOTE_MODEL**: Optional model override for remote transcription endpoint (Example: `whisper-1`)



### Server Configuration Notes

When using **remote transcription with a vLLM server**, the server must be configured with the `VLLM_MAX_AUDIO_DECODE_DURATION_S` environment variable to handle longer audio files. This is a **server-side configuration** that controls the maximum audio duration (in seconds) that can be decoded in a single request.

- **Requirement:** Set `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600` (or higher) to handle 1-hour audio files
- **Example:** For handling 1-hour audio files, set the environment variable to `3600`
- **Note:** This configuration is applied to the vLLM server itself, not the feed generator client

### Example `.env` File

```bash
# Required configuration
BASE_URL=https://your-domain.example.com
DATA_DIR=./data
LLM_ENDPOINT=https://your-llm-endpoint.example.com/v1
LLM_API_KEY=your-api-key-here
LLM_MODEL=your-model-name
LLM_SYSTEM_PROMPT="Summarize key takeaways from this document. Focus on frameworks or analytical tools to develop critical, creative, and systems thinking."

# Optional: Feed format (rss or atom)
FEED_FORMAT=rss

# Optional: Transcription settings
# For best quality with sufficient GPU memory:
TRANSCRIPTION_MODEL=large-v3-turbo
TRANSCRIPTION_DEVICE=cuda
TRANSCRIPTION_COMPUTE_TYPE=float16

# For lower GPU memory usage:
# TRANSCRIPTION_MODEL=small
# TRANSCRIPTION_COMPUTE_TYPE=int8

# Optional: Remote transcription configuration
# When TRANSCRIPTION_ENDPOINT is configured, uses OpenAI-compatible API instead of local faster-whisper
# This is the default transcription mode when TRANSCRIPTION_ENDPOINT is set
# TRANSCRIPTION_ENDPOINT=https://your-transcription-endpoint.example.com/v1
# TRANSCRIPTION_API_KEY=your-transcription-api-key-here
# TRANSCRIPTION_REMOTE_MODEL=whisper-1
```

## Transcription Model Configuration

By default, audio transcription uses **local mode** with the `faster-whisper` library on your Nvidia GPU. This provides full offline capability with no external dependencies.

When **remote mode** is activated by configuring `TRANSCRIPTION_ENDPOINT`, the feed generator will automatically use the OpenAI-compatible API at that endpoint for transcription instead of local processing. The remote endpoint determines which model to use by default, and **remote mode becomes the default transcription method** when `TRANSCRIPTION_ENDPOINT` is set.

You can optionally specify `TRANSCRIPTION_REMOTE_MODEL` if your endpoint requires an explicit model parameter, but in most cases this is not needed as the endpoint will use its default transcription model.

Fallback behavior: When using remote mode, if the remote endpoint fails or is unavailable, the system will not automatically fall back to local transcription. Both modes are mutually exclusive based on the `TRANSCRIPTION_ENDPOINT` configuration.

## Requirements

### Hardware
- **Nvidia GPU with CUDA** - Required for efficient audio transcription
- **CUDA Toolkit** - Must be installed and configured

### Software
- **Python 3.14 or later**
- **uv** - For dependency management
- **Local LLM Endpoint** - Running OpenAI-compatible API (e.g., vLLM, Ollama)
- **Optional**: Remote OpenAI-compatible transcription endpoint - Required only if using remote transcription mode (when `TRANSCRIPTION_ENDPOINT` is configured)

### Models
- **Transcription**: `faster-whisper` models downloaded automatically on first run (~2-6 GB)
- **Summarization**: Any OpenAI-compatible model with 128K+ context window recommended

## Usage

### Process All Files (Batch Mode)

Process all files in your data directory at once:

```bash
uv run local-feed-generator process --data-dir ./data
```

This will:
- Scan all subfolders
- Process new files (those without `.json` metadata)
- Skip already-processed files
- Generate both feeds

### Watch for Changes (Incremental Mode)

Continuously monitor your data directory for new or updated files:

```bash
uv run local-feed-generator watch --data-dir ./data
```

This will:
- Watch for file system changes
- Automatically process new files
- Regenerate feeds after each processing
- Warn about orphaned JSON files and special characters

### Generate Feeds Only

If you've added JSON metadata manually and just need to rebuild feeds:

```bash
uv run local-feed-generator feeds --data-dir ./data
```

### Validate Metadata

Check all JSON metadata files for validity:

```bash
uv run local-feed-generator validate --data-dir ./data
```

### Granular Processing (Step-by-Step)

For more control over the processing pipeline, you can run individual steps
instead of using the all-in-one `process` command. This is useful when you want
to run long-running steps (e.g. transcription) separately or inspect intermediate
results.

**For MP4 (audio) files:**

```bash
# Step 1: Transcribe audio to markdown text
uv run local-feed-generator transcribe --data-dir ./data

# Step 2: Generate summaries from transcripts
uv run local-feed-generator summarize --data-dir ./data

# Step 3: Validate and finalize metadata (JSON finalization happens here)
uv run local-feed-generator validate --data-dir ./data
```

**For PDF/HTML (document) files:**

```bash
# Step 1: Extract metadata (title, description, date)
uv run local-feed-generator extract --data-dir ./data

# Step 2: Validate and finalize metadata (JSON finalization happens here)
uv run local-feed-generator validate --data-dir ./data
```

**Then generate feeds (JSON finalization also happens here):**

```bash
uv run local-feed-generator feeds --data-dir ./data
```

#### Processing Pipeline

The granular processing pipeline works as follows:

1. **scan** → The scanner discovers MP4, PDF, and HTML files in your data directory
2. **transcribe** → MP4 files are transcribed to markdown text using local AI
3. **summarize** → Transcripts are summarized by the LLM to produce key takeaways
4. **extract** → PDF/HTML files have their metadata (title, description) extracted
5. **feeds** → All valid metadata is aggregated into `podcast.xml` and `articles.xml`

#### Using `--file` and `--force`

All granular commands support the `--file` and `--force` flags for targeted processing:

```bash
# Transcribe a single file
uv run local-feed-generator transcribe --data-dir ./data --file videos/lecture.mp4

# Force re-transcription of a file
uv run local-feed-generator transcribe --data-dir ./data --file videos/lecture.mp4 --force

# Preview which files would be processed (no changes made)
uv run local-feed-generator transcribe --data-dir ./data --dry-run
```

- **`--file <path>`**: Process only the specified file instead of scanning the entire directory. The path can be relative to the data directory or absolute.
- **`--force`**: Run the step even if it appears to already be complete. For example, `--force` on `transcribe` will re-transcribe files that already have `transcribe.status="complete"`.
- **`--dry-run`**: Available on `transcribe`, `summarize`, and `extract`. Shows which files would be processed without making any changes.

All subcommands also support `--config-file` to specify a custom `.env`-style configuration file.

## CLI Command Reference

The `local-feed-generator` CLI provides the following subcommands. All commands
support `--data-dir` (optional, defaults to `DATA_DIR` from configuration),
`--config-file`, `--file`, and `--force`. The `--dry-run` flag is available on
granular commands (`transcribe`, `summarize`, `extract`).

### `process`

Process all files in the data directory (batch mode). Scans for content files
(MP4, PDF, HTML), processes each through the appropriate pipeline, and generates
both feeds afterwards.

**Flags:**
- `--data-dir <path>` — Path to the data directory (optional; defaults to `DATA_DIR`)
- `--config-file <path>` — Path to a `.env`-style configuration file
- `--file <path>` — Process only the specified file instead of scanning the directory
- `--force` — Re-process all files even if already feed-eligible

**Examples:**
```bash
# Process all new files in the data directory
uv run local-feed-generator process --data-dir ./data

# Process a single file
uv run local-feed-generator process --data-dir ./data --file videos/lecture.mp4

# Force reprocessing of all files
uv run local-feed-generator process --data-dir ./data --force
```

### `watch`

Watch the data directory for changes (incremental mode). Starts a file-system
watcher that processes new and modified files in real-time, regenerating feeds
after each file is processed. Press `Ctrl+C` to stop.

**Flags:**
- `--data-dir <path>` — Path to the data directory (optional; defaults to `DATA_DIR`)
- `--config-file <path>` — Path to a `.env`-style configuration file
- `--file <path>` — Process only the specified file on startup, then watch it
- `--force` — Process all files on startup even if already feed-eligible

**Examples:**
```bash
# Watch for new files and changes
uv run local-feed-generator watch --data-dir ./data

# Watch a specific file
uv run local-feed-generator watch --data-dir ./data --file videos/lecture.mp4
```

### `feeds`

Generate RSS/Atom feeds from existing JSON metadata (no processing). Reads all
existing JSON metadata files, **finalizes JSON metadata** by computing feed eligibility for all items, and generates `podcast.xml` and `articles.xml`.

**Flags:**
- `--data-dir <path>` — Path to the data directory (optional; defaults to `DATA_DIR`)
- `--config-file <path>` — Path to a `.env`-style configuration file
- `--file <path>` — Include only the specified file's metadata in the feeds
- `--force` — Regenerate feeds regardless of current state

**Examples:**
```bash
# Generate feeds from existing metadata
uv run local-feed-generator feeds --data-dir ./data

# Regenerate feeds for a single file
uv run local-feed-generator feeds --data-dir ./data --file videos/lecture.mp4 --force
```

### `validate`

Validate all metadata files and finalize incomplete ones. Scans for all JSON
metadata files, validates each against the schema, checks for orphaned JSON
files, checks for special characters in filenames, and **finalizes JSON metadata** by setting step completion statuses in the step maps and computing feed eligibility.

**Flags:**
- `--data-dir <path>` — Path to the data directory (optional; defaults to `DATA_DIR`)
- `--config-file <path>` — Path to a `.env`-style configuration file
- `--file <path>` — Validate only the specified file's JSON metadata
- `--force` — Perform validation regardless of previous validation state

**Examples:**
```bash
# Validate all metadata files
uv run local-feed-generator validate --data-dir ./data

# Validate a single file
uv run local-feed-generator validate --data-dir ./data --file videos/lecture.mp4
```

### `transcribe`

Transcribe MP4 files to markdown text. Scans for MP4 files that do not yet have
`transcribe.status="complete"`, transcribes each using the faster-whisper model (or remote API
when configured), and writes the transcript to the `content` field in the adjacent JSON metadata file.

**Flags:**
- `--data-dir <path>` — Path to the data directory (optional; defaults to `DATA_DIR`)
- `--config-file <path>` — Path to a `.env`-style configuration file
- `--file <path>` — Transcribe only the specified MP4 file
  - `--force` — Re-transcribe even if `transcribe.status` is already `"complete"`
- `--dry-run` — Preview which files would be transcribed without making changes

**Examples:**
```bash
# Transcribe all MP4 files that need it
uv run local-feed-generator transcribe --data-dir ./data

# Transcribe a single file
uv run local-feed-generator transcribe --data-dir ./data --file videos/lecture.mp4

# Force re-transcription
uv run local-feed-generator transcribe --data-dir ./data --file videos/lecture.mp4 --force

# Preview files that would be transcribed
uv run local-feed-generator transcribe --data-dir ./data --dry-run
```

### `summarize`

Generate summaries from existing transcripts. Scans for MP4 files that have a
transcript but no summary, generates key takeaways via the LLM, and writes the
summary to the `summary` field in the adjacent JSON metadata file.

**Flags:**
- `--data-dir <path>` — Path to the data directory (optional; defaults to `DATA_DIR`)
- `--config-file <path>` — Path to a `.env`-style configuration file
- `--file <path>` — Summarize only the specified MP4 file
  - `--force` — Re-summarize even if `summarize.status` is already `"complete"`
- `--dry-run` — Preview which files would be summarized without making changes

**Examples:**
```bash
# Summarize all MP4 files that need it
uv run local-feed-generator summarize --data-dir ./data

# Summarize a single file
uv run local-feed-generator summarize --data-dir ./data --file videos/lecture.mp4

# Force re-summarization
uv run local-feed-generator summarize --data-dir ./data --file videos/lecture.mp4 --force

# Preview files that would be summarized
uv run local-feed-generator summarize --data-dir ./data --dry-run
```

### `extract`

Extract metadata from PDF/HTML files. Scans for PDF and HTML files that do not
yet have complete metadata extraction, extracts title, description, date, and
size using pypdf or BeautifulSoup, and writes the extracted data to the adjacent
JSON metadata file.

**Flags:**
- `--data-dir <path>` — Path to the data directory (optional; defaults to `DATA_DIR`)
- `--config-file <path>` — Path to a `.env`-style configuration file
- `--file <path>` — Extract metadata from only the specified PDF/HTML file
  - `--force` — Re-extract even if `extract.status` is already `"complete"`
- `--dry-run` — Preview which files would be processed without making changes

**Examples:**
```bash
# Extract metadata from all PDF/HTML files that need it
uv run local-feed-generator extract --data-dir ./data

# Extract metadata from a single file
uv run local-feed-generator extract --data-dir ./data --file documents/report.pdf

# Force re-extraction
uv run local-feed-generator extract --data-dir ./data --file documents/report.pdf --force

# Preview files that would be processed
uv run local-feed-generator extract --data-dir ./data --dry-run
```

### Shared Flags

- **`--data-dir <path>`**: All commands. Path to the data directory. Defaults to `DATA_DIR` from configuration.
- **`--config-file <path>`**: All commands. Path to a `.env`-style configuration file.
- **`--file <path>`**: All commands. Process only the specified file instead of scanning the directory. Takes precedence over `--data-dir` for file selection.
- **`--force`**: All commands. Force execution even if the step appears already complete.
- **`--dry-run`**: transcribe, summarize, extract. Show which files would be processed without making any changes.

## Directory Structure

```
data/
├── podcast.xml          # Audio content feed (MP4 files)
├── articles.xml         # Text content feed (PDF/HTML files)
├── Course1/
│   ├── lecture.mp4
│   ├── lecture.mp4.json # Auto-generated metadata
│   └── notes.pdf
├── Course2/
│   ├── document.html
│   └── document.html.json # Auto-generated metadata
└── ...
```

Each processed file gets a companion `.json` file containing:
- Extracted title and description (underscores in titles are replaced with spaces)
- File metadata (type, size, date as UNIX epoch timestamps)
- For MP4 files: `description` is blank, `summary` contains LLM-generated key takeaways, `content` contains the full transcript
- For PDF/HTML files: `description` contains extracted metadata, `summary` and `content` are null
- Step maps (`transcribe`, `summarize`, `extract`) tracking processing progress with detailed timing and metrics
- `duration` field for MP4 files
- `creation_time` field in ISO8601 format

## Features

### Smart Processing
- **Idempotent**: Skips files with existing JSON metadata
- **Incremental**: Only processes new or changed files
- **Flexible**: Works with any folder structure under `data/`

### Audio Processing
- **GPU-accelerated**: Uses `faster-whisper` with CUDA for efficient transcription
- **High quality**: Recommended model for best results
- **Markdown output**: Clean transcripts without timestamps stored in `content` field
- **Summaries**: LLM-generated key takeaways stored in `summary` field become podcast descriptions

### Feed Generation
- **Two separate feeds**: Podcast (audio) and Articles (text)
- **Standards-compliant**: RSS or Atom format
- **Sorted alphabetically**: By folder name, then filename
- **Proper encoding**: XML special characters properly escaped

### Error Handling
- **Persistent error state**: Failed files marked in JSON for review
- **Warnings**: Special characters and orphaned JSON files flagged
- **Recoverable**: Delete JSON file to retry processing

## File Types Supported

- **MP4 (audio)**: Transcribed + summarized (Feed: Podcast feed). Fields: `description=""`, `summary`=LLM key takeaways, `content`=full transcript, `duration`=audio length in seconds
- **PDF**: Metadata extracted (Feed: Articles feed). Fields: `description`=extracted metadata, `summary`=null, `content`=null
- **HTML**: Metadata extracted (Feed: Articles feed). Fields: `description`=extracted metadata, `summary`=null, `content`=null

## How It Works

1. **Scanning**: The scanner discovers MP4, PDF, and HTML files in your data directory
2. **Processing**: 
   - Audio files → transcribed to markdown → summarized by LLM → stored in `content` and `summary` fields
   - Documents → metadata extracted (title, author) → stored in `description` field → LLM generates title if missing
3. **Metadata**: JSON file created alongside each source file with all extracted information including step maps
4. **Feed Generation**: All valid JSON files aggregated into two feeds (podcast and articles)
   - Podcast feed uses `summary` field for descriptions
   - Articles feed uses `description` field for descriptions
5. **Serving**: Your existing web server serves the `data` directory, making feeds accessible

## Tips

### Choosing Transcription Model

| Model | Size | Quality | GPU Memory |
|-------|------|---------|------------|
| `tiny` | 39M | Basic | ~1 GB |
| `base` | 74M | Good | ~2 GB |
| `small` | 244M | Very Good | ~4 GB |
| `medium` | 769M | Excellent | ~8 GB |
| `large-v3-turbo` | ~800M | **Best** | ~10 GB |
| `large` | 1550M | Best (slower) | ~12 GB |

Use the recommended model with `float16` for maximum quality if you have sufficient GPU memory. Use `small` or `base` with `int8` for lower memory usage.

### LLM Recommendations

- Use a model with **128K+ context window** for full transcript processing
- Any **OpenAI-compatible** endpoint works (vLLM, Ollama, etc.)
- **Mistral models** work well for summarization tasks

## Configuration for Different Environments

### Development (Lower GPU Memory)

```bash
TRANSCRIPTION_MODEL=small
TRANSCRIPTION_COMPUTE_TYPE=int8
```

### Production (Best Quality)

```bash
TRANSCRIPTION_MODEL=large-v3-turbo
TRANSCRIPTION_DEVICE=cuda
TRANSCRIPTION_COMPUTE_TYPE=float16
```

## Troubleshooting

### GPU Memory Errors

If you encounter CUDA out-of-memory errors:
1. Try a smaller model: `large-v3-turbo`, `medium`, `small`, `base`, `tiny`
2. Use `int8` instead of `float16` compute type
3. Ensure no other GPU-intensive processes are running

### LLM API Connection Issues

Verify your LLM endpoint is running and accessible:
- Check `LLM_ENDPOINT` URL
- Verify `LLM_API_KEY` is correct
- Test the endpoint with curl:
  ```bash
  curl -X POST -H "Authorization: Bearer $LLM_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model": "$LLM_MODEL", "messages": [{"role": "user", "content": "test"}]}' \
    $LLM_ENDPOINT/chat/completions
  ```

### File Processing Issues

- Ensure files are readable and not corrupted
- Check for special characters in filenames (generated warnings)
- Delete `.json` file to force re-processing

### Remote Transcription Issues

- **Audio exceeds maximum allowed duration**: If you receive an error that audio exceeds the maximum allowed duration when using remote transcription with vLLM, ensure the vLLM server is configured with `VLLM_MAX_AUDIO_DECODE_DURATION_S` (see Server Configuration Notes section above). Restart the vLLM server after setting this variable.

## Project Artifacts

For implementation details and development information:

- [Proposal](./artifacts/proposal.md) - Problem statement and success criteria
- [Specifications](./artifacts/specs.md) - Complete functional/non-functional requirements
- [Design Document](./artifacts/design.md) - Architecture, interfaces, and design decisions
- [Tasks](./artifacts/tasks.md) - Detailed implementation plan
