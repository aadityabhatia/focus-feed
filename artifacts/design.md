# Local Feed Generator - Design

**Runner:** All commands run via `uv run local-feed-generator <command>` from project root.

## 1. Overview
Standalone app that processes MP4, PDF, HTML files in `data/` and generates private RSS feeds for personal consumption. Local-first by default; remote transcription optional.

**Goals**
- Privacy-first, local processing
- Idempotent, skips completed files
- Mobile-ready RSS feeds
- Per-folder feeds via `feed.json`
- Auto root `index.html` discovery

## 2. Architecture

**Components**
- **Scanner & Watcher (`watchdog`):** Recursively discovers files, debounced change detection, periodic rescan fallback for unreliable events
- **Producer Pipeline:** Queue manager that decides skip/process/resume based on JSON step maps
  - Transcription Service: MP4 → markdown via faster-whisper local CUDA or remote OpenAI-compatible API
  - Summarization Service: LLM key takeaways from transcript/extracted text, with reflection.md injection
  - Extraction Service: PDF/HTML metadata+plaintext via `pypdf`/`beautifulsoup4`; audio metadata via `av`
- **Feed Generator (`feedgen`):** Aggregates eligible metadata per feed.json, builds RSS items with enclosure links, pubDate logic, and GUIDs; writes podcast.xml / articles.xml
- **Index Generator:** Builds root index.html with links/titles/descriptions from all feed.json configs
- **CLI (`click`) + Config (`pydantic-settings` + `.env`)**

**Data Flow**
1. Scanner discovers file in data/ recursively
2. If JSON exists, check step statuses. Skip if all applicable steps complete; otherwise resume incomplete steps
3. Dispatch by type:
   - MP4: load reflection.md if present → transcribe to markdown → store in content → summarize with reflection prompt → extract audio metadata via av → build title dict → apply parent folder prefix
   - PDF/HTML: extract metadata & plaintext → sanitize control chars → store description/content → build title dict → apply folder prefix
4. Write/update JSON with step maps, hash, timestamps
5. Feed Generator scans for feed.json, collects feed-eligible items from subtree, sorts by folder/filename, converts markdown→HTML with XSS protection, writes podcast.xml / articles.xml
6. Index Generator updates root index.html with all feeds

## 3. Interfaces & Contracts

**CLI**
`uv run local-feed-generator <process|watch|feeds|validate|transcribe|summarize|extract|create-feed>`
Flags: `--data-dir`, `--config-file`, `--file`, `--force`, `--dry-run`

**Environment**
Required: BASE_URL, DATA_DIR, LLM_ENDPOINT, LLM_API_KEY, LLM_MODEL, PROMPT_ARTICLE, PROMPT_PODCAST, PROMPT_TITLE
Optional: PROMPT_ARTICLE_REFLECTION, PROMPT_PODCAST_REFLECTION, TRANSCRIPTION_ENDPOINT, TRANSCRIPTION_API_KEY, TRANSCRIPTION_MODEL, TRANSCRIPTION_DEVICE, TRANSCRIPTION_COMPUTE_TYPE, TRANSCRIPTION_REMOTE_MODEL

**JSON Metadata**
Per file `<name>.json` with title dict `{filename,extracted,generated}`, description, summary, content, filetype, dates ms, size, hash, step maps `transcribe/summarize/extract` with status/timing, error.
Title precedence: generated > extracted > filename. Parent folder name auto-prepended.

**Feed Eligibility**
MP4: transcribe.complete AND summarize.complete
PDF/HTML: extract.complete

**feed.json**
Required for feeds. Fields: title required, description, language, author, link, image optional. Aggregates subtree.

**reflection.md**
Optional per folder. Questions prepended to LLM summarization prompt. Uses reflection prompts if set.

## 4. Technology Stack

**Core deps**
feedgen, watchdog, openai, pydantic-settings, pydantic, click, pypdf, beautifulsoup4, av, markdown>=3.5.0, python-dotenv

**Transcription**
- Local default: faster-whisper with CUDA GPU, model large-v3-turbo float16
- Remote optional: OpenAI-compatible API via TRANSCRIPTION_ENDPOINT. Nvidia GPU not required for remote mode.

**Summarization**
OpenAI-compatible endpoint. Prompts per filetype. Context 128K+ recommended.

**Extraction**
PDF: pypdf; HTML: beautifulsoup4; Audio metadata: av library. Control characters removed for XML safety.

## 5. Key Decisions
- Title as dict with provenance for fallback and display
- LLM title generation for MP4s with sparse metadata
- Auto folder name prefix for context in readers
- Regenerate feeds/index after each processing
- Persistent error state in JSON, manual fix or --force
- Full plaintext extraction for richer feeds
- Markdown→HTML with XSS protection
- Per-folder feeds via feed.json only
- Optional reflection.md for focused summarization
- Millisecond precision timestamps
- Conditional generation: feed.json + eligible files required

## 6. Risks & Mitigations
Transcription quality → use established models
LLM inconsistency → tested prompts, configurable
Watcher misses events → periodic rescan
Special chars → escape output, warn
Large audio → acceptable for personal use
GPU memory → float16, smaller models
API unavailable → clear errors, retry
Control chars → auto remove
Missing feeds → document feed.json requirement

## 7. Assumptions & Dependencies
- data directory exists and writable
- Existing web server serves data directory
- Python 3.14+ and uv for dependency management
- Local LLM endpoint available
- Nvidia GPU optional for local transcription; required for best performance in local mode, not needed for remote transcription
- No ffprobe required; av library provides audio metadata
