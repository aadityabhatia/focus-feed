# Focus Feed - Design

**Runner:** All commands run via `uv run focus-feed <command>` from project root.

## 1. Overview
Standalone app that processes supported audio/video media, PDF, and HTML files and generates private RSS feeds for personal consumption. Remote transcription only; media metadata and audio extraction use PyAV.

**Goals**
- Privacy-first, local processing
- Idempotent, skips completed files
- Mobile-ready RSS feeds
- Per-folder feeds via `feed.json`
- Auto `index.html` discovery in every directory with `feed.json`

## 2. Architecture

**Components**
- **Scanner & Watcher (`watchdog`):** Recursively discovers files, debounced change detection, periodic rescan fallback for unreliable events
- **Producer Pipeline:** Queue manager that decides skip/process/resume based on JSON step maps
  - Transcription Service: compatible audio inputs → markdown via remote OpenAI-compatible API only
  - **Audio Extractor (`services/audio_extractor.py`):** Passes supported audio inputs through unchanged; extracts audio from MKV, MOV, AVI, and WMV to a temporary transcription-compatible file. Uses stream copy where compatible and FLAC conversion as fallback.
  - Summarization Service: LLM key takeaways from transcript/extracted text, with reflection.md injection
  - Extraction Service: PDF/HTML metadata+plaintext via `pypdf`/`beautifulsoup4`; media metadata via `av` (PyAV); hashing (SHA256) computed during first extraction
- **Feed Generator (`feedgen`):** Aggregates eligible metadata per feed.json, builds RSS items with enclosure links, `pubDate` from `summarize.end_time`, and GUIDs; writes podcast.xml / articles.xml
- **Index Generator:** Builds an `index.html` in every directory with `feed.json`; each lists XML feeds in its subtree, and the root index lists feeds from the whole tree
- **CLI (`click`) + Config (`pydantic-settings` + `.env`)**

**Data Flow**
1. Scanner discovers file recursively
2. If JSON exists, check file modification time against JSON mtime. If file unchanged and all applicable steps complete, skip; otherwise resume incomplete steps
3. Dispatch by type:
   - Direct-submit media (`.mp3`, `.m4a`, `.wav`, `.webm`, `.aac`, `.flac`, `.ogg`, `.opus`, `.mp4`, `.mpeg`, `.mpga`): send original file to remote transcription API → store transcript in content → summarize → extract media metadata via PyAV → generate title.
   - Video requiring extraction (`.mkv`, `.mov`, `.avi`, `.wmv`): audio_extractor selects the first audio stream and creates a temporary compatible audio file → remote transcription API → delete temporary file → store transcript in content → summarize → extract media metadata via PyAV → generate title. No-audio or extraction failures are reported as transcription errors.
   - PDF/HTML: extract metadata & plaintext → sanitize control chars → compute SHA256 hash → store description/content/mtime → generate_title via LLM → build title dict → apply folder prefix, using an extended filename code when present
4. Write/update JSON with step maps, hash, timestamps, mtime
5. Feed Generator scans for feed.json, collects feed-eligible items from subtree, sorts by folder/filename, converts markdown→HTML with XSS protection, writes podcast.xml / articles.xml
6. Index Generator updates an `index.html` in every directory with `feed.json`; the root index lists all feeds and each nested index lists feeds in its subtree

## 3. Interfaces & Contracts

**CLI**
`uv run focus-feed <process|watch|feeds|validate|transcribe|summarize|extract|title|create-feed|cleanup>`
Flags: `--config-file`, `--force`, `--dry-run`

**Environment**
Required: BASE_URL, LLM_ENDPOINT, LLM_API_KEY, LLM_MODEL, PROMPT_ARTICLE, PROMPT_PODCAST, PROMPT_TITLE
Optional: PROMPT_ARTICLE_REFLECTION, PROMPT_PODCAST_REFLECTION, TRANSCRIPTION_ENDPOINT, TRANSCRIPTION_API_KEY, TRANSCRIPTION_REMOTE_MODEL

**JSON Metadata**
Per file `<name>.json` with title dict `{filename,extracted,generated}`, description, summary, content, filetype, dates ms, mtime, size, hash, step maps `transcribe/summarize/extract/generate_title` with status/timing, error.
Title precedence: generated > extracted > filename. Parent folder name auto-prepended unless the filename begins with the folder name plus an extended code, in which case that code is used as the prefix (e.g. `H103RA` in folder `H103`).
Filepath is NOT stored; it is inferred from the JSON filename location.

**Feed Eligibility**
Transcribed audio/video: transcribe.complete AND summarize.complete
PDF/HTML: extract.complete AND summarize.complete

Note: Nested feeds include all content from their entire subtree; inner feeds are subsets of outer feeds.

**feed.json**
Required for feeds. Fields: title required, description, language, author, link, image optional. Aggregates subtree.

**reflection.md**
Optional per folder. Questions prepended to LLM summarization prompt. Uses reflection prompts if set.

## 4. Technology Stack

**Core deps**
feedgen, watchdog, openai, pydantic-settings, pydantic, click, pypdf, beautifulsoup4, av, markdown>=3.5.0, python-dotenv

**Transcription**
Remote only: OpenAI-compatible API via TRANSCRIPTION_ENDPOINT. No local mode.

**Summarization**
OpenAI-compatible endpoint. Prompts per filetype. Context 128K+ recommended.

**Extraction**
PDF: pypdf; HTML: beautifulsoup4; media metadata and audio extraction: av (PyAV). Control characters removed for XML safety. SHA256 hash computed during first extraction. Audio extraction uses compatible stream copy/remux when possible and FLAC conversion otherwise; quality is prioritized over temporary-file size.

## 5. Key Decisions
- Title as dict with provenance for fallback and display
- LLM title generation for MP4s with sparse metadata
- Auto folder name prefix for context in readers, extended to the filename code when present
- Regenerate feeds/index after each processing
- Persistent error state in JSON, manual fix or --force
- Full plaintext extraction for richer feeds
- Markdown→HTML with XSS protection
- Per-folder feeds via feed.json only
- Optional reflection.md for focused summarization
- Millisecond precision timestamps
- Conditional generation: feed.json + eligible files required
- Remote-only transcription
- File modification time stored in JSON; used to detect changes and trigger reprocessing

## 6. Risks & Mitigations
Transcription quality → use established models
LLM inconsistency → tested prompts, configurable
Watcher misses events → periodic rescan
Special chars → escape output, warn
Large audio → acceptable for personal use
API unavailable → clear errors, retry
Control chars → auto remove
Missing feeds → document feed.json requirement

## 7. Assumptions & Dependencies
- `data/` directory exists and is writable at the workspace root, alongside `src/` and `tests/`
- Existing web server serves data directory
- Python 3.14+ and uv for dependency management
- LLM endpoint available (can be remote or local)
- Transcription endpoint required; remote-only
- No ffprobe required; audio metadata uses PyAV
- Remote transcription: vLLM server requires `VLLM_MAX_AUDIO_DECODE_DURATION_S` configuration for longer audio files
