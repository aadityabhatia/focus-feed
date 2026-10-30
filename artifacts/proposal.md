# Education Content Feed System - Proposal

## Problem
Education files (PDF, HTML, MP4) in `data/` are disconnected. Users cannot consume them efficiently across devices via standard apps.

## User Intent
Make content accessible via private RSS feeds hosted on an existing internal web server:
- Podcast feed for Antennapod and podcast apps
- Articles feed for RSS readers
Per-folder feeds via `feed.json`, with auto-generated root `index.html` for discovery.

## Scope

### In Scope
- **Input:** PDF, HTML, MP4 in `data/` directory (at workspace root) and its subfolders
- **Processing:** Batch and incremental watch mode
  - MP4: transcribe to markdown (remote-only), LLM key takeaway summary
  - PDF/HTML: extract metadata + content; summarization required for feed eligibility
  - Track steps: `transcribe`, `summarize`, `extract`, `generate_title`
  - File modification time stored in JSON and checked on each process
- **Metadata:** `<filename>.json` per file with title dict `{filename,extracted,generated}`, description, summary, content, filename, filetype, dates (ms), mtime, size, duration/creation_time, hash, step maps `transcribe/summarize/extract/generate_title`
  - filepath is NOT stored; inferred from JSON filename location
  - MP4: description blank, summary = takeaways, content = transcript
  - PDF/HTML: description = extracted metadata, content = plaintext, summary null unless summarized
  - SHA256 hash computed during first extraction
- **Feeds:** `podcast.xml` for MP4, `articles.xml` for PDF/HTML, per-folder only when `feed.json` exists
- **Features:** `reflection.md` injection, auto folder-name title prefix, markdown→HTML conversion, control character sanitization

### Out of Scope
- AI agents for metadata gap filling
- Cloud services by default; remote transcription required
- Local transcription (faster-whisper removed)
- Web server deployment / public distribution

## Success Criteria
- Content consumable via standard RSS/podcast apps
- Automatic transcription (remote-only) + summarization for MP4, extraction for PDF/HTML
- Local-first processing for extraction/feed generation, remote-only for transcription
- Incremental skips completed files unless mtime changed
- Warnings for orphaned JSONs and special characters
- Millisecond-precision timestamps, feed eligibility from step statuses
- Root `index.html` lists all feeds
