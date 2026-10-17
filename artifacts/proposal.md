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
- **Input:** PDF, HTML, MP4 in `data/` and subfolders
- **Processing:** Batch and incremental watch mode
  - MP4: transcribe to markdown, LLM key takeaway summary
  - PDF/HTML: extract metadata + content; optional summarization
  - Track steps: `transcribe`, `summarize`, `extract`
- **Metadata:** `<filename>.json` per file with title dict, description, summary, content, dates (ms), hash, size, duration/creation_time, step maps
  - MP4: description blank, summary = takeaways, content = transcript
  - PDF/HTML: description = extracted metadata, content = plaintext, summary null unless summarized
- **Feeds:** `podcast.xml` for MP4, `articles.xml` for PDF/HTML, per-folder only when `feed.json` exists
- **Features:** `reflection.md` injection, auto folder-name title prefix, markdown→HTML conversion, control character sanitization

### Out of Scope
- AI agents for metadata gap filling
- Cloud services by default; remote transcription optional
- Web server deployment / public distribution

## Success Criteria
- Content consumable via standard RSS/podcast apps
- Automatic transcription + summarization for MP4, extraction for PDF/HTML
- Local-first processing, incremental skips completed files
- Warnings for orphaned JSONs and special characters
- Millisecond timestamps, feed eligibility from step statuses
- Root `index.html` lists all feeds
