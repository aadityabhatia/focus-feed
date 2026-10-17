# Education Content Feed System - Proposal

## Problem Statement

Education content exists as disconnected files (PDF, HTML, MP4) in a `data` directory with no unified access method. Users cannot efficiently consume this content across devices through standard applications.

## User Intent

Make education content readily available for personal consumption through mobile applications:
- **Podcast feed** for Antennapod and other podcast apps (audio content)
- **Articles feed** for RSS reader apps (text content from PDF/HTML)

Feeds are hosted on an existing internal web server, accessible only to the user, and organized with:
- Per-folder feeds with custom titles and descriptions via `feed.json`
- Auto-generated root-level `index.html` page for feed discovery

## Scope

### In Scope

**Input & Processing:**
- Input files: PDF, HTML, and MP4 in `data` directory and subfolders
- Processing modes: batch and incremental (file watching)
- Audio: Transcribe MP4 to markdown; generate key takeaway summaries using local/compatible LLM
- Documents: Extract metadata AND content from PDF/HTML; optional summarization
- Track processing via step maps: `transcribe`, `summarize`, `extract`

**Metadata:**
- Generate JSON metadata with `.json` extension for each file
- Fields: `title` (dict: filename/extracted/generated), `extracted_title`, `description`, `summary`, `content`, `filename`, `filepath`, `filetype`, `date` (UNIX epoch ms), `size`, `publication_date` (PDF/HTML), `duration`, `creation_time` (MP4), `hash`
- MP4: blank description, key takeaways in summary, full transcript in content
- PDF/HTML: extracted metadata in description, extracted content in content, null summary unless explicitly summarized

**Feed Generation:**
- Podcast RSS feed for MP4 files
- Articles RSS feed for PDF/HTML files
- Per-folder feeds when `feed.json` exists
- Conditional: feeds only generated where `feed.json` exists AND applicable files present
- Root-level feeds require explicit `feed.json` in data directory root
- Auto-generated root-level `index.html` listing all feeds

**Features:**
- Reflection questions via `reflection.md` files in source folders
- Automatic folder name prefixing for titles
- Markdown to HTML conversion for feed descriptions
- Control character normalization for XML safety

### Out of Scope

- AI agent for filling metadata gaps
- Cloud-based services (all processing local; remote transcription optional)
- Web server deployment
- Public distribution

## Success Criteria

- All education content in `data` directory accessible through standard RSS feeds
- Audio content transcribed and summarized automatically
- PDF/HTML files have content extracted and optionally summarized
- Content consumable through Antennapod and any RSS reader
- No dependency on external/cloud services (remote transcription excepted)
- Only new or updated files processed (skips complete JSON metadata)
- Orphaned JSON files and special characters generate warnings
- Millisecond precision timestamps throughout
- Feed eligibility computed from step map statuses
- Root-level `index.html` lists all available feeds
