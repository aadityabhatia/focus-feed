# Education Content Feed System - Proposal

## Problem Statement

Currently, education content exists as a disorganized collection of files (PDFs, HTML documents, and MP4 audio recordings) with no coherent way to consume them. There is no unified method to access, browse, or track this content across devices.

## User Intent

The primary goal is to **make education content readily available for personal consumption** through mobile applications:
- **Podcast feed** for Antennapod (audio content)
- **Articles feed** for RSS reader apps (text content, including extracted content from PDF/HTML)

These feeds will be:
- Hosted on an existing internal web server
- Accessible only to the user
- Placed in the `data` directory at the root level and optionally in subfolders
- Organized with per-folder feeds with custom titles and descriptions
- Discovered via auto-generated root-level index.html page

## Scope

### In Scope
- **Input files**: PDF, HTML, and MP4 audio files in the `data` directory (can be in subfolders)
- **Processing modes**: Batch processing and incremental processing (watching for new files or JSON updates)
- **Audio processing**: Transcribe MP4 files to markdown, generate key takeaway summaries using a local OpenAI-compatible LLM with optional reflection questions
- **Document processing**: Extract metadata AND content from PDF/HTML files (not just metadata)
- **Document summarization**: Generate summaries from PDF/HTML extracted content using the same LLM summarization
- **Metadata**: Generate JSON metadata for each input file with `.json` extension (e.g., `file.mp4` \=> `file.mp4.json`). JSON metadata includes:
  - Basic file information (title as dict with sources, filename, filepath, filetype, date, size, version)
  - New: `title` field is a dict with keys: `filename`, `extracted`, `generated`, `final`
  - New: `extracted_title` field for storing source metadata titles separately
  - Content fields: `description` (blank for MP4, extracted metadata for PDF/HTML), `summary` (LLM-generated key takeaways for MP4 and optionally PDF/HTML), `content` (full transcript for MP4, extracted text for PDF/HTML)
  - Millisecond precision timestamps: `date` stored as UNIX epoch float with millisecond precision
  - Step maps for processing tracking: `transcribe` (with status, timing, metrics), `summarize` (with status, timing, metrics), `extract` (with status, timing, metrics)
  - New: `duration` field for MP4 (from audio metadata via ffprobe)
  - New: `creation_time` field for MP4 (ISO8601 string from audio metadata)
  - Feed eligibility computed from step map statuses
- **Feed generation**: Create two distinct feeds based on configuration:
  1. **Podcast RSS/Atom feed** for audio content (MP4 files with summaries as descriptions)
  2. **Articles RSS/Atom feed** for text content (PDF/HTML files with extracted content as descriptions)
  3. **Per-folder feeds**: Optional feeds in subfolders when `feed.json` exists
  4. **Conditional feed generation**: Feeds only generated where feed.json exists and applicable files are present
  5. **No root feeds without feed.json**: Root-level feeds require explicit feed.json configuration
- **Index pages**: Auto-generated `index.html` in each folder with feed.json, listing all available feeds
- **Configuration**: System prompts for LLM summarization and title generation stored in environment variables
- **Reflection support**: reflection.md files in source folders provide reflection questions for summarization
- **Module prefixing**: Automatic folder name prepend to titles when appropriate
- **New features**:
  - Reflection questions via `reflection.md` files in source folders
  - Per-folder feeds via `feed.json` configuration files
  - Markdown to HTML conversion for feed descriptions
  - Module name prefixing for titles (automatic folder name prepend)
  - Whitespace/control character normalization in extracted content
  - Index.html generation for feed discovery
  - Conditional feed generation (podcast only with MP4, articles only with PDF/HTML)

### Out of Scope
- AI agent for filling metadata gaps (deferred to future iteration)
- Cloud-based services (all processing must be local, except optional remote transcription)
- Web server deployment (existing server will serve the `data` directory)
- Public distribution

## Success Criteria

- All education content in the `data` directory is accessible through standard RSS/Atom feeds
- Audio content is transcribed and summarized automatically using local models
- **Documents (PDF/HTML) have their content extracted and included in feeds**
- **PDF/HTML documents can optionally be summarized**
- Content can be consumed through Antennapod and any RSS reader app
- No dependency on external/cloud services (optional remote transcription excepted)
- System processes only new or updated files (skips files with existing complete JSON metadata)
- Orphaned JSON files and special characters in filenames generate warnings
- MP4 files have blank description, key takeaways in summary, full transcript in content
- PDF/HTML files have extracted metadata in description, extracted content in content field, null summary (unless summarized)
- Processing progress tracked via step maps (transcribe, summarize, extract) instead of legacy status fields
- **Title field is a dict with sources**: filename, extracted, generated, final
- **Millisecond precision**: All date/timestamp fields support millisecond precision
- **Per-folder feeds**: Feeds can be generated in subfolders with custom titles/descriptions via feed.json
- **Conditional feed generation**: podcast.xml only generated if MP4 files present AND feed.json exists; articles.xml only if PDF/HTML files present AND feed.json exists
- **NO root feeds without feed.json**: Root-level feeds require explicit feed.json configuration
- **Index pages**: Auto-generated index.html in each folder with feed.json for easy feed discovery
- **Reflection support**: reflection.md files in folders provide reflection questions for summarization
- **Module prefixing**: Folder names automatically prepended to titles when appropriate
- **Markdown to HTML**: Feed descriptions are properly converted from markdown to HTML
- **Control character removal**: Invalid XML characters removed from extracted content
- **summarize command**: Works with both MP4 (transcripts) and PDF/HTML (extracted content) files
