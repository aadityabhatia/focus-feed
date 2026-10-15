# Education Content Feed System - Proposal

## Problem Statement

Currently, education content exists as a disorganized collection of files (PDFs, HTML documents, and MP4 audio recordings) with no coherent way to consume them. There is no unified method to access, browse, or track this content across devices.

## User Intent

The primary goal is to **make education content readily available for personal consumption** through mobile applications:
- **Podcast feed** for Antennapod (audio content)
- **Articles feed** for RSS reader apps (text content)

These feeds will be:
- Hosted on an existing internal web server
- Accessible only to the user
- Placed in the `data` directory at the root level

## Scope

### In Scope
- **Input files**: PDF, HTML, and MP4 audio files in the `data` directory (can be in subfolders)
- **Processing modes**: Batch processing and incremental processing (watching for new files or JSON updates)
- **Audio processing**: Transcribe MP4 files to markdown, generate key takeaway summaries using a local OpenAI-compatible LLM
- **Metadata**: Generate JSON metadata for each input file with `.json` extension (e.g., `file.mp4` → `file.mp4.json`)
- **Feed generation**: Create two distinct feeds in the `data` directory:
  1. **Podcast RSS/Atom feed** for audio content (MP4 files with summaries as descriptions)
  2. **Articles RSS/Atom feed** for text content (PDF/HTML files)
- **Configuration**: System prompt for LLM summarization stored in a config file for easy updates

### Out of Scope
- AI agent for filling metadata gaps (deferred to future iteration)
- Cloud-based services (all processing must be local)
- Web server deployment (existing server will serve the `data` directory)
- Public distribution

## Success Criteria

- All education content in the `data` directory is accessible through standard RSS/Atom feeds
- Audio content is transcribed and summarized automatically using local models
- Content can be consumed through Antennapod and any RSS reader app
- No dependency on external/cloud services
- System processes only new or updated files (skips files with existing JSON metadata)
- Orphaned JSON files and special characters in filenames generate warnings
