# Education Content Feed System - Proposal

## PyAV Audio and Video Capability Assessment

### Problem
The current project supports MP4 as its only media input and uses PyAV for selected media metadata. It is not clear whether PyAV can reliably cover additional audio/video inputs or provide audio extraction without requiring a separately installed FFmpeg executable. This uncertainty can lead to overly broad format promises or unnecessary dependencies.

### User Intent
Establish the practical PyAV `av>=11.0` capabilities for reading common audio and video files, extracting audio from video, converting audio, and reading media metadata. Use the assessment to guide future scope decisions. This request is for investigation and specification only, not implementation.

### Scope
- Explain PyAV's relationship to FFmpeg and why supported containers/codecs depend on the linked FFmpeg build.
- Assess common input examples including MP3, WAV, OGG, M4A, AAC, MP4, MKV, AVI, MOV, and WMV. Treat these as likely support examples, not guarantees independent of FFmpeg build configuration.
- Summarize common FFmpeg command-line methods for extracting audio and ffprobe's machine-readable metadata capabilities.
- Distinguish audio stream copying from decode/resample/re-encode conversion and explain the resulting quality tradeoffs.
- Identify metadata exposed through PyAV and cases where values may be missing or estimated.
- Assess transcription-friendly output choices and FFmpeg licensing considerations.
- Verify whether the project has PyAV and existing direct ffprobe/FFmpeg use.
- Record the recommended approach and operational caveats in `artifacts/specs.md`.

### Out of Scope
- Implementing support for new file extensions or changing transcription behavior.
- Promising an exhaustive or universal list of readable codecs and containers.
- Benchmarking PyAV against FFmpeg CLI or validating the project's transcription service formats.

### Success Criteria
- The specification explains that PyAV exposes FFmpeg capabilities rather than maintaining an independent, fixed format list.
- It distinguishes container demuxing, codec decoding, audio stream extraction, and transcoding.
- It documents metadata fields, limitations, recommended transcription audio handling, licensing caveats, and the direct-FFmpeg versus PyAV decision with authoritative documentation references.
- It leaves product support changes subject to a future decision on intended input scope.

### Research conclusion
- `av>=11.0` is already a required project dependency. PyAV is a binding to FFmpeg's libraries and is already used in the remote transcription service to read audio duration.
- The project also has code that calls the `ffprobe` executable with `subprocess` for duration, container format, bit rate, and creation time. Thus the current tree already uses both PyAV and the FFmpeg command-line tooling, but not the `ffmpeg` executable for audio extraction.
- PyAV can demux and decode selected audio streams and can encode output, so it can implement extraction/conversion without invoking the `ffmpeg` executable. Exact support follows the linked FFmpeg libraries and their enabled codecs/muxers.
- Recommendation: retain PyAV for in-process metadata and adopt it for extraction only if future input scope requires it and the endpoint can consume the resulting audio. Use the `ffmpeg` CLI when a concise, maintainable conversion pipeline or wider command-line format coverage is more valuable. Do not introduce conversion now without a product requirement and endpoint-format contract.
- For transcription, prefer avoiding unnecessary lossy conversion. If the transcription endpoint requires a normalized audio asset, PCM WAV at 16 kHz mono is a conservative interoperability option, but the endpoint's accepted formats and any duration/size constraints must be confirmed first.
- Current functional scope remains MP4 video transcription only. Supporting MKV, AVI, MOV, WMV or standalone audio files is not implied by this research.

## Problem
Education files (supported media formats such as MP3/M4A/WAV/WebM/AAC/FLAC/OGG/Opus/MP4/MPEG/MPGA/MKV/MOV/AVI/WMV, PDF, HTML, and Docling documents such as DOCX/DOC/PPTX/PPT/XLSX/XLS) in `data/` are disconnected. Users cannot consume them efficiently across devices via standard apps.

## User Intent
Make content accessible via private RSS feeds hosted on an existing internal web server:
- Podcast feed for Antennapod and podcast apps
- Articles feed for RSS readers
Per-folder feeds via `feed.json`, with an auto-generated `index.html` in every directory containing `feed.json`. Each index lists XML feeds in its subtree, and the root index lists feeds from the whole tree.

## Scope

### In Scope
- **Input:** Supported media files (mp3, m4a, wav, webm, aac, flac, ogg, opus, mp4, mpeg, mpga, mkv, mov, avi, wmv), PDF, HTML, and Docling documents (docx, doc, pptx, ppt, xlsx, xls) in `data/` directory (at workspace root) and its subfolders
- **Processing:** Batch and incremental watch mode
  - Media: transcribe to markdown (remote-only), LLM key takeaway summary
  - Documents (PDF, HTML, Docling): extract metadata + content; summarization required for feed eligibility
  - Track steps: `transcribe`, `summarize`, `extract`, `generate_title`
  - File modification time stored in JSON and checked on each process
- **Metadata:** `<filename>.json` per file with title dict `{filename,extracted,generated}`, description, summary, content, filename, filetype (media formats, PDF, HTML, and Docling documents `docx/doc/pptx/ppt/xlsx/xls`), `date`/`publication_date` (ms), `mtime`, `size`, `duration`, `creation_time`, `hash`, step maps `transcribe/summarize/extract/generate_title`
  - filepath is NOT stored; inferred from JSON filename location
  - Media: description blank, summary = takeaways, content = transcript
  - Documents (PDF, HTML, Docling): description = extracted metadata, content = plaintext, summary null unless summarized
  - SHA256 hash computed during first extraction
- **Feeds:** `podcast.xml` for supported media files, `articles.xml` for documents (PDF, HTML, Docling), per-folder only when `feed.json` exists
- **Features:** `reflection.md` injection, auto folder-name title prefix, markdown→HTML conversion, control character sanitization

### Out of Scope
- AI agents for metadata gap filling
- Cloud services by default; remote transcription required
- Local transcription (faster-whisper removed)
- Web server deployment / public distribution

## Success Criteria
- Content consumable via standard RSS/podcast apps
- Automatic transcription (remote-only) + summarization for supported media files, extraction for documents (PDF, HTML, Docling)
- Local-first processing for extraction/feed generation, remote-only for transcription
- Incremental skips completed files unless mtime changed
- Warnings for orphaned JSONs and special characters
- Millisecond-precision timestamps, feed eligibility from step statuses
- An `index.html` is generated in every directory containing `feed.json`; each lists XML feeds in its subtree, and the root index lists all feeds in the tree
