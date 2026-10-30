# Education Content Feed System - Specifications

See proposal.md for problem, intent and scope.

## Functional Requirements

### Input & Processing
- Accept one or more files or directories as positional arguments for each subcommand
- Files must be actual file paths (absolute or relative to current directory)
- For each directory provided, process all relevant files within that directory recursively
- Skip files with JSON where all applicable steps complete AND file modification time matches JSON mtime. Treat missing JSON as new.
- Warn on orphaned JSON and special characters in filenames. Preserve filenames.
- Supported types: PDF, HTML, MP4
- Check file modification time on each process; store mtime in JSON; use to verify if file was modified after processing

### Metadata
- Generate `<filename>.json` alongside source file.
- Fields: `title:{filename,extracted,generated}`, `description`, `summary`, `content`, `filename`, `filetype`, `date` ms, `publication_date`, `mtime`, `duration`, `creation_time`, `size`, `hash`, step maps `transcribe/summarize/extract/generate_title`, `error`.
- filepath is NOT stored; it is inferred from the JSON filename location.
- Title precedence: generated > extracted > filename. Auto-prepend parent folder name if missing. Sanitize filename key.
- Remove control chars 0x00-0x1f except tab/newline/CR and 0x7f from extracted PDF/HTML.
- Compute SHA256 during first extraction or before feed generation (whichever comes first).
- MP4: description blank, summary = key takeaways, content = transcript
- PDF/HTML: description = extracted metadata, content = plaintext, summary null unless summarized

### Feed Eligibility
- MP4: transcribe.complete AND summarize.complete
- PDF/HTML: extract.complete AND summarize.complete
- Ineligible items excluded from feeds
- Nested feeds: each feed includes all content from its entire subtree (inner feeds are subsets of outer feeds)

### Audio
- Transcribe MP4 to markdown via remote OpenAI-compatible API only (no local transcription). TRANSCRIPTION_ENDPOINT required.
- Summarize transcript via LLM, feeding entire transcript.
- Extract duration/creation_time via PyAV (av library).
- Record remote API errors in JSON.

### Documents
- Extract metadata + content from PDF/HTML.
- Summarization via explicit `summarize` command (required for feed eligibility).
- Use `publication_date` for feed pubDate if present.

### Reflection
- Check for `reflection.md` in source folder. Prepend its content to summarization prompt.
- Prompts: MP4 `PROMPT_PODCAST_REFLECTION` fallback `PROMPT_PODCAST`; PDF/HTML `PROMPT_ARTICLE_REFLECTION` fallback `PROMPT_ARTICLE`.

### Feeds
- Generate `podcast.xml` and `articles.xml` per folder with `feed.json`.
- Items include title, description, link `BASE_URL+relative_filepath`, pubDate, guid from hash.
- Sort by folder then filename. Markdown→HTML with XSS protection.
- Conditional generation: only if feed.json exists and eligible files present.
- Nested feeds: each feed includes all content from its entire subtree (inner feeds are subsets of outer feeds).
- Root `index.html` lists all feeds. Root feeds require feed.json in data root.
- Ensure hash present before feed generation.

### CLI
Commands: `process`, `watch`, `transcribe`, `summarize`, `extract`, `title`, `feeds`, `validate`, `create-feed`, `cleanup`
Accept one or more files or directories as positional arguments.
Flags: `--config-file`, `--force`, `--dry-run` for transcribe/summarize/extract.
`--force` re-runs step regardless of completion status or mtime.

## Non-Functional
- Local processing by default for extraction and feed generation; transcription is remote-only
- Open-weight models supported via OpenAI-compatible endpoints for transcription and summarization
- GPU acceleration when available (for PyAV metadata extraction)
- Summaries = concise key takeaways, transcripts = markdown
- Watch mode debounced with periodic rescan fallback

## Constraints
- Remote-only transcription; no local whisper
- Feeds only where feed.json exists
- Directory structure preserved
- Special characters preserved with warnings only
- Unix epoch timestamps with millisecond precision
- Title dict with filename/extracted/generated
- Markdown library (>=3.5.0) required for markdown to HTML conversion with XSS protection
- Remote transcription: vLLM server requires `VLLM_MAX_AUDIO_DECODE_DURATION_S` configuration for longer audio files

## Risks
- Transcription quality varies → use established models
- LLM summarization inconsistent → tested prompts, configurable
- File watching may miss events → periodic rescan fallback
- Special characters → escape output, warn
- Control chars → removed during extraction

## Metadata Example
```json
{
  "title":{"filename":"lecture","extracted":"Lecture 1","generated":"..."},
  "description":"",
  "summary":"Key takeaways...",
  "content":"Transcript or text...",
  "filename":"lecture.mp4",
  "filetype":"mp4",
  "date":1710000000.123,
  "publication_date":1710000000.0,
  "mtime":1710000000.0,
  "duration":45.5,
  "creation_time":"2024-03-10T10:00:00Z",
  "size":1234567,
  "hash":"sha256...",
  "transcribe":{"status":"complete"},
  "summarize":{"status":"complete"},
  "extract":null,
  "generate_title":{"status":"complete"},
  "error":""
}
```

## Environment
Required: BASE_URL, LLM_ENDPOINT, LLM_API_KEY, LLM_MODEL, PROMPT_ARTICLE, PROMPT_PODCAST, PROMPT_TITLE, TRANSCRIPTION_ENDPOINT, TRANSCRIPTION_API_KEY
Optional: PROMPT_ARTICLE_REFLECTION, PROMPT_PODCAST_REFLECTION, TRANSCRIPTION_REMOTE_MODEL
