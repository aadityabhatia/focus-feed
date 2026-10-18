# Education Content Feed System - Specifications

See proposal.md for problem, intent and scope.

## Functional Requirements

### Input & Processing
- Accept one or more files or directories as positional arguments for each subcommand
- Files must be actual file paths (absolute or relative to current directory)
- For each directory provided, process all relevant files within that directory recursively
- Skip files with JSON where all applicable steps complete. Treat missing JSON as new.
- Warn on orphaned JSON and special characters in filenames. Preserve filenames.
- Supported types: PDF, HTML, MP4

### Metadata
- Generate `<filename>.json` alongside source file.
- Fields: `title:{filename,extracted,generated}`, `description`, `summary`, `content`, `filename`, `filepath`, `filetype`, `date` ms, `publication_date`, `duration`, `creation_time`, `size`, `hash`, step maps `transcribe/summarize/extract`, `error`.
- Title precedence: filename < extracted < generated. Auto-prepend parent folder name if missing. Sanitize filename key.
- Remove control chars 0x00-0x1f except tab/newline/CR and 0x7f from extracted PDF/HTML.
- Compute SHA256 on first process or before feed generation.
- MP4: description blank, summary = key takeaways, content = transcript
- PDF/HTML: description = extracted metadata, content = plaintext, summary null unless summarized

### Feed Eligibility
- MP4: transcribe.complete AND summarize.complete
- PDF/HTML: extract.complete
- Ineligible items excluded from feeds

### Audio
- Transcribe MP4 to markdown via local faster-whisper or remote OpenAI-compatible API if `TRANSCRIPTION_ENDPOINT` set.
- Summarize transcript via LLM, feeding entire transcript.
- Extract duration/creation_time via ffprobe.
- Record remote API errors in JSON.

### Documents
- Extract metadata + content from PDF/HTML.
- Summarization optional via explicit `summarize` command.
- Use `publication_date` for feed pubDate if present.

### Reflection
- Check for `reflection.md` in source folder. Prepend its content to summarization prompt.
- Prompts: MP4 `PROMPT_PODCAST_REFLECTION` fallback `PROMPT_PODCAST`; PDF/HTML `PROMPT_ARTICLE_REFLECTION` fallback `PROMPT_ARTICLE`.

### Feeds
- Generate `podcast.xml` and `articles.xml` per folder with `feed.json`.
- Items include title, description, link `BASE_URL+filepath`, pubDate, guid from hash.
- Sort by folder then filename. Markdown→HTML with XSS protection.
- Conditional generation: only if feed.json exists and eligible files present.
- Root `index.html` lists all feeds. Root feeds require feed.json in data root.
- Ensure hash present before feed generation.

### CLI
Commands: `process`, `watch`, `transcribe`, `summarize`, `extract`, `feeds`, `validate`, `create-feed`
Accept one or more files or directories as positional arguments.
Flags: `--config-file`, `--force`, `--dry-run` for transcribe/summarize/extract.
`--force` re-runs step.

## Non-Functional
- Local processing by default, no cloud dependency
- Open-weight transcription, OpenAI-compatible LLM for summarization
- GPU acceleration when available
- Summaries = concise key takeaways, transcripts = markdown
- Watch mode debounced with periodic rescan fallback

## Constraints
- Local first; remote transcription optional
- Feeds only where feed.json exists
- Directory structure preserved
- Special characters preserved with warnings only
- Unix epoch ms timestamps
- Title dict with filename/extracted/generated
- Markdown library required for markdown to HTML conversion with XSS protection

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
  "filetype":"mp4",
  "date":1710000000.123,
  "hash":"...",
  "transcribe":{"status":"complete"},
  "summarize":{"status":"complete"},
  "extract":null
}
```

## Environment
Required: BASE_URL, LLM_ENDPOINT, LLM_API_KEY, LLM_MODEL, PROMPT_ARTICLE, PROMPT_PODCAST, PROMPT_TITLE
Optional: PROMPT_ARTICLE_REFLECTION, PROMPT_PODCAST_REFLECTION, TRANSCRIPTION_ENDPOINT, TRANSCRIPTION_API_KEY, TRANSCRIPTION_MODEL, TRANSCRIPTION_DEVICE, TRANSCRIPTION_COMPUTE_TYPE, TRANSCRIPTION_REMOTE_MODEL
