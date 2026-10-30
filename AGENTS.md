# AGENTS.md - Focus Feed

Purpose: Transform education content (PDF/HTML/MP4) into RSS feeds.

Command: `uv run focus-feed <command>` from the workspace root (directory containing `src/`, `tests/`, and `data/`).

## Core
- MP4: Requires `transcribe` (remote-only) → `summarize` → feed-eligible.
- PDF/HTML: Requires `extract` → `summarize` → feed-eligible.
- Feeds: Generated only where `feed.json` exists. Requires feed-eligible files.
- File modification time stored in JSON and checked on each process; use `--force` or delete JSON to trigger update.

## Key Commands
- `process` / `watch`: Process files batch/incrementally.
- `transcribe`, `summarize`, `extract`, `title`, `feeds`, `validate`, `create-feed`, `cleanup`.
- Flags: `--force`, `--dry-run`.

## Metadata Schema
Each file gets a `[filename].json` alongside the source file with:
```json
{
  "title": {"filename": "", "extracted": "", "generated": ""},
  "description": "",
  "summary": "",
  "content": "",
  "filename": "lecture.mp4",
  "filetype": "mp4|pdf|html",
  "hash": "sha256...",
  "date": 1710000000.123,
  "mtime": 1710000000.0,
  "publication_date": 1710000000.0,
  "duration": 45.5,
  "creation_time": "ISO-8601",
  "size": 1234567,
  "transcribe": {"status": "complete"},
  "summarize": {"status": "complete"},
  "extract": {"status": "complete"},
  "generate_title": {"status": "complete"},
  "error": ""
}
```

Feed Eligibility: MP4 = transcribe & summarize complete. PDF/HTML = extract & summarize complete.

Title Rules: `generated` > `extracted` > `filename`. Parent folder name prepended.

Filepath is NOT stored; it is inferred from the JSON filename location.

## Config (Env Vars)
Required: `BASE_URL`, `LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`, `PROMPT_ARTICLE`, `PROMPT_PODCAST`, `PROMPT_TITLE`, `TRANSCRIPTION_ENDPOINT`, `TRANSCRIPTION_API_KEY`.

Optional: `PROMPT_ARTICLE_REFLECTION`, `PROMPT_PODCAST_REFLECTION`, `TRANSCRIPTION_REMOTE_MODEL`.

## Artifacts
- `artifacts/proposal.md` - Project proposal and scope.
- `artifacts/specs.md` - Functional/non-functional specs.
- `artifacts/design.md` - System architecture and data flow.
- `artifacts/tasks.md` - Implementation task list.

## Development
```bash
uv sync
uv run pytest
uv run ruff check
uv run ruff format
uv run mypy .
```

Run commands from workspace root.
