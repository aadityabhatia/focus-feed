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

## Parallel Processing
- Batch and watch mode process independent files concurrently with a bounded in-process thread pool.
- Steps within a single file remain sequential; JSON metadata and step ordering are unchanged.
- New flags: `--max-workers` (default 4) and `--max-summarize-jobs` (default 4); env var `MAX_SUMMARIZE_CONCURRENCY` also configures the summarization cap.
- New module `src/focus_feed/concurrency.py` provides `ConcurrencyOrchestrator` and `InFlightRegistry`; watch mode uses an in-flight guard to avoid duplicate processing.
- Feed generation runs only after all workers settle.
- Concurrency tests live in `tests/test_concurrency.py`; the test command remains `uv run pytest`.

## Metadata Schema
Each file gets a `[filename].json` alongside the source file with:
```json
{
  "title": {"filename": "", "extracted": "", "generated": ""},
  "description": "",
  "summary": "",
  "content": "",
  "filename": "lecture.mp4",
  "filetype": "mp4|pdf|html|docx|doc|pptx|ppt|xlsx|xls (also supported audio/video extensions)",
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

Feed Eligibility: Media = transcribe & summarize complete. PDF/HTML/Docling documents = extract & summarize complete.

Title Rules: `generated` > `extracted` > `filename`. Parent folder name prepended unless the filename begins with the folder name plus an extended code, in which case that code is used as the prefix (e.g. `H103RA` in folder `H103`).

Filepath is NOT stored; it is inferred from the JSON filename location.

## Config (Env Vars)
Required: `BASE_URL`, `LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`, `PROMPT_ARTICLE`, `PROMPT_PODCAST`, `PROMPT_TITLE`, `TRANSCRIPTION_ENDPOINT`, `TRANSCRIPTION_API_KEY`.

Optional: `DOCLING_API`, `PROMPT_ARTICLE_REFLECTION`, `PROMPT_PODCAST_REFLECTION`, `TRANSCRIPTION_REMOTE_MODEL`, `MAX_WORKERS`, `MAX_SUMMARIZE_CONCURRENCY`.

Use `example.env` as the template when creating `.env`.

## Artifacts
- `artifacts/proposal.md` - Project proposal and scope.
- `artifacts/specs.md` - Functional/non-functional specs.
- `artifacts/design.md` - System architecture and data flow.

## Development
```bash
uv sync
uv run pytest
uv run ruff check
uv run ruff format
uv run mypy .
```

Run commands from workspace root.
