# AGENTS.md - Local Feed Generator

Purpose: Transform education content (PDF/HTML/MP4) into RSS feeds.

Command: `uv run local-feed-generator <command>` from project root.

## Core
*   MP4: Requires `transcribe` → `summarize` → feed-eligible.
*   PDF/HTML: Requires `extract` → feed-eligible. Summarization optional.
*   Feeds: Generated only where `feed.json` exists. Requires feed-eligible files.

## Key Commands
*   `process` / `watch`: Process files batch/incrementally.
*   `transcribe`, `summarize`, `extract`, `feeds`, `validate`, `create-feed`.
*   Flags: `--data-dir`, `--file`, `--force`, `--dry-run`.

## Metadata Schema
Each file gets a `[filename].json` with:
```json
{
  "title": {"filename": "", "extracted": "", "generated": ""},
  "description": "", "summary": "", "content": "",
  "filetype": "mp4|pdf|html", "hash": "sha256...",
  "date": 1710000000.123, "publication_date": 1710000000.0,
  "duration": 45.5, "creation_time": "ISO-8601",
  "transcribe": {"status": "complete"},
  "summarize": {"status": "complete"},
  "extract": {"status": "complete"},
  "error": ""
}
```
Feed Eligibility: MP4 = transcribe & summarize complete. PDF/HTML = extract complete.

Title Rules: `generated` > `extracted` > `filename`. Parent folder name prepended.

## Config (Env Vars)
Required: `BASE_URL`, `DATA_DIR`, `LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`, `PROMPT_ARTICLE`, `PROMPT_PODCAST`, `PROMPT_TITLE`.

Optional: `PROMPT_ARTICLE_REFLECTION`, `PROMPT_PODCAST_REFLECTION`, `TRANSCRIPTION_ENDPOINT` (remote overrides local), `TRANSCRIPTION_MODEL`, `TRANSCRIPTION_DEVICE`, `TRANSCRIPTION_COMPUTE_TYPE`.

## Development
```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format . && uv run mypy .
```
Run commands from project root.
