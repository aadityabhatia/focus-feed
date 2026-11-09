"""Shared Click option decorators for CLI commands."""

from __future__ import annotations

from typing import Any

import click


def _config_file_option(func: Any) -> Any:
    """Add the ``--config-file`` optional option to a Click command."""
    return click.option(
        "--config-file",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="Path to a .env-style configuration file.",
    )(func)


def _force_option(func: Any) -> Any:
    """Add the ``--force`` flag to a Click command.

    When specified, the step is run even if it appears to already be
    complete (based on the ``status`` dict).
    """
    return click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Run the step even if it appears to already be complete.",
    )(func)


def _dry_run_option(func: Any) -> Any:
    """Add the ``--dry-run`` flag to a Click command."""
    return click.option(
        "--dry-run",
        is_flag=True,
        default=False,
        help="Show what would be processed without making any changes.",
    )(func)


def _max_workers_option(func: Any) -> Any:
    """Add the ``--max-workers`` option to a Click command."""
    return click.option(
        "--max-workers",
        type=click.IntRange(min=1),
        default=None,
        help="Maximum number of files processed concurrently (default: 4).",
    )(func)


def _max_summarize_jobs_option(func: Any) -> Any:
    """Add the ``--max-summarize-jobs`` option to a Click command."""
    return click.option(
        "--max-summarize-jobs",
        type=click.IntRange(min=1),
        default=None,
        help="Maximum number of concurrent summarization calls (default: 4).",
    )(func)
