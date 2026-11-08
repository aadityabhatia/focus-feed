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
