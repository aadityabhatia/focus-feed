"""Shared retry and exponential-backoff helpers for transient API failures."""

from __future__ import annotations

import random

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

#: Maximum number of retry attempts for transient API failures.
DEFAULT_MAX_RETRIES = 3

#: Base delay in seconds for the first retry; doubles with each attempt.
DEFAULT_BASE_DELAY = 1.0

#: Maximum jitter factor applied on top of the exponential delay (0 to 1).
DEFAULT_JITTER = 0.1

#: Exceptions indicating a transient failure that should be retried.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

#: Exceptions indicating a permanent failure that should not be retried.
NON_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
)


def calculate_backoff(base_delay: float, jitter: float, attempt: int) -> float:
    """Calculate an exponential backoff delay with jitter.

    Uses the formula ``base_delay * (2 ** attempt)`` plus a random jitter in
    ``[-delay * jitter, +delay * jitter]``, clamped to be non-negative.

    Args:
        base_delay: Base delay in seconds for the first retry.
        jitter: Maximum jitter fraction (0.0-1.0) applied to the delay.
        attempt: Zero-based attempt index.

    Returns:
        Delay in seconds before the next retry.
    """
    delay = base_delay * (2**attempt)
    jitter_amount = delay * jitter
    jitter_value = random.uniform(-jitter_amount, jitter_amount)
    return float(max(0.0, delay + jitter_value))


def is_retryable(exc: BaseException) -> bool:
    """Return whether *exc* represents a transient, retryable failure."""
    return isinstance(exc, RETRYABLE_ERRORS)


def is_non_retryable(exc: BaseException) -> bool:
    """Return whether *exc* represents a permanent, non-retryable failure."""
    return isinstance(exc, NON_RETRYABLE_ERRORS)
