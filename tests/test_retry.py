"""Tests for shared retry helpers."""

from unittest import mock

from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
import pytest

from focus_feed.retry import (
    DEFAULT_BASE_DELAY,
    DEFAULT_JITTER,
    DEFAULT_MAX_RETRIES,
    calculate_backoff,
    is_non_retryable,
    is_retryable,
)


@pytest.mark.parametrize("attempt,expected", [(0, 1.0), (1, 2.0), (2, 4.0), (3, 8.0)])
def test_calculate_backoff_without_jitter(attempt: int, expected: float) -> None:
    """Zero jitter should return exact powers of two from the base delay."""
    assert calculate_backoff(1.0, 0.0, attempt) == expected


def test_calculate_backoff_is_non_negative_for_many_attempts() -> None:
    """Backoff is clamped non-negative across a broad range of attempts."""
    with mock.patch("focus_feed.retry.random.uniform", return_value=-1000000.0):
        for attempt in range(100):
            assert calculate_backoff(1.0, 0.1, attempt) >= 0.0


@pytest.mark.parametrize(
    "exc",
    [
        APIConnectionError(message="connection failed", request=mock.MagicMock()),
        RateLimitError(message="rate limited", response=mock.MagicMock(), body=None),
    ],
)
def test_is_retryable_classifies_transient_errors(exc: Exception) -> None:
    assert is_retryable(exc)
    assert not is_non_retryable(exc)


@pytest.mark.parametrize(
    "exc",
    [
        AuthenticationError(message="unauthorized", response=mock.MagicMock(), body=None),
        BadRequestError(message="bad request", response=mock.MagicMock(), body=None),
    ],
)
def test_is_non_retryable_classifies_permanent_errors(exc: Exception) -> None:
    assert is_non_retryable(exc)
    assert not is_retryable(exc)


def test_default_retry_constants_match_existing_values() -> None:
    assert DEFAULT_MAX_RETRIES == 3
    assert DEFAULT_BASE_DELAY == 1.0
    assert DEFAULT_JITTER == 0.1
