"""Tests for the Summarizer service."""

import os
from unittest import mock

from openai import (
    APIConnectionError,
    OpenAI,
)
import pytest

from focus_feed.errors import SummarizationError
from focus_feed.services.summarizer import Summarizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_response():
    """Create a mock OpenAI chat completion response."""
    response = mock.MagicMock()
    response.choices[0].message.content = "Key takeaways: This is a test summary."
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response


@pytest.fixture
def mock_title_response():
    """Create a mock OpenAI chat completion response for title generation."""
    response = mock.MagicMock()
    response.choices[0].message.content = "Test Title"
    response.usage.prompt_tokens = 30
    response.usage.completion_tokens = 5
    return response


@pytest.fixture
def summarizer(mock_config, valid_env):
    """Create a Summarizer instance with a mocked OpenAI client."""
    with mock.patch.object(OpenAI, "__init__", return_value=None):
        s = Summarizer(config=mock_config)
        s._client = mock.MagicMock()
        yield s


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestSummarizerInit:
    """Tests for Summarizer initialization."""

    def test_initializes_with_config(self, mock_config):
        """Summarizer should accept a config parameter."""
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            s = Summarizer(config=mock_config)
            assert s._config is mock_config

    def test_uses_get_config_when_no_config(self):
        """Summarizer should use get_config() when no config is provided."""
        env = {
            "BASE_URL": "https://example.com",
            "LLM_ENDPOINT": "https://llm.example.com/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "PROMPT_ARTICLE": "Test article prompt",
            "PROMPT_ARTICLE_REFLECTION": "Test article reflection prompt",
            "PROMPT_PODCAST": "Test podcast prompt",
            "PROMPT_PODCAST_REFLECTION": "Test podcast reflection prompt",
            "PROMPT_TITLE": "Test title prompt",
            "TRANSCRIPTION_ENDPOINT": "https://transcribe.example.com/v1",
            "TRANSCRIPTION_API_KEY": "transcribe-test-key",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            # Reset the singleton
            import focus_feed.config as config_module

            config_module._config_instance = None

            with mock.patch.object(OpenAI, "__init__", return_value=None):
                s = Summarizer()
                assert s._config is not None
                assert s._config.llm_endpoint == "https://llm.example.com/v1"

            # Restore singleton for other tests
            config_module._config_instance = None

    def test_client_uses_endpoint_from_config(self, mock_config):
        """OpenAI client should be initialized with the configured endpoint."""
        with mock.patch.object(OpenAI, "__init__", return_value=None) as mock_init:
            Summarizer(config=mock_config)
            mock_init.assert_called_once_with(
                base_url="https://llm.example.com/v1",
                api_key="test-key",
            )

    def test_uses_injected_client_without_constructing_openai(self, mock_config):
        """An injected client should be used without constructing OpenAI."""
        client = mock.MagicMock()
        with mock.patch("focus_feed.services.summarizer.OpenAI") as openai:
            summarizer = Summarizer(config=mock_config, client=client)

        assert summarizer._client is client
        openai.assert_not_called()

    def test_client_init_failure_raises_summarization_error(self, mock_config):
        """Summarizer should raise SummarizationError if client init fails."""
        with (
            mock.patch.object(OpenAI, "__init__", side_effect=RuntimeError("connection refused")),
            pytest.raises(SummarizationError, match="Failed to initialize"),
        ):
            Summarizer(config=mock_config)


class TestSummarize:
    """Tests for the summarize() method."""

    def test_returns_summary_string(self, summarizer, mock_response):
        """summarize() should return a tuple of (summary, prompt_tokens, completion_tokens)."""
        summarizer._client.chat.completions.create.return_value = mock_response
        result = summarizer.summarize("This is a transcript about ILE courses.")
        assert result == ("Key takeaways: This is a test summary.", 100, 50)

    def test_api_call_parameters(self, summarizer, mock_response):
        """summarize() should pass correct system prompt, model, and user content."""
        summarizer._client.chat.completions.create.return_value = mock_response
        transcript = "This is a long transcript about military education."
        summarizer.summarize(transcript)

        call_kwargs = summarizer._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "Summarize key takeaways from this document."
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == transcript

    def test_filetype_mp4_uses_podcast_prompt(self, summarizer, mock_response):
        """summarize() with filetype='mp4' should use prompt_podcast as system prompt."""
        summarizer._client.chat.completions.create.return_value = mock_response
        summarizer.summarize("Test transcript", filetype="mp4")

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["content"] == "Summarize key takeaways from this podcast."

    def test_filetype_pdf_uses_article_prompt(self, summarizer, mock_response):
        """summarize() with filetype='pdf' should use prompt_article as system prompt."""
        summarizer._client.chat.completions.create.return_value = mock_response
        summarizer.summarize("Test transcript", filetype="pdf")

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["content"] == "Summarize key takeaways from this document."

    def test_filetype_html_uses_article_prompt(self, summarizer, mock_response):
        """summarize() with filetype='html' should use prompt_article as system prompt."""
        summarizer._client.chat.completions.create.return_value = mock_response
        summarizer.summarize("Test transcript", filetype="html")

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["content"] == "Summarize key takeaways from this document."

    def test_filetype_none_uses_article_prompt(self, summarizer, mock_response):
        """summarize() with filetype=None (default) should use prompt_article as system prompt."""
        summarizer._client.chat.completions.create.return_value = mock_response
        summarizer.summarize("Test transcript")

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["content"] == "Summarize key takeaways from this document."

    def test_filetype_mp4_with_reflection_uses_podcast_reflection_prompt(
        self, summarizer, mock_response
    ):
        """summarize() with filetype='mp4' and reflection questions uses podcast reflection."""
        summarizer._client.chat.completions.create.return_value = mock_response
        summarizer.summarize(
            "Test transcript",
            reflection_questions="What are the key points?",
            filetype="mp4",
        )

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["content"] == "Summarize key takeaways from this podcast with reflection."

    def test_filetype_pdf_with_reflection_uses_article_reflection_prompt(
        self, summarizer, mock_response
    ):
        """summarize() with filetype='pdf' and reflection questions uses article reflection."""
        summarizer._client.chat.completions.create.return_value = mock_response
        summarizer.summarize(
            "Test transcript",
            reflection_questions="What are the key points?",
            filetype="pdf",
        )

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert (
            system_msg["content"] == "Summarize key takeaways from this document with reflection."
        )

    def test_strips_whitespace_from_response(self, summarizer):
        """summarize() should strip leading/trailing whitespace from response."""
        response = mock.MagicMock()
        response.choices[0].message.content = "  Summary with spaces  "
        response.usage.prompt_tokens = 20
        response.usage.completion_tokens = 10
        summarizer._client.chat.completions.create.return_value = response
        result = summarizer.summarize("Test transcript")
        assert result == ("Summary with spaces", 20, 10)

    @pytest.mark.parametrize("transcript", ["", "   \n\t  "])
    def test_empty_or_whitespace_transcript_raises_error(self, summarizer, transcript):
        """summarize() should raise SummarizationError for empty or whitespace-only transcript."""
        with pytest.raises(SummarizationError, match="must not be empty"):
            summarizer.summarize(transcript)

    def test_none_content_raises_error(self, summarizer):
        """summarize() should raise SummarizationError when LLM returns None content."""
        response = mock.MagicMock()
        response.choices[0].message.content = None
        summarizer._client.chat.completions.create.return_value = response
        with pytest.raises(SummarizationError, match="empty response"):
            summarizer.summarize("Test transcript")


class TestGenerateTitle:
    """Tests for the generate_title() method."""

    def test_returns_title_string(self, summarizer, mock_title_response):
        """generate_title() should return a tuple of (title, prompt_tokens, completion_tokens)."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        result = summarizer.generate_title("Some content to generate a title from.")
        assert result == ("Test Title", 30, 5)

    def test_api_call_parameters(self, summarizer, mock_title_response):
        """generate_title() should use title-focused system prompt and pass content."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        content = "Content about military strategy and tactics."
        summarizer.generate_title(content)

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert "title" in system_msg["content"].lower()
        user_msg = messages[1]
        assert user_msg["role"] == "user"
        assert content in user_msg["content"]

    def test_truncates_content_to_4000_chars(self, summarizer, mock_title_response):
        """generate_title() should truncate content to ~4000 characters."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        long_content = "x" * 10000
        summarizer.generate_title(long_content)

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]
        # The content portion should be truncated to 4000 chars
        assert "x" * 4000 in user_msg["content"]
        assert "x" * 4001 not in user_msg["content"]

    @pytest.mark.parametrize(
        "filename,extracted,should_include",
        [
            ("lecture notes", "Lecture 1", True),
            ("notes", None, False),
            ("notes", "", False),
        ],
    )
    def test_filename_and_extracted_in_content(
        self, summarizer, mock_title_response, filename, extracted, should_include
    ):
        """generate_title() should include filename and extracted when provided."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        summarizer.generate_title("Some content", filename=filename, extracted=extracted)

        call_kwargs = summarizer._client.chat.completions.create.call_args
        user_msg = call_kwargs.kwargs["messages"][1]
        assert f"Filename: {filename}" in user_msg["content"]
        if should_include:
            assert "Extracted Title: Lecture 1" in user_msg["content"]
        else:
            assert "Extracted Title" not in user_msg["content"]

    def test_includes_filename_and_extracted_before_content(self, summarizer, mock_title_response):
        """generate_title() should include both filename and extracted title before content."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        summarizer.generate_title(
            "Some content here", filename="lecture notes", extracted="Lecture 1"
        )

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]
        user_content = user_msg["content"]
        # Filename and extracted title should appear before the content
        assert "Filename: lecture notes" in user_content
        assert "Extracted Title: Lecture 1" in user_content
        assert user_content.index("Filename:") < user_content.index("Some content here")
        assert user_content.index("Extracted Title:") < user_content.index("Some content here")

    def test_omits_content_label(self, summarizer, mock_title_response):
        """generate_title() should not include a 'Content:' label before the content."""
        summarizer._client.chat.completions.create.return_value = mock_title_response
        summarizer.generate_title("Some content here", filename="notes")

        call_kwargs = summarizer._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]
        assert "Content:\n" not in user_msg["content"]

    def test_empty_content_raises_error(self, summarizer):
        """generate_title() should raise SummarizationError for empty content."""
        with pytest.raises(SummarizationError, match="must not be empty"):
            summarizer.generate_title("")

    def test_whitespace_only_content_raises_error(self, summarizer):
        """generate_title() should raise SummarizationError for whitespace-only content."""
        with pytest.raises(SummarizationError, match="must not be empty"):
            summarizer.generate_title("   \n\t  ")


class TestModelParameter:
    """Tests for LLM model parameter handling."""

    def test_passes_model_from_config(self, summarizer, mock_response):
        """When llm_model is set, it should be passed to the API."""
        summarizer._client.chat.completions.create.return_value = mock_response
        summarizer.summarize("Test")
        call_kwargs = summarizer._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    def test_passes_none_model_when_not_configured(self, mock_config):
        """When llm_model is None, None should be passed (endpoint default)."""
        mock_config.llm_model = None
        with mock.patch.object(OpenAI, "__init__", return_value=None):
            s = Summarizer(config=mock_config)
            s._client = mock.MagicMock()
            mock_resp = mock.MagicMock()
            mock_resp.choices[0].message.content = "Summary"
            mock_resp.usage.prompt_tokens = 10
            mock_resp.usage.completion_tokens = 5
            s._client.chat.completions.create.return_value = mock_resp
            s.summarize("Test")
            call_kwargs = s._client.chat.completions.create.call_args
            assert call_kwargs.kwargs["model"] is None


class TestTokenTracking:
    """Tests for LLM token usage tracking."""

    @pytest.mark.parametrize(
        "method,response_fixture,expected_tokens",
        [
            ("summarize", "mock_response", (100, 50)),
            ("generate_title", "mock_title_response", (30, 5)),
        ],
    )
    def test_returns_token_counts(
        self, request, summarizer, method, response_fixture, expected_tokens
    ):
        """Methods should return (text, prompt_tokens, completion_tokens) tuples."""
        response = request.getfixturevalue(response_fixture)
        summarizer._client.chat.completions.create.return_value = response
        result = getattr(summarizer, method)("Test content")
        assert result[1] == expected_tokens[0]
        assert result[2] == expected_tokens[1]

    def test_summarize_returns_zero_tokens_when_usage_is_none(self, summarizer):
        """summarize() should return 0 tokens when usage is None."""
        response = mock.MagicMock()
        response.choices[0].message.content = "Summary text"
        response.usage = None
        summarizer._client.chat.completions.create.return_value = response
        result = summarizer.summarize("Test transcript")
        assert result == ("Summary text", 0, 0)

    def test_chat_completion_returns_llm_response(self, summarizer, mock_response):
        """_chat_completion() should return an LLmResponse NamedTuple."""
        from focus_feed.services.summarizer import LLmResponse

        summarizer._client.chat.completions.create.return_value = mock_response
        result = summarizer._chat_completion(
            system_prompt="Test prompt",
            user_content="Test content",
        )
        assert isinstance(result, LLmResponse)
        assert result.content == "Key takeaways: This is a test summary."
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50

    def test_token_counts_returned_on_retry(self, summarizer, mock_response):
        """Token counts should come from the successful (retry) response."""
        summarizer._client.chat.completions.create.side_effect = [
            APIConnectionError(message="fail", request=mock.MagicMock()),
            mock_response,
        ]
        with mock.patch("time.sleep"):
            result = summarizer.summarize("Test transcript")
        assert result == ("Key takeaways: This is a test summary.", 100, 50)
