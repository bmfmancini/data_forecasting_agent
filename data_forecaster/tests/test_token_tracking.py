"""Tests for LLM token usage tracking utilities."""

import sys
import os
from types import SimpleNamespace

import pytest
from langchain_core.prompts import ChatPromptTemplate

# Add the backend directory to the path
backend_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "backend"
)
sys.path.insert(0, backend_dir)

from utils.token_tracking import extract_token_usage, estimate_input_text


class TestExtractTokenUsage:
    """Test extract_token_usage function."""

    def test_native_usage_metadata_returned(self):
        """Test that native usage_metadata is returned when present."""
        response = SimpleNamespace(
            content="Hello world",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        result = extract_token_usage(response)
        assert result == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    def test_heuristic_fallback_no_metadata(self):
        """Test heuristic fallback when usage_metadata is absent."""
        content = "a" * 400  # 400 chars -> 100 tokens
        response = SimpleNamespace(content=content, usage_metadata=None)
        result = extract_token_usage(response, input_text="b" * 200)
        assert result["output_tokens"] == 100
        assert result["input_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_heuristic_fallback_no_input_text(self):
        """Test heuristic fallback without input_text."""
        content = "a" * 80  # 80 chars -> 20 tokens
        response = SimpleNamespace(content=content, usage_metadata=None)
        result = extract_token_usage(response)
        assert result["output_tokens"] == 20
        assert result["input_tokens"] == 0
        assert result["total_tokens"] == 20

    def test_empty_content_no_metadata(self):
        """Test that empty content with no metadata returns zeros."""
        response = SimpleNamespace(content="", usage_metadata=None)
        result = extract_token_usage(response)
        assert result["output_tokens"] == 0
        assert result["input_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_none_response_does_not_raise(self):
        """Test that None response does not raise an exception."""
        result = extract_token_usage(None)
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "total_tokens" in result

    def test_response_without_content_attr(self):
        """Test that a response without content attribute does not raise."""
        response = SimpleNamespace(usage_metadata=None)
        result = extract_token_usage(response)
        assert result["output_tokens"] == 0

    def test_non_string_content(self):
        """Test that non-string content is handled gracefully."""
        response = SimpleNamespace(content=12345, usage_metadata=None)
        result = extract_token_usage(response)
        # Should convert to string and estimate
        assert result["output_tokens"] >= 0

    def test_partial_usage_metadata(self):
        """Test partial usage_metadata with only total_tokens."""
        response = SimpleNamespace(
            content="test",
            usage_metadata={"total_tokens": 200, "input_tokens": 150},
        )
        result = extract_token_usage(response)
        assert result["total_tokens"] == 200
        assert result["input_tokens"] == 150
        assert result["output_tokens"] == 0

    def test_returns_three_keys(self):
        """Test that the result always has the three required keys."""
        response = SimpleNamespace(content="test", usage_metadata=None)
        result = extract_token_usage(response)
        assert set(result.keys()) == {
            "input_tokens",
            "output_tokens",
            "total_tokens",
        }


class TestEstimateInputText:
    """Test estimate_input_text function."""

    def test_formats_template_correctly(self):
        """Test that the template is formatted with inputs."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "Analyze this: {data}"),
        ])
        result = estimate_input_text(prompt, {"data": "test data"})
        assert "test data" in result
        assert "helpful assistant" in result

    def test_returns_string(self):
        """Test that the result is a string."""
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Hello {name}"),
        ])
        result = estimate_input_text(prompt, {"name": "World"})
        assert isinstance(result, str)

    def test_format_error_returns_empty_string(self):
        """Test that formatting errors return an empty string."""
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Hello {missing_key}"),
        ])
        result = estimate_input_text(prompt, {"wrong_key": "value"})
        assert result == ""

    def test_none_inputs_handled(self):
        """Test that None inputs don't raise."""
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Hello"),
        ])
        result = estimate_input_text(prompt, {})
        assert isinstance(result, str)