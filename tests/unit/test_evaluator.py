"""
Unit tests for the Evaluator MCP Server.

Tests Step 2.3-2.5: Evaluator core logic and MCP integration.
"""

import pytest

from schemas import EvaluationResult, TrainingExample


class TestEvaluatorImports:
    """Test that evaluator modules can be imported."""

    def test_evaluator_module_imports(self):
        """Verify evaluator module can be imported."""
        from mcp_servers.evaluator import evaluator

        assert hasattr(evaluator, "evaluate_prompt")
        assert hasattr(evaluator, "get_default_model")
        assert hasattr(evaluator, "format_examples")

    def test_server_module_imports(self):
        """Verify server module can be imported."""
        from mcp_servers.evaluator import server

        assert hasattr(server, "mcp")
        assert hasattr(server, "evaluate")


class TestFormatExamples:
    """Test the format_examples helper function."""

    def test_format_single_example(self):
        """Test formatting a single example."""
        from mcp_servers.evaluator.evaluator import format_examples

        examples = [
            TrainingExample(input="hello", expected_output="Hello! How can I help?")
        ]
        result = format_examples(examples)

        assert "Example 1:" in result
        assert "hello" in result
        assert "Hello! How can I help?" in result

    def test_format_multiple_examples(self):
        """Test formatting multiple examples."""
        from mcp_servers.evaluator.evaluator import format_examples

        examples = [
            TrainingExample(input="input1", expected_output="output1"),
            TrainingExample(input="input2", expected_output="output2"),
        ]
        result = format_examples(examples)

        assert "Example 1:" in result
        assert "Example 2:" in result
        assert "input1" in result
        assert "input2" in result

    def test_format_examples_truncation(self):
        """Test that examples are truncated to max_examples."""
        from mcp_servers.evaluator.evaluator import format_examples

        examples = [
            TrainingExample(input=f"input{i}", expected_output=f"output{i}")
            for i in range(10)
        ]
        result = format_examples(examples, max_examples=3)

        assert "Example 1:" in result
        assert "Example 2:" in result
        assert "Example 3:" in result
        assert "Example 4:" not in result
        assert "and 7 more examples" in result

    def test_format_empty_examples(self):
        """Test formatting empty examples list."""
        from mcp_servers.evaluator.evaluator import format_examples

        result = format_examples([])
        assert "No training examples provided" in result


class TestGetDefaultModel:
    """Test the get_default_model function."""

    def test_returns_string(self):
        """Test that get_default_model returns a string."""
        from mcp_servers.evaluator.evaluator import get_default_model

        model = get_default_model()
        assert isinstance(model, str)
        assert len(model) > 0


class TestEvaluationResultSchema:
    """Test EvaluationResult integration with evaluator."""

    def test_evaluation_result_creation(self):
        """Test creating EvaluationResult with all fields."""
        result = EvaluationResult(
            prompt="Test prompt",
            score=0.85,
            passed=True,
            feedback="Good prompt overall",
            strengths=["Clear", "Concise"],
            weaknesses=["Missing examples"],
            suggestions=["Add examples"],
            clarity_score=0.9,
            task_alignment_score=0.8,
            example_quality_score=0.85,
        )

        assert result.score == 0.85
        assert result.passed is True
        assert len(result.strengths) == 2
        assert result.clarity_score == 0.9


# Integration tests that require API keys - marked to skip if not configured
@pytest.mark.asyncio
class TestEvaluatorIntegration:
    """Integration tests for evaluator (require API keys)."""

    @pytest.mark.skip(reason="Requires API keys - run manually")
    async def test_evaluate_prompt_basic(self):
        """Test basic evaluation works end-to-end."""
        from mcp_servers.evaluator.evaluator import evaluate_prompt

        examples = [
            TrainingExample(
                input="Write hello world in Python",
                expected_output="print('Hello, World!')",
            )
        ]

        result = await evaluate_prompt(
            prompt="You are a helpful coding assistant. Write clean, working code.",
            training_examples=examples,
        )

        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert len(result.feedback) > 0

    @pytest.mark.skip(reason="Requires API keys - run manually")
    async def test_evaluate_vague_prompt(self):
        """Test that vague prompts get lower scores."""
        from mcp_servers.evaluator.evaluator import evaluate_prompt

        examples = [
            TrainingExample(
                input="What is 2+2?",
                expected_output="4",
            )
        ]

        result = await evaluate_prompt(
            prompt="Do the thing",  # Intentionally vague
            training_examples=examples,
        )

        # Vague prompt should have weaknesses or lower score
        assert len(result.weaknesses) > 0 or result.score < 0.8
