"""Unit tests for few-shot optimizer."""

import pytest

from schemas import TrainingExample


class TestOptimizerImports:
    """Test that optimizer modules can be imported."""

    def test_optimizer_module_imports(self):
        """Test core optimizer module imports."""
        from mcp_servers.few_shot_optimizer.optimizer import (
            ScoredExample,
            SelectionResult,
            format_examples,
            select_examples,
        )
        assert select_examples is not None
        assert format_examples is not None
        assert ScoredExample is not None
        assert SelectionResult is not None

    def test_server_module_imports(self):
        """Test MCP server module imports."""
        from mcp_servers.few_shot_optimizer.server import (
            format_example_set,
            mcp,
            select_optimal_examples,
        )
        assert mcp is not None
        assert select_optimal_examples is not None
        assert format_example_set is not None


class TestFormatExamples:
    """Test example formatting."""

    def test_format_markdown(self):
        """Test markdown formatting."""
        from mcp_servers.few_shot_optimizer.optimizer import format_examples
        examples = [
            TrainingExample(input="hello", expected_output="world"),
        ]
        result = format_examples(examples, "markdown")
        assert "**Example:**" in result
        assert "hello" in result
        assert "world" in result

    def test_format_xml(self):
        """Test XML formatting."""
        from mcp_servers.few_shot_optimizer.optimizer import format_examples
        examples = [
            TrainingExample(input="test", expected_output="result"),
        ]
        result = format_examples(examples, "xml")
        assert "<example>" in result
        assert "<input>" in result
        assert "<output>" in result

    def test_format_numbered(self):
        """Test numbered formatting."""
        from mcp_servers.few_shot_optimizer.optimizer import format_examples
        examples = [
            TrainingExample(input="a", expected_output="b"),
            TrainingExample(input="c", expected_output="d"),
        ]
        result = format_examples(examples, "numbered")
        assert "1." in result
        assert "2." in result

    def test_format_chat(self):
        """Test chat formatting."""
        from mcp_servers.few_shot_optimizer.optimizer import format_examples
        examples = [
            TrainingExample(input="question", expected_output="answer"),
        ]
        result = format_examples(examples, "chat")
        assert "User:" in result
        assert "Assistant:" in result


class TestComplexityEstimation:
    """Test complexity scoring."""

    def test_simple_example_low_complexity(self):
        """Simple examples should have low complexity."""
        from mcp_servers.few_shot_optimizer.optimizer import _estimate_complexity
        simple = TrainingExample(input="hi", expected_output="hello")
        score = _estimate_complexity(simple)
        assert score < 0.3

    def test_complex_example_high_complexity(self):
        """Complex examples should have high complexity."""
        from mcp_servers.few_shot_optimizer.optimizer import _estimate_complexity
        complex_ex = TrainingExample(
            input="def calculate(x, y):\n    return x + y\n\nresult = calculate(1, 2)",
            expected_output="The function adds two numbers.\nOutput: 3",
        )
        score = _estimate_complexity(complex_ex)
        assert score > 0.3


class TestDiversityBonus:
    """Test diversity calculation."""

    def test_first_selection_neutral(self):
        """First selection should have neutral diversity bonus."""
        from mcp_servers.few_shot_optimizer.optimizer import _compute_diversity_bonus
        candidate = TrainingExample(input="test", expected_output="result")
        bonus = _compute_diversity_bonus(candidate, [])
        assert bonus == 0.5

    def test_similar_example_low_bonus(self):
        """Similar examples should have low diversity bonus."""
        from mcp_servers.few_shot_optimizer.optimizer import _compute_diversity_bonus
        candidate = TrainingExample(input="hello world", expected_output="hi")
        selected = [TrainingExample(input="hello there", expected_output="hey")]
        bonus = _compute_diversity_bonus(candidate, selected)
        assert bonus < 0.5

    def test_different_example_high_bonus(self):
        """Different examples should have high diversity bonus."""
        from mcp_servers.few_shot_optimizer.optimizer import _compute_diversity_bonus
        candidate = TrainingExample(input="x = 1\ny = 2", expected_output="code")
        selected = [TrainingExample(input="hello", expected_output="greeting")]
        bonus = _compute_diversity_bonus(candidate, selected)
        assert bonus > 0.5


class TestSelectExamplesIntegration:
    """Integration tests requiring API keys."""

    @pytest.mark.skip(reason="Requires API keys - run manually")
    @pytest.mark.asyncio
    async def test_select_examples_basic(self):
        """Test basic example selection."""
        from mcp_servers.few_shot_optimizer.optimizer import select_examples

        pool = [
            TrainingExample(input="reverse 'hello'", expected_output="'olleh'"),
            TrainingExample(input="reverse 'world'", expected_output="'dlrow'"),
            TrainingExample(input="reverse ''", expected_output="''"),
            TrainingExample(input="reverse 'a'", expected_output="'a'"),
            TrainingExample(input="reverse 'ab cd'", expected_output="'dc ba'"),
        ]

        result = await select_examples(
            prompt="You are a string reversal assistant.",
            example_pool=pool,
            num_examples=3,
            strategy="balanced",
        )

        assert len(result.selected_examples) == 3
        assert result.prompt_with_examples is not None
        assert result.selection_reasoning is not None

    @pytest.mark.skip(reason="Requires API keys - run manually")
    @pytest.mark.asyncio
    async def test_select_examples_diverse(self):
        """Test diverse strategy selection."""
        from mcp_servers.few_shot_optimizer.optimizer import select_examples

        pool = [
            TrainingExample(input="add 1 + 2", expected_output="3"),
            TrainingExample(input="add 3 + 4", expected_output="7"),
            TrainingExample(input="subtract 5 - 3", expected_output="2"),
            TrainingExample(input="multiply 2 * 3", expected_output="6"),
        ]

        result = await select_examples(
            prompt="You are a calculator.",
            example_pool=pool,
            num_examples=2,
            strategy="diverse",
        )

        assert len(result.selected_examples) == 2
        # Should pick different operations due to diversity
