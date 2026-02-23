"""Unit tests for meta-prompt optimizer."""

import pytest


class TestOptimizerImports:
    """Test that optimizer modules can be imported."""

    def test_optimizer_module_imports(self):
        """Test core optimizer module imports."""
        from mcp_servers.meta_prompt_optimizer.optimizer import (
            OptimizationResult,
            OptimizedCandidate,
            optimize,
        )

        assert optimize is not None
        assert OptimizedCandidate is not None
        assert OptimizationResult is not None

    def test_server_module_imports(self):
        """Test MCP server module imports."""
        from mcp_servers.meta_prompt_optimizer.server import (
            generate_candidates,
            mcp,
        )

        assert mcp is not None
        assert generate_candidates is not None


class TestOptimizedCandidate:
    """Test OptimizedCandidate dataclass."""

    def test_creation(self):
        """Test creating an OptimizedCandidate."""
        from mcp_servers.meta_prompt_optimizer.optimizer import OptimizedCandidate

        candidate = OptimizedCandidate(
            prompt="Test prompt",
            strategy="Added format",
            addressed_weaknesses=["Missing format"],
        )
        assert candidate.prompt == "Test prompt"
        assert candidate.strategy == "Added format"
        assert candidate.addressed_weaknesses == ["Missing format"]


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_creation(self):
        """Test creating an OptimizationResult."""
        from mcp_servers.meta_prompt_optimizer.optimizer import (
            OptimizationResult,
            OptimizedCandidate,
        )

        candidate = OptimizedCandidate(
            prompt="Improved",
            strategy="Better",
            addressed_weaknesses=["Issue"],
        )
        result = OptimizationResult(
            candidates=[candidate],
            original_prompt="Original",
        )
        assert result.original_prompt == "Original"
        assert len(result.candidates) == 1
        assert result.candidates[0].prompt == "Improved"


class TestOptimizerIntegration:
    """Integration tests requiring API keys."""

    @pytest.mark.skip(reason="Requires API keys - run manually")
    @pytest.mark.asyncio
    async def test_optimize_basic(self):
        """Test basic optimization with feedback."""
        from mcp_servers.meta_prompt_optimizer.optimizer import optimize

        result = await optimize(
            prompt="You are a helpful assistant.",
            feedback={
                "score": 0.5,
                "weaknesses": ["Too vague", "No format"],
                "suggestions": ["Add specifics", "Define output format"],
                "strengths": ["Clear role"],
            },
            num_candidates=2,
        )

        assert result.original_prompt == "You are a helpful assistant."
        assert len(result.candidates) >= 1
        for c in result.candidates:
            assert len(c.prompt) > 0
            assert len(c.strategy) > 0

    @pytest.mark.skip(reason="Requires API keys - run manually")
    @pytest.mark.asyncio
    async def test_optimize_with_cross_pollination(self):
        """Test optimization with cross-pollination prompts."""
        from mcp_servers.meta_prompt_optimizer.optimizer import optimize

        result = await optimize(
            prompt="Help me code.",
            feedback={
                "score": 0.3,
                "weaknesses": ["No role", "No format", "Too short"],
                "suggestions": ["Add role", "Add structure"],
                "strengths": [],
            },
            num_candidates=2,
            cross_pollination_prompts=[
                "You are an expert Python developer. Given a task, write clean code with comments.",
            ],
        )

        assert len(result.candidates) >= 1
