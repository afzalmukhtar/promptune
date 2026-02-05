"""Unit tests for beam orchestrator."""

import pytest

from schemas import TrainingExample


class TestOrchestratorImports:
    """Test that orchestrator modules can be imported."""

    def test_orchestrator_module_imports(self):
        """Test core orchestrator module imports."""
        from mcp_servers.beam_orchestrator.orchestrator import (
            BeamConfig,
            IterationResult,
            OptimizationResult,
            get_default_model,
            optimize_beam,
            step,
        )
        assert optimize_beam is not None
        assert step is not None
        assert BeamConfig is not None
        assert IterationResult is not None
        assert OptimizationResult is not None
        assert get_default_model is not None

    def test_server_module_imports(self):
        """Test MCP server module imports."""
        from mcp_servers.beam_orchestrator.server import (
            mcp,
            optimization_step,
            optimize,
        )
        assert mcp is not None
        assert optimize is not None
        assert optimization_step is not None


class TestBeamConfig:
    """Test BeamConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from mcp_servers.beam_orchestrator.orchestrator import BeamConfig
        config = BeamConfig()
        assert config.beam_width == 3
        assert config.max_iterations == 10
        assert config.target_score == 0.90
        assert config.convergence_threshold == 0.02
        assert config.convergence_patience == 3
        assert "meta_prompt" in config.optimizers
        assert "few_shot" in config.optimizers

    def test_custom_config(self):
        """Test custom configuration."""
        from mcp_servers.beam_orchestrator.orchestrator import BeamConfig
        config = BeamConfig(
            beam_width=5,
            max_iterations=20,
            target_score=0.95,
            optimizers=["meta_prompt"],
        )
        assert config.beam_width == 5
        assert config.max_iterations == 20
        assert config.target_score == 0.95
        assert config.optimizers == ["meta_prompt"]


class TestIterationResult:
    """Test IterationResult dataclass."""

    def test_creation(self):
        """Test creating an IterationResult."""
        from mcp_servers.beam_orchestrator.orchestrator import IterationResult
        result = IterationResult(
            iteration=1,
            best_score=0.75,
            beam_scores=[0.75, 0.72, 0.70],
            beam_prompts=["p1", "p2", "p3"],
            candidates_generated=6,
            candidates_evaluated=9,
        )
        assert result.iteration == 1
        assert result.best_score == 0.75
        assert len(result.beam_scores) == 3
        assert result.candidates_generated == 6


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_creation(self):
        """Test creating an OptimizationResult."""
        from mcp_servers.beam_orchestrator.orchestrator import (
            IterationResult,
            OptimizationResult,
        )
        iter_result = IterationResult(
            iteration=1,
            best_score=0.85,
            beam_scores=[0.85],
            beam_prompts=["best prompt"],
            candidates_generated=3,
            candidates_evaluated=4,
        )
        result = OptimizationResult(
            best_prompt="best prompt",
            best_score=0.85,
            iterations=3,
            converged=True,
            convergence_reason="Target reached",
            history=[iter_result],
        )
        assert result.best_prompt == "best prompt"
        assert result.best_score == 0.85
        assert result.converged is True
        assert len(result.history) == 1


class TestGetDefaultModel:
    """Test model configuration."""

    def test_returns_string(self):
        """get_default_model should return a string."""
        from mcp_servers.beam_orchestrator.orchestrator import get_default_model
        result = get_default_model()
        assert isinstance(result, str)
        assert len(result) > 0


class TestBeamOrchestrationIntegration:
    """Integration tests requiring API keys."""

    @pytest.mark.skip(reason="Requires API keys - run manually")
    @pytest.mark.asyncio
    async def test_optimize_beam_basic(self):
        """Test basic beam optimization."""
        from mcp_servers.beam_orchestrator.orchestrator import (
            BeamConfig,
            optimize_beam,
        )

        examples = [
            TrainingExample(
                input="reverse 'hello'",
                expected_output="'olleh'",
            ),
        ]

        config = BeamConfig(
            beam_width=2,
            max_iterations=2,
            target_score=0.95,
        )

        result = await optimize_beam(
            initial_prompt="You are a helper.",
            training_examples=examples,
            config=config,
        )

        assert result.best_prompt is not None
        assert result.best_score > 0
        assert len(result.history) > 0

    @pytest.mark.skip(reason="Requires API keys - run manually")
    @pytest.mark.asyncio
    async def test_step_basic(self):
        """Test single optimization step."""
        from mcp_servers.beam_orchestrator.orchestrator import step

        examples = [
            TrainingExample(
                input="add 1 + 2",
                expected_output="3",
            ),
        ]

        result = await step(
            beam=["You are a calculator."],
            training_examples=examples,
            optimizers=["meta_prompt"],
        )

        assert "new_beam" in result
        assert "scores" in result
        assert len(result["new_beam"]) > 0
