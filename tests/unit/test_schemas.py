"""
Unit tests for Promptune schemas.

Tests Steps 1.3-1.5: Base schemas, evaluation schemas, beam state schemas.
"""

import pytest

from schemas import (
    BeamConfig,
    BeamState,
    EvaluationResult,
    OptimizationResult,
    PromptCandidate,
    TrainingExample,
)

# =============================================================================
# Step 1.3: Base Schemas Tests
# =============================================================================


class TestTrainingExample:
    """Tests for TrainingExample model."""

    def test_creation_basic(self):
        """Test basic creation with required fields."""
        ex = TrainingExample(input="hello", expected_output="world")
        assert ex.input == "hello"
        assert ex.expected_output == "world"
        assert ex.metadata == {}

    def test_creation_with_metadata(self):
        """Test creation with optional metadata."""
        ex = TrainingExample(
            input="hello",
            expected_output="world",
            metadata={"source": "test", "difficulty": "easy"},
        )
        assert ex.metadata["source"] == "test"


class TestPromptCandidate:
    """Tests for PromptCandidate model."""

    def test_defaults(self):
        """Test default values are set correctly."""
        pc = PromptCandidate(prompt="Test prompt")
        assert pc.prompt == "Test prompt"
        assert pc.score == 0.0
        assert pc.source_optimizer is None
        assert pc.parent_prompt is None
        assert pc.round_created == 0
        assert pc.feedback == ""
        assert pc.strengths == []
        assert pc.weaknesses == []

    def test_full_creation(self):
        """Test creation with all fields."""
        pc = PromptCandidate(
            prompt="You are a helpful assistant.",
            score=0.85,
            source_optimizer="meta-prompt",
            parent_prompt="Original prompt",
            round_created=2,
            feedback="Good clarity",
            strengths=["Clear instructions"],
            weaknesses=["Missing examples"],
        )
        assert pc.score == 0.85
        assert pc.source_optimizer == "meta-prompt"
        assert len(pc.strengths) == 1

    def test_score_validation(self):
        """Test score must be between 0 and 1."""
        with pytest.raises(ValueError):
            PromptCandidate(prompt="Test", score=1.5)
        with pytest.raises(ValueError):
            PromptCandidate(prompt="Test", score=-0.1)


# =============================================================================
# Step 1.4: Evaluation Schemas Tests
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_basic_creation(self):
        """Test basic creation with required fields."""
        result = EvaluationResult(
            prompt="Test prompt",
            score=0.75,
            passed=True,
            feedback="Good prompt overall",
        )
        assert result.score == 0.75
        assert result.passed is True
        assert result.strengths == []
        assert result.weaknesses == []

    def test_with_detailed_feedback(self):
        """Test creation with detailed feedback."""
        result = EvaluationResult(
            prompt="Test",
            score=0.85,
            passed=True,
            feedback="Excellent",
            strengths=["Clear instructions", "Good examples"],
            weaknesses=["Slightly verbose"],
            suggestions=["Trim unnecessary words"],
        )
        assert len(result.strengths) == 2
        assert len(result.suggestions) == 1

    def test_with_detailed_metrics(self):
        """Test creation with optional detailed metrics."""
        result = EvaluationResult(
            prompt="Test",
            score=0.85,
            passed=True,
            feedback="Good",
            clarity_score=0.9,
            task_alignment_score=0.8,
            example_quality_score=0.85,
        )
        assert result.clarity_score == 0.9
        assert result.task_alignment_score == 0.8


# =============================================================================
# Step 1.5: Beam State Schemas Tests
# =============================================================================


class TestBeamConfig:
    """Tests for BeamConfig model."""

    def test_defaults(self):
        """Test default configuration values."""
        config = BeamConfig()
        assert config.beam_width == 3
        assert config.max_rounds == 5
        assert config.patience == 3
        assert config.min_improvement == 0.01
        assert config.pass_threshold == 0.8

    def test_custom_config(self):
        """Test custom configuration."""
        config = BeamConfig(
            beam_width=5,
            max_rounds=10,
            patience=2,
            pass_threshold=0.9,
        )
        assert config.beam_width == 5
        assert config.max_rounds == 10

    def test_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            BeamConfig(beam_width=0)  # Must be >= 1
        with pytest.raises(ValueError):
            BeamConfig(beam_width=25)  # Must be <= 20


class TestBeamState:
    """Tests for BeamState model."""

    def test_empty_state(self):
        """Test empty initial state."""
        state = BeamState()
        assert state.candidates == []
        assert state.current_round == 0
        assert state.best_score == 0.0
        assert state.best_prompt == ""

    def test_with_candidates(self):
        """Test state with candidates."""
        candidates = [
            PromptCandidate(prompt="Prompt 1", score=0.7),
            PromptCandidate(prompt="Prompt 2", score=0.8),
        ]
        state = BeamState(
            candidates=candidates,
            current_round=2,
            best_score=0.8,
            best_prompt="Prompt 2",
        )
        assert len(state.candidates) == 2
        assert state.best_score == 0.8


class TestOptimizationResult:
    """Tests for OptimizationResult model."""

    def test_creation(self):
        """Test optimization result creation."""
        result = OptimizationResult(
            best_prompt="Optimized prompt",
            best_score=0.9,
            original_prompt="Original prompt",
            original_score=0.5,
            improvement=0.4,
            rounds_completed=3,
            total_candidates_evaluated=15,
            converged=True,
            convergence_reason="threshold_reached",
        )
        assert result.improvement == 0.4
        assert result.converged is True
        assert result.convergence_reason == "threshold_reached"
