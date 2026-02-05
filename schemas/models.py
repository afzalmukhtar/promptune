"""
Promptune Core Data Models.

Defines all Pydantic schemas used across the system:
- TrainingExample: Input/output training pairs
- PromptCandidate: A prompt with evaluation results
- EvaluationResult: Score and feedback from evaluator
- BeamConfig: Beam search configuration
- BeamState: Current beam state
- OptimizationResult: Final optimization output
"""

from datetime import datetime

from pydantic import BaseModel, Field

# =============================================================================
# Step 1.3: Base Schemas
# =============================================================================


class TrainingExample(BaseModel):
    """A single input/output training pair for evaluation."""

    input: str = Field(..., description="Input text for the task")
    expected_output: str = Field(..., description="Expected output for this input")
    metadata: dict = Field(default_factory=dict, description="Optional metadata")


class PromptCandidate(BaseModel):
    """A prompt candidate with evaluation results and lineage tracking."""

    prompt: str = Field(..., description="The prompt text")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Evaluation score")

    # Source tracking
    source_optimizer: str | None = Field(
        default=None, description="Optimizer that created this candidate"
    )
    parent_prompt: str | None = Field(default=None, description="The prompt this was derived from")
    round_created: int = Field(default=0, ge=0, description="Round when created")

    # Feedback
    feedback: str = Field(default="", description="Evaluation feedback text")
    strengths: list[str] = Field(default_factory=list, description="What works well")
    weaknesses: list[str] = Field(default_factory=list, description="Issues identified")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Step 1.4: Evaluation Schemas
# =============================================================================


class EvaluationResult(BaseModel):
    """Result from evaluating a prompt against training examples."""

    prompt: str = Field(..., description="The evaluated prompt")
    score: float = Field(..., ge=0.0, le=1.0, description="Overall score (0-1)")
    passed: bool = Field(..., description="Whether prompt meets threshold")
    feedback: str = Field(..., description="Overall feedback text")

    # Detailed feedback
    strengths: list[str] = Field(default_factory=list, description="What works well")
    weaknesses: list[str] = Field(default_factory=list, description="Issues found")
    suggestions: list[str] = Field(default_factory=list, description="Improvements")

    # Optional detailed metrics
    clarity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    task_alignment_score: float | None = Field(default=None, ge=0.0, le=1.0)
    example_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)


# =============================================================================
# Step 1.5: Beam State Schemas
# =============================================================================


class BeamConfig(BaseModel):
    """Configuration for beam search optimization."""

    beam_width: int = Field(default=3, ge=1, le=20, description="Candidates to keep (k)")
    max_rounds: int = Field(default=5, ge=1, le=50, description="Maximum rounds")
    patience: int = Field(default=3, ge=1, description="Rounds without improvement to stop")
    min_improvement: float = Field(default=0.01, ge=0.0, description="Minimum score gain")
    pass_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Success threshold")


class BeamState(BaseModel):
    """Current state of beam search optimization."""

    candidates: list[PromptCandidate] = Field(
        default_factory=list, description="Current beam candidates"
    )
    current_round: int = Field(default=0, ge=0, description="Current round number")
    best_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Best score seen")
    best_prompt: str = Field(default="", description="Best prompt seen")
    rounds_without_improvement: int = Field(default=0, ge=0)
    total_evaluated: int = Field(default=0, ge=0, description="Total candidates evaluated")


class OptimizationResult(BaseModel):
    """Final result of beam search optimization."""

    # Best result
    best_prompt: str = Field(..., description="Best prompt found")
    best_score: float = Field(..., ge=0.0, le=1.0, description="Score of best prompt")

    # Comparison
    original_prompt: str = Field(..., description="Starting prompt")
    original_score: float = Field(..., ge=0.0, le=1.0, description="Score of original")
    improvement: float = Field(..., description="Score improvement (best - original)")

    # Run info
    rounds_completed: int = Field(..., ge=0, description="Rounds executed")
    total_candidates_evaluated: int = Field(..., ge=0, description="Total evaluated")
    converged: bool = Field(..., description="Whether optimization converged")
    convergence_reason: str = Field(..., description="Why optimization stopped")
