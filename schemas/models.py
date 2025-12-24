"""
Promptune Core Data Models.

Defines all Pydantic schemas used across the system:
- TrainingExample / NegativeTrainingExample / TrainingDataset: Input data
- PromptCandidate: A prompt with evaluation results
- EvaluationResult: Score and feedback from evaluator
- PromptSectionAnalysis / PromptUnderstanding: Prompt understanding feedback
- SampleConfig / SampleState / OptimizationResult: Optimization types
- OutputComparison, StructuralAnalysis, etc.: Structured LLM response models
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

# =============================================================================
# Step 1.3: Base Schemas
# =============================================================================


class TrainingExample(BaseModel):
    """A single input/output training pair for evaluation."""

    input: str = Field(..., description="Input text for the task")
    expected_output: str = Field(..., description="Expected output for this input")
    metadata: dict = Field(default_factory=dict, description="Optional metadata")


class NegativeTrainingExample(BaseModel):
    """A negative training example showing bad output and why it's bad."""

    input: str = Field(..., description="The input given")
    bad_output: str = Field(..., description="The bad output produced")
    reason_why_bad: str = Field(..., description="Why this output is bad")


class TrainingDataset(BaseModel):
    """Unified dataset holding both positive and negative training examples."""

    examples: list[TrainingExample] = Field(default_factory=list, description="Positive examples")
    negative_examples: list[NegativeTrainingExample] = Field(
        default_factory=list, description="Negative examples showing what to avoid"
    )

    @classmethod
    def from_examples(cls, examples: list[TrainingExample]) -> "TrainingDataset":
        """Create a dataset from a list of positive examples (backward compat)."""
        return cls(examples=examples)


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


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

    # Prompt understanding analysis
    prompt_understanding: "PromptUnderstanding | None" = Field(
        default=None, description="Analysis of which prompt sections were followed/ignored"
    )


# =============================================================================
# Step 1.5: Candidate Sample Schemas
# =============================================================================


class SampleConfig(BaseModel):
    """Configuration for iterative optimization."""

    sample_width: int = Field(default=3, ge=1, le=20, description="Candidates to keep (k)")
    max_rounds: int = Field(default=5, ge=1, le=50, description="Maximum rounds")
    patience: int = Field(default=3, ge=1, description="Rounds without improvement to stop")
    min_improvement: float = Field(default=0.01, ge=0.0, description="Minimum score gain")
    pass_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Success threshold")


class SampleState(BaseModel):
    """Current state of iterative optimization."""

    candidates: list[PromptCandidate] = Field(
        default_factory=list, description="Current candidates"
    )
    current_round: int = Field(default=0, ge=0, description="Current round number")
    best_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Best score seen")
    best_prompt: str = Field(default="", description="Best prompt seen")
    rounds_without_improvement: int = Field(default=0, ge=0)
    total_evaluated: int = Field(default=0, ge=0, description="Total candidates evaluated")


class OptimizationResult(BaseModel):
    """Final result of iterative optimization."""

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


# =============================================================================
# Prompt Understanding Schemas
# =============================================================================


class PromptSectionAnalysis(BaseModel):
    """Analysis of how well a specific prompt section was followed by the LLM."""

    section: str = Field(..., description="Which part/instruction of the prompt")
    evidence: str = Field(..., description="What the LLM actually did for this section")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="How well followed (0=ignored, 1=perfect)"
    )
    reason: str = Field(default="", description="Why it was poorly followed, if applicable")


class PromptUnderstanding(BaseModel):
    """Full analysis of prompt understanding by the target LLM."""

    well_followed: list[PromptSectionAnalysis] = Field(
        default_factory=list, description="Prompt sections that were well followed"
    )
    poorly_followed: list[PromptSectionAnalysis] = Field(
        default_factory=list, description="Prompt sections that were poorly followed or ignored"
    )
    overall_compliance: float = Field(..., ge=0.0, le=1.0, description="Overall compliance score")


# =============================================================================
# Structured LLM Response Models (for tool calling)
# =============================================================================


class OutputComparison(BaseModel):
    """Compare actual LLM output to expected output."""

    semantic_match: bool = Field(
        ..., description="Does actual output convey same meaning as expected?"
    )
    format_match: bool = Field(..., description="Is the format similar?")
    correctness: bool = Field(
        ..., description="Is the actual output factually/functionally correct?"
    )
    completeness: bool = Field(..., description="Does actual output fully address the task?")
    explanation: str = Field(..., description="Brief explanation of differences if any")


class NegativeOutputComparison(BaseModel):
    """Compare actual output to known BAD output — reverse scoring (match = bad)."""

    matches_bad_output: bool = Field(
        ..., description="Is the actual output semantically similar to the known bad output?"
    )
    matches_bad_pattern: bool = Field(
        ..., description="Does the actual output exhibit the failure described in reason_why_bad?"
    )
    same_tone: bool = Field(
        ...,
        description="Does the actual output have the same problematic tone/style as the bad output?",
    )
    same_mistakes: bool = Field(
        ...,
        description="Does the actual output repeat the same specific mistakes as the bad output?",
    )
    explanation: str = Field(
        ..., description="Brief explanation of how the output compares to the bad example"
    )


class StructuralAnalysis(BaseModel):
    """Analyze prompt structure for required components."""

    has_role: bool = Field(..., description="Defines WHO the AI should be?")
    has_task: bool = Field(..., description="Explains WHAT to do?")
    has_format: bool = Field(..., description="Specifies HOW to format output?")
    has_constraints: bool = Field(..., description="Sets boundaries or rules?")
    has_examples: bool = Field(..., description="Includes examples?")


class AdversarialAnalysis(BaseModel):
    """Adversarial critique finding weaknesses in a prompt."""

    weaknesses: list[str] = Field(..., description="Specific weaknesses found")
    suggestions: list[str] = Field(..., description="Actionable suggestions for fixes")
    assessment: str = Field(..., description="One sentence harsh but fair assessment")


class ExampleScore(BaseModel):
    """Score for a single training example."""

    index: int = Field(..., description="Index of the example in the pool")
    relevance: float = Field(..., ge=0.0, le=1.0, description="How relevant to the task")
    diversity: float = Field(..., ge=0.0, le=1.0, description="How different from typical cases")
    complexity: float = Field(..., ge=0.0, le=1.0, description="How complex the example is")


class ExampleScores(BaseModel):
    """Scores for all candidate examples."""

    scores: list[ExampleScore] = Field(..., description="Scores for each example")


class CandidateOutput(BaseModel):
    """A single optimized prompt candidate from an optimizer."""

    prompt: str = Field(..., description="The complete improved prompt")
    strategy: str = Field(..., description="What improvements were made")
    addressed_weaknesses: list[str] = Field(
        default_factory=list, description="List of weaknesses fixed"
    )


class OptimizationCandidates(BaseModel):
    """Result from an optimizer containing multiple candidate prompts."""

    candidates: list[CandidateOutput] = Field(..., description="List of improved prompt candidates")


class PromptUnderstandingResponse(BaseModel):
    """LLM response for prompt understanding analysis."""

    well_followed: list[PromptSectionAnalysis] = Field(
        default_factory=list,
        description="Sections well followed: each with 'section', 'evidence', 'score'",
    )
    poorly_followed: list[PromptSectionAnalysis] = Field(
        default_factory=list,
        description="Sections poorly followed: each with 'section', 'evidence', 'score', 'reason'",
    )
    overall_compliance: float = Field(..., ge=0.0, le=1.0, description="Overall compliance score")


class ClarityAnalysis(BaseModel):
    """Identify unclear sentences in a prompt."""

    unclear_sentences: list[str] = Field(..., description="Sentences that are ambiguous or unclear")
    rewritten_sentences: list[str] = Field(
        ..., description="Improved versions of each unclear sentence (same order)"
    )
    reasoning: list[str] = Field(..., description="Why each sentence was unclear (same order)")


class AdversarialInputs(BaseModel):
    """Generated adversarial inputs that may break a prompt."""

    adversarial_cases: list[str] = Field(
        ..., description="Adversarial inputs designed to break the prompt"
    )
    failure_modes: list[str] = Field(
        ..., description="Expected failure mode for each adversarial input"
    )
    hardening_suggestions: list[str] = Field(
        ..., description="Suggestions to harden the prompt against these failures"
    )
