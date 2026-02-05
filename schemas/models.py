"""Core data models."""

from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """Input/output training pair."""
    input: str = Field(..., description="Input text")
    expected_output: str = Field(..., description="Expected output")
    metadata: dict = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Evaluation result."""
    prompt: str
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    feedback: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
