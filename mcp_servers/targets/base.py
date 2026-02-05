"""Evaluation Target Protocol."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class EvaluationTarget(Protocol):
    """Protocol for black-box evaluation targets."""
    
    async def invoke(self, prompt: str, input_text: str) -> str:
        ...


class BaseTarget(ABC):
    """Base class for targets."""
    
    @abstractmethod
    async def invoke(self, prompt: str, input_text: str) -> str:
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
