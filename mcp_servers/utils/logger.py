"""
Promptune Logging Utilities

Provides structured logging for the Promptune optimization pipeline.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    STAGE = "STAGE"
    RESULT = "RESULT"
    ERROR = "ERROR"


class Component(Enum):
    ORCHESTRATOR = "🎯 Orchestrator"
    EVALUATOR = "📊 Evaluator"
    META_OPTIMIZER = "🔧 Meta-Optimizer"
    FEW_SHOT_OPTIMIZER = "📝 Few-Shot"
    ADVERSARIAL_OPTIMIZER = "⚔️ Adversarial"
    EXAMPLE_AUGMENTOR = "📚 Example-Augmentor"
    CLARITY_REWRITER = "✨ Clarity-Rewriter"
    LLM = "🤖 LLM"


@dataclass
class PromptuneLogger:
    """Structured logger for Promptune pipeline."""

    verbose: bool = True
    show_timestamps: bool = False
    indent_level: int = 0

    def _format_time(self) -> str:
        if self.show_timestamps:
            return datetime.now().strftime("%H:%M:%S") + " "
        return ""

    def _indent(self) -> str:
        return "  " * self.indent_level

    def _print(self, msg: str):
        if self.verbose:
            print(msg, file=sys.stdout, flush=True)

    def stage(self, component: Component, message: str):
        """Log a major stage transition."""
        self._print(f"\n{self._format_time()}{component.value} │ {message}")

    def info(self, message: str):
        """Log general info."""
        self._print(f"{self._format_time()}{self._indent()}  ├─ {message}")

    def detail(self, message: str):
        """Log a detail/sub-item."""
        self._print(f"{self._format_time()}{self._indent()}  │  {message}")

    def result(self, label: str, value: str):
        """Log a result with label."""
        self._print(f"{self._format_time()}{self._indent()}  ├─ {label}: {value}")

    def success(self, message: str):
        """Log a success message."""
        self._print(f"{self._format_time()}{self._indent()}  ✓ {message}")

    def warning(self, message: str):
        """Log a warning."""
        self._print(f"{self._format_time()}{self._indent()}  ⚠ {message}")

    def error(self, message: str):
        """Log an error."""
        self._print(f"{self._format_time()}{self._indent()}  ✗ {message}")

    def separator(self):
        """Print a separator line."""
        self._print("─" * 60)

    def header(self, title: str):
        """Print a header."""
        self._print(f"\n{'═' * 60}")
        self._print(f"  {title}")
        self._print(f"{'═' * 60}")

    def subheader(self, title: str):
        """Print a subheader."""
        self._print(f"\n{'─' * 40}")
        self._print(f"  {title}")
        self._print(f"{'─' * 40}")

    def prompt_preview(self, prompt: str, max_len: int = 100):
        """Show a preview of a prompt."""
        preview = prompt.replace("\n", " ")[:max_len]
        if len(prompt) > max_len:
            preview += "..."
        self._print(f'{self._format_time()}{self._indent()}  │  "{preview}"')

    def iteration_start(self, iteration: int, total: int):
        """Log the start of an iteration."""
        self._print(f"\n┌{'─' * 58}┐")
        self._print(f"│  ITERATION {iteration}/{total}")
        self._print(f"└{'─' * 58}┘")

    def iteration_end(self, score: float, improved: bool):
        """Log the end of an iteration."""
        status = "↑ Improved" if improved else "→ No change"
        self._print(f"\n  └─ Score: {score:.0%} {status}")


# Global logger instance
logger = PromptuneLogger()
