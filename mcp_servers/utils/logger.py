"""Structured logging for Promptune."""

import sys
from dataclasses import dataclass
from enum import Enum


class Component(Enum):
    ORCHESTRATOR = "🎯 Orchestrator"
    EVALUATOR = "📊 Evaluator"
    META_OPTIMIZER = "🔧 Meta-Optimizer"
    FEW_SHOT_OPTIMIZER = "📝 Few-Shot"


@dataclass
class PromptuneLogger:
    verbose: bool = True

    def _print(self, msg: str):
        if self.verbose:
            print(msg, file=sys.stdout, flush=True)

    def stage(self, component: Component, message: str):
        self._print(f"\n{component.value} │ {message}")

    def info(self, message: str):
        self._print(f"  ├─ {message}")

    def success(self, message: str):
        self._print(f"  ✓ {message}")

    def warning(self, message: str):
        self._print(f"  ⚠ {message}")

    def header(self, title: str):
        self._print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")

    def iteration_start(self, i: int, total: int):
        self._print(f"\n┌{'─' * 58}┐\n│  ITERATION {i}/{total}\n└{'─' * 58}┘")


logger = PromptuneLogger()
