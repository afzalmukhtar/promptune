"""
Promptune Configuration System.

Loads configuration from YAML file with a single LLM role:
- target: The model the prompt is being tuned FOR

The agent itself acts as tuner (generates improved prompts) and judge
(evaluates outputs), so no separate tuner/judge models are needed.

Raises clear errors if config file is missing.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CONFIG_FILENAME = "promptune.yaml"

EXAMPLE_CONFIG = """# Promptune Configuration
# =======================
# Place API keys in .env file (LiteLLM reads them automatically):
#   OPENAI_API_KEY=...
#   ANTHROPIC_API_KEY=...
#   OLLAMA_API_BASE=http://localhost:11434

models:
  # The model the prompt is being tuned FOR (runs the prompt + input).
  # The agent itself acts as tuner and judge — no separate models needed.
  target: "gpt-4o-mini"

optimization:
  batch_size: 5
"""

@dataclass
class ModelConfig:
    """LLM model configuration — only the target model.

    The agent itself acts as tuner and judge.
    """

    target: str = "gpt-4o-mini"


@dataclass
class OptimizationConfig:
    """Optimization parameters."""

    batch_size: int = 5


@dataclass
class PromptuneConfig:
    """Top-level Promptune configuration."""

    models: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptuneConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            PromptuneConfig instance.

        Raises:
            FileNotFoundError: If config file doesn't exist (with helpful message).
            ValueError: If config is invalid.
        """
        path = Path(path)
        if not path.exists():
            _raise_missing_config_error(path)

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Config file {path} must be a YAML mapping, got {type(raw).__name__}")

        return cls._from_dict(raw, source=str(path))

    @classmethod
    def _from_dict(cls, data: dict, source: str = "config") -> "PromptuneConfig":
        """Build config from a dictionary."""
        # --- Models ---
        models_raw = data.get("models")
        if not models_raw or not isinstance(models_raw, dict):
            raise ValueError(
                f"{source}: 'models' section is required with a 'target' key."
            )

        if "target" not in models_raw or not models_raw["target"]:
            raise ValueError(
                f"{source}: 'models.target' is required. "
                f"Provide a LiteLLM model string (e.g. 'gpt-4o-mini', 'ollama/llama3.2')."
            )

        model_config = ModelConfig(
            target=str(models_raw["target"]),
        )

        # --- Optimization ---
        opt_raw = data.get("optimization", {})
        if not isinstance(opt_raw, dict):
            opt_raw = {}

        opt_config = OptimizationConfig(
            batch_size=opt_raw.get("batch_size", 5),
        )

        return cls(models=model_config, optimization=opt_config)

    @classmethod
    def from_params(
        cls,
        target_model: str,
        **optimization_kwargs,
    ) -> "PromptuneConfig":
        """Create config programmatically (for agent/skill usage).

        Args:
            target_model: LiteLLM model string for the target.
            **optimization_kwargs: Override any OptimizationConfig fields.
        """
        return cls(
            models=ModelConfig(target=target_model),
            optimization=OptimizationConfig(
                **{
                    k: v
                    for k, v in optimization_kwargs.items()
                    if k in OptimizationConfig.__dataclass_fields__
                }
            ),
        )


def load_config(config_path: str | Path | None = None) -> PromptuneConfig:
    """Load config from file, resolving default path.

    Args:
        config_path: Explicit path, or None to use default 'promptune.yaml'.

    Returns:
        PromptuneConfig instance.
    """
    path = Path(config_path) if config_path else Path(DEFAULT_CONFIG_FILENAME)
    return PromptuneConfig.from_yaml(path)


def _raise_missing_config_error(path: Path) -> None:
    """Raise a helpful error when config file is missing."""
    print(
        f"\n{'=' * 60}\n"
        f"  ERROR: Config file not found: {path}\n"
        f"{'=' * 60}\n\n"
        f"Promptune requires a configuration file.\n\n"
        f"1. Create '{path}' with the following content:\n\n"
        f"{EXAMPLE_CONFIG}\n"
        f"2. Create a '.env' file with your API keys:\n\n"
        f"   OPENAI_API_KEY=your-key\n"
        f"   # Or for Ollama (local):\n"
        f"   # OLLAMA_API_BASE=http://localhost:11434\n\n"
        f"{'=' * 60}\n",
        file=sys.stderr,
    )
    raise FileNotFoundError(
        f"Config file not found: {path}. "
        f"Create it with model configuration. See error output above for example."
    )
