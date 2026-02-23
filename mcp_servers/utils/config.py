"""
Promptune Configuration System.

Loads configuration from YAML file with 3 LLM roles:
- target: The model the prompt is being tuned FOR
- tuner: Generates improved prompt candidates (optimizers)
- judge: Scores outputs, analyzes prompt understanding

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
#   AZURE_OPENAI_API_KEY=...
#   AZURE_OPENAI_ENDPOINT=...
#   OPENAI_API_KEY=...
#   ANTHROPIC_API_KEY=...
#   OLLAMA_API_BASE=http://localhost:11434

models:
  # The model the prompt is being tuned FOR (runs the prompt + input)
  target: "azure/gpt-4o-mini"
  # Generates improved prompt candidates (all optimizers)
  tuner: "azure/gpt-4o"
  # Scores outputs, analyzes prompt understanding
  judge: "azure/gpt-4o-mini"

optimization:
  beam_width: 3
  max_iterations: 10
  target_score: 0.90
  convergence_threshold: 0.02
  convergence_patience: 3
  batch_size: 5
  optimizers:
    - meta_prompt
    - few_shot
    - adversarial
    - example_augmentor
    - clarity_rewriter
"""

VALID_OPTIMIZERS = {
    "meta_prompt",
    "few_shot",
    "adversarial",
    "example_augmentor",
    "clarity_rewriter",
}


@dataclass
class ModelConfig:
    """LLM model configuration with 3 roles."""

    target: str = "gpt-4o-mini"
    tuner: str = "gpt-4o-mini"
    judge: str = "gpt-4o-mini"


@dataclass
class OptimizationConfig:
    """Beam search optimization parameters."""

    beam_width: int = 3
    max_iterations: int = 10
    target_score: float = 0.90
    convergence_threshold: float = 0.02
    convergence_patience: int = 3
    batch_size: int = 5
    optimizers: list[str] = field(
        default_factory=lambda: [
            "meta_prompt",
            "few_shot",
            "adversarial",
            "example_augmentor",
            "clarity_rewriter",
        ]
    )


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
                f"{source}: 'models' section is required with 'target', 'tuner', and 'judge' keys."
            )

        for key in ("target", "tuner", "judge"):
            if key not in models_raw or not models_raw[key]:
                raise ValueError(
                    f"{source}: 'models.{key}' is required. "
                    f"Provide a LiteLLM model string (e.g. 'azure/gpt-4o-mini', 'ollama/llama3.2')."
                )

        model_config = ModelConfig(
            target=str(models_raw["target"]),
            tuner=str(models_raw["tuner"]),
            judge=str(models_raw["judge"]),
        )

        # --- Optimization ---
        opt_raw = data.get("optimization", {})
        if not isinstance(opt_raw, dict):
            opt_raw = {}

        default_optimizers = [
            "meta_prompt",
            "few_shot",
            "adversarial",
            "example_augmentor",
            "clarity_rewriter",
        ]
        optimizers = opt_raw.get("optimizers", default_optimizers)
        if isinstance(optimizers, list):
            invalid = set(optimizers) - VALID_OPTIMIZERS
            if invalid:
                raise ValueError(
                    f"{source}: Invalid optimizer(s): {invalid}. Valid options: {VALID_OPTIMIZERS}"
                )

        opt_config = OptimizationConfig(
            beam_width=opt_raw.get("beam_width", 3),
            max_iterations=opt_raw.get("max_iterations", 10),
            target_score=opt_raw.get("target_score", 0.90),
            convergence_threshold=opt_raw.get("convergence_threshold", 0.02),
            convergence_patience=opt_raw.get("convergence_patience", 3),
            batch_size=opt_raw.get("batch_size", 5),
            optimizers=optimizers if isinstance(optimizers, list) else list(optimizers),
        )

        return cls(models=model_config, optimization=opt_config)

    @classmethod
    def from_params(
        cls,
        target_model: str,
        tuner_model: str,
        judge_model: str,
        **optimization_kwargs,
    ) -> "PromptuneConfig":
        """Create config programmatically (for agent/skill usage).

        Args:
            target_model: LiteLLM model string for the target.
            tuner_model: LiteLLM model string for the tuner.
            judge_model: LiteLLM model string for the judge.
            **optimization_kwargs: Override any OptimizationConfig fields.
        """
        return cls(
            models=ModelConfig(target=target_model, tuner=tuner_model, judge=judge_model),
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
        f"   AZURE_OPENAI_API_KEY=your-key\n"
        f"   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/\n"
        f"   # Or for OpenAI:\n"
        f"   # OPENAI_API_KEY=your-key\n"
        f"   # Or for Ollama (local):\n"
        f"   # OLLAMA_API_BASE=http://localhost:11434\n\n"
        f"{'=' * 60}\n",
        file=sys.stderr,
    )
    raise FileNotFoundError(
        f"Config file not found: {path}. "
        f"Create it with model configuration. See error output above for example."
    )
