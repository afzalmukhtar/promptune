"""Unit tests for the configuration system."""

import tempfile

import pytest
import yaml


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_defaults(self):
        from mcp_servers.utils.config import ModelConfig
        config = ModelConfig()
        assert config.target == "gpt-4o-mini"
        assert config.tuner == "gpt-4o-mini"
        assert config.judge == "gpt-4o-mini"

    def test_custom(self):
        from mcp_servers.utils.config import ModelConfig
        config = ModelConfig(target="azure/gpt-4o", tuner="ollama/llama3.2", judge="gpt-4o-mini")
        assert config.target == "azure/gpt-4o"
        assert config.tuner == "ollama/llama3.2"


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""

    def test_defaults(self):
        from mcp_servers.utils.config import OptimizationConfig
        config = OptimizationConfig()
        assert config.beam_width == 3
        assert config.max_iterations == 10
        assert config.target_score == 0.90
        assert config.batch_size == 5
        assert "meta_prompt" in config.optimizers

    def test_custom(self):
        from mcp_servers.utils.config import OptimizationConfig
        config = OptimizationConfig(beam_width=5, batch_size=10, optimizers=["meta_prompt"])
        assert config.beam_width == 5
        assert config.batch_size == 10
        assert config.optimizers == ["meta_prompt"]


class TestPromptuneConfig:
    """Test PromptuneConfig dataclass."""

    def test_defaults(self):
        from mcp_servers.utils.config import PromptuneConfig
        config = PromptuneConfig()
        assert config.models.target == "gpt-4o-mini"
        assert config.optimization.beam_width == 3

    def test_from_params(self):
        from mcp_servers.utils.config import PromptuneConfig
        config = PromptuneConfig.from_params(
            target_model="azure/gpt-4o",
            tuner_model="azure/gpt-4o",
            judge_model="azure/gpt-4o-mini",
            beam_width=5,
            batch_size=10,
        )
        assert config.models.target == "azure/gpt-4o"
        assert config.models.tuner == "azure/gpt-4o"
        assert config.models.judge == "azure/gpt-4o-mini"
        assert config.optimization.beam_width == 5
        assert config.optimization.batch_size == 10

    def test_from_yaml_valid(self):
        from mcp_servers.utils.config import PromptuneConfig
        data = {
            "models": {
                "target": "azure/gpt-4o-mini",
                "tuner": "azure/gpt-4o",
                "judge": "azure/gpt-4o-mini",
            },
            "optimization": {
                "beam_width": 4,
                "max_iterations": 15,
                "batch_size": 8,
                "optimizers": ["meta_prompt", "few_shot"],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = PromptuneConfig.from_yaml(f.name)

        assert config.models.target == "azure/gpt-4o-mini"
        assert config.models.tuner == "azure/gpt-4o"
        assert config.optimization.beam_width == 4
        assert config.optimization.batch_size == 8
        assert config.optimization.optimizers == ["meta_prompt", "few_shot"]

    def test_from_yaml_missing_file(self):
        from mcp_servers.utils.config import PromptuneConfig
        with pytest.raises(FileNotFoundError):
            PromptuneConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_missing_models(self):
        from mcp_servers.utils.config import PromptuneConfig
        data = {"optimization": {"beam_width": 3}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="models"):
                PromptuneConfig.from_yaml(f.name)

    def test_from_yaml_missing_model_key(self):
        from mcp_servers.utils.config import PromptuneConfig
        data = {
            "models": {
                "target": "gpt-4o-mini",
                "tuner": "gpt-4o",
                # missing judge
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="judge"):
                PromptuneConfig.from_yaml(f.name)

    def test_from_yaml_invalid_optimizer(self):
        from mcp_servers.utils.config import PromptuneConfig
        data = {
            "models": {
                "target": "gpt-4o-mini",
                "tuner": "gpt-4o",
                "judge": "gpt-4o-mini",
            },
            "optimization": {
                "optimizers": ["meta_prompt", "nonexistent_optimizer"],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="Invalid optimizer"):
                PromptuneConfig.from_yaml(f.name)

    def test_from_yaml_defaults_for_optimization(self):
        """If optimization section is missing, defaults should be used."""
        from mcp_servers.utils.config import PromptuneConfig
        data = {
            "models": {
                "target": "gpt-4o-mini",
                "tuner": "gpt-4o",
                "judge": "gpt-4o-mini",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = PromptuneConfig.from_yaml(f.name)

        assert config.optimization.beam_width == 3
        assert config.optimization.max_iterations == 10


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_missing_default(self, monkeypatch):
        """load_config with no args should raise if promptune.yaml doesn't exist."""
        from mcp_servers.utils.config import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            with pytest.raises(FileNotFoundError):
                load_config()

    def test_load_config_custom_path(self):
        from mcp_servers.utils.config import load_config
        data = {
            "models": {
                "target": "gpt-4o-mini",
                "tuner": "gpt-4o",
                "judge": "gpt-4o-mini",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_config(f.name)

        assert config.models.tuner == "gpt-4o"
