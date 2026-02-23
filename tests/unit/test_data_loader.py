"""Unit tests for the data loader."""

import json
import tempfile

import pytest


class TestLoadDatasetJSON:
    """Test loading datasets from JSON files."""

    def test_load_positive_examples(self):
        from mcp_servers.utils.data_loader import load_dataset

        data = [
            {"input": "hello", "expected_output": "world"},
            {"input": "foo", "expected_output": "bar"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            dataset = load_dataset(f.name)

        assert len(dataset.examples) == 2
        assert len(dataset.negative_examples) == 0
        assert dataset.examples[0].input == "hello"
        assert dataset.examples[1].expected_output == "bar"

    def test_load_negative_examples(self):
        from mcp_servers.utils.data_loader import load_dataset

        data = [
            {
                "sample_prompt": "Be helpful",
                "input": "hello",
                "bad_output": "go away",
                "reason_why_bad": "Not helpful",
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            dataset = load_dataset(f.name)

        assert len(dataset.examples) == 0
        assert len(dataset.negative_examples) == 1
        assert dataset.negative_examples[0].bad_output == "go away"

    def test_load_mixed_examples(self):
        from mcp_servers.utils.data_loader import load_dataset

        data = [
            {"input": "hello", "expected_output": "world"},
            {
                "sample_prompt": "Be helpful",
                "input": "hello",
                "bad_output": "go away",
                "reason_why_bad": "Not helpful",
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            dataset = load_dataset(f.name)

        assert len(dataset.examples) == 1
        assert len(dataset.negative_examples) == 1

    def test_load_empty_file_raises(self):
        from mcp_servers.utils.data_loader import load_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            f.flush()
            with pytest.raises(ValueError, match="empty"):
                load_dataset(f.name)


class TestLoadDatasetCSV:
    """Test loading datasets from CSV files."""

    def test_load_positive_csv(self):
        from mcp_servers.utils.data_loader import load_dataset

        csv_content = "input,expected_output\nhello,world\nfoo,bar\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            dataset = load_dataset(f.name)

        assert len(dataset.examples) == 2
        assert dataset.examples[0].input == "hello"

    def test_load_negative_csv(self):
        from mcp_servers.utils.data_loader import load_dataset

        csv_content = "sample_prompt,input,bad_output,reason_why_bad\nBe helpful,hello,go away,Not helpful\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            dataset = load_dataset(f.name)

        assert len(dataset.negative_examples) == 1
        assert dataset.negative_examples[0].reason_why_bad == "Not helpful"


class TestLoadDatasetFromDicts:
    """Test load_dataset_from_dicts function."""

    def test_positive_dicts(self):
        from mcp_servers.utils.data_loader import load_dataset_from_dicts

        rows = [
            {"input": "a", "expected_output": "b"},
            {"input": "c", "expected_output": "d"},
        ]
        dataset = load_dataset_from_dicts(rows)
        assert len(dataset.examples) == 2
        assert len(dataset.negative_examples) == 0

    def test_negative_dicts(self):
        from mcp_servers.utils.data_loader import load_dataset_from_dicts

        rows = [
            {
                "sample_prompt": "p",
                "input": "i",
                "bad_output": "b",
                "reason_why_bad": "r",
            },
        ]
        dataset = load_dataset_from_dicts(rows)
        assert len(dataset.examples) == 0
        assert len(dataset.negative_examples) == 1

    def test_unsupported_format(self):
        from mcp_servers.utils.data_loader import load_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not a valid format")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported"):
                load_dataset(f.name)
