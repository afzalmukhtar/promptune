"""
Data Loader — Auto-infer input format from JSON/CSV files.

Supports two formats with fixed column names:
- Positive: input, expected_output
- Negative: sample_prompt, input, bad_output, reason_why_bad

Auto-detects which format based on columns present. Mixed datasets supported.
"""

import csv
import json
from pathlib import Path

from schemas import NegativeTrainingExample, TrainingDataset, TrainingExample

POSITIVE_REQUIRED_COLUMNS = {"input", "expected_output"}
NEGATIVE_REQUIRED_COLUMNS = {"sample_prompt", "input", "bad_output", "reason_why_bad"}
ALL_VALID_COLUMNS = POSITIVE_REQUIRED_COLUMNS | NEGATIVE_REQUIRED_COLUMNS | {"metadata"}


def load_dataset(path: str | Path) -> TrainingDataset:
    """Load a training dataset from a JSON or CSV file.

    Auto-infers the format based on column names present in the data.

    Args:
        path: Path to JSON or CSV file.

    Returns:
        TrainingDataset with positive and/or negative examples.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If format cannot be inferred or columns are invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        rows = _load_json(path)
    elif suffix == ".csv":
        rows = _load_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Use .json or .csv"
        )

    if not rows:
        raise ValueError(f"Dataset file is empty: {path}")

    return _parse_rows(rows, source=str(path))


def load_dataset_from_dicts(rows: list[dict]) -> TrainingDataset:
    """Load a training dataset from a list of dicts (programmatic use).

    Args:
        rows: List of dicts with appropriate column names.

    Returns:
        TrainingDataset with positive and/or negative examples.
    """
    if not rows:
        return TrainingDataset()
    return _parse_rows(rows, source="dict input")


def _load_json(path: Path) -> list[dict]:
    """Load rows from a JSON file (expects list of objects)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Support {"examples": [...], "negative_examples": [...]} wrapper
        rows = []
        if "examples" in data:
            rows.extend(data["examples"])
        if "negative_examples" in data:
            rows.extend(data["negative_examples"])
        if rows:
            return rows
        # Single object — wrap in list
        return [data]

    raise ValueError(f"JSON must be a list of objects or a dict with 'examples' key, got {type(data).__name__}")


def _load_csv(path: Path) -> list[dict]:
    """Load rows from a CSV file."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _parse_rows(rows: list[dict], source: str) -> TrainingDataset:
    """Parse rows into TrainingDataset, auto-inferring format per row."""
    examples = []
    negative_examples = []
    errors = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Row {i}: expected dict, got {type(row).__name__}")
            continue

        keys = set(row.keys())

        # Check if this is a negative example
        if NEGATIVE_REQUIRED_COLUMNS.issubset(keys):
            try:
                negative_examples.append(NegativeTrainingExample(
                    sample_prompt=str(row["sample_prompt"]),
                    input=str(row["input"]),
                    bad_output=str(row["bad_output"]),
                    reason_why_bad=str(row["reason_why_bad"]),
                ))
            except Exception as e:
                errors.append(f"Row {i}: invalid negative example: {e}")

        # Check if this is a positive example
        elif POSITIVE_REQUIRED_COLUMNS.issubset(keys):
            try:
                metadata = row.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                examples.append(TrainingExample(
                    input=str(row["input"]),
                    expected_output=str(row["expected_output"]),
                    metadata=metadata if isinstance(metadata, dict) else {},
                ))
            except Exception as e:
                errors.append(f"Row {i}: invalid positive example: {e}")

        else:
            missing_pos = POSITIVE_REQUIRED_COLUMNS - keys
            missing_neg = NEGATIVE_REQUIRED_COLUMNS - keys
            errors.append(
                f"Row {i}: cannot infer format. "
                f"For positive examples, missing: {missing_pos}. "
                f"For negative examples, missing: {missing_neg}. "
                f"Got columns: {keys}"
            )

    if not examples and not negative_examples:
        error_detail = "\n".join(errors[:5]) if errors else "No valid rows found"
        raise ValueError(
            f"No valid examples found in {source}.\n"
            f"Expected columns — positive: {POSITIVE_REQUIRED_COLUMNS}, "
            f"negative: {NEGATIVE_REQUIRED_COLUMNS}.\n"
            f"Errors:\n{error_detail}"
        )

    return TrainingDataset(examples=examples, negative_examples=negative_examples)
