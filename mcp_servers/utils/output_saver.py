"""
Output Saver - Persist optimization results to disk.

Saves:
- Best prompt as plain text (.txt)
- Full optimization result as JSON (.json)
- Optional: iteration history for analysis
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _make_serializable(obj: Any) -> Any:
    """Convert dataclass/objects to JSON-serializable dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _make_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    return obj


def save_optimization_result(
    result: Any,
    output_dir: str | Path = "outputs",
    name: str | None = None,
    include_history: bool = True,
) -> dict[str, Path]:
    """
    Save optimization result to disk.

    Args:
        result: OptimizationResult from optimize_prompt()
        output_dir: Directory to save outputs (created if needed)
        name: Optional name for the output files (defaults to timestamp)
        include_history: Whether to include full iteration history in JSON

    Returns:
        Dict with paths to saved files: {"prompt": Path, "json": Path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = name or f"optimized_{timestamp}"

    # Save best prompt as plain text
    prompt_path = output_dir / f"{base_name}.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(result.best_prompt)

    # Build JSON output
    json_data = {
        "best_prompt": result.best_prompt,
        "best_score": result.best_score,
        "iterations": result.iterations,
        "converged": result.converged,
        "convergence_reason": result.convergence_reason,
        "saved_at": datetime.now().isoformat(),
    }

    if include_history and hasattr(result, "history"):
        json_data["history"] = [
            {
                "iteration": h.iteration,
                "best_score": h.best_score,
                "sample_scores": h.sample_scores,
                "candidates_generated": h.candidates_generated,
                "candidates_evaluated": h.candidates_evaluated,
            }
            for h in result.history
        ]

    # Save full result as JSON
    json_path = output_dir / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return {"prompt": prompt_path, "json": json_path}


def load_prompt(path: str | Path) -> str:
    """Load a saved prompt from file."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_optimization_result(path: str | Path) -> dict:
    """Load a saved optimization result JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_saved_outputs(output_dir: str | Path = "outputs") -> list[dict]:
    """
    List all saved optimization outputs.

    Returns:
        List of dicts with: name, prompt_path, json_path, saved_at
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    outputs = []
    for json_file in sorted(output_dir.glob("*.json"), reverse=True):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            outputs.append(
                {
                    "name": json_file.stem,
                    "prompt_path": output_dir / f"{json_file.stem}.txt",
                    "json_path": json_file,
                    "score": data.get("best_score"),
                    "saved_at": data.get("saved_at"),
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue

    return outputs
