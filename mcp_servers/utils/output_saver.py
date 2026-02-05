"""Persist optimization results."""

import json
from datetime import datetime
from pathlib import Path


def save_optimization_result(result, output_dir: str = "outputs", name: str = None) -> dict:
    """Save result to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = name or f"optimized_{timestamp}"
    
    prompt_path = output_dir / f"{base_name}.txt"
    with open(prompt_path, "w") as f:
        f.write(result.best_prompt)
    
    json_path = output_dir / f"{base_name}.json"
    with open(json_path, "w") as f:
        json.dump({"best_prompt": result.best_prompt, "best_score": result.best_score}, f, indent=2)
    
    return {"prompt": prompt_path, "json": json_path}
