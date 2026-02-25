import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """
    Lightweight config loader.
    Accepts JSON files (recommended for dependency-free operation).
    """
    p = Path(path)
    with open(p, "r") as f:
        return json.load(f)


def save_json(payload: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)

