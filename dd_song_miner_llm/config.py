from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
    },
    "asr": {
        "model": "small",
        "device": "auto",
        "compute_type": "default",
        "language": None,
        "beam_size": 5,
        "vad_filter": True,
        "initial_prompt": None,
    },
    "llm": {
        "api_key": None,
        "api_key_env": None,
        "base_url": None,
        "model": "gpt-4o",
        "temperature": 0.3,
        "max_tokens": 4096,
        "max_completion_tokens": None,
        "retry_empty_with_reasoning": True,
        "reasoning_followup_rounds": 2,
        "reasoning_followup_max_tokens": 8192,
        "batch_size": None,
        "fallbacks": [],
    },
    "padding": {
        "before_seconds": 3.0,
        "after_seconds": 5.0,
        "min_song_seconds": 15.0,
        "merge_gap_seconds": 30.0,
    },
    "output": {
        "video_clips": True,
        "audio_segments": True,
        "audio_extension": "m4a",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install PyYAML") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return deep_merge(DEFAULT_CONFIG, loaded)
