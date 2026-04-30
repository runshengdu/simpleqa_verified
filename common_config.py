from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


def _expand_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    if expanded.startswith("${") and expanded.endswith("}"):
        return os.environ.get(expanded[2:-1], "")
    return expanded


def load_yaml_config(path: str, model_name: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for model in data.get("models", []):
            if isinstance(model, dict) and model.get("name") == model_name:
                config = model.copy()
                config["api_key"] = _expand_env(config.get("api_key"))
                return config
    except Exception as e:
        print(f"Error loading config from {path}: {e}")
    return None


def get_client_params(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params = {
        "api_key": config.get("api_key"),
        "base_url": config.get("base_url"),
    }
    exclude_keys = {"name", "api_key", "base_url"}
    chat_kwargs = {k: v for k, v in config.items() if k not in exclude_keys}
    return params, chat_kwargs


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def sanitize_path_component(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")
    return text or "unknown"


def make_chat_completion_create_kwargs(
    model_cfg: Dict[str, Any],
    messages: list[dict[str, Any]],
) -> Dict[str, Any]:
    _, chat_kwargs = get_client_params(model_cfg)
    return {
        "model": model_cfg.get("name"),
        "messages": messages,
        **chat_kwargs,
    }
