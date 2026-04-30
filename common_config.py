from __future__ import annotations

import os
import re
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import yaml

CLIENT_CONFIG_EXCLUDE_KEYS = ("name", "api_key", "base_url")


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
    except (OSError, yaml.YAMLError, TypeError, ValueError) as e:
        print(f"Error loading config from {path}: {e}")
    return None


def get_client_params(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params = {
        "api_key": config.get("api_key"),
        "base_url": config.get("base_url"),
    }
    chat_kwargs = {k: v for k, v in config.items() if k not in CLIENT_CONFIG_EXCLUDE_KEYS}
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


def build_default_result_path(
    model_id: str,
    dataset_path: str,
    save_to: Optional[str] = None,
) -> Path:
    if save_to:
        return Path(save_to)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = str(model_id).replace("/", "_")
    test_set_slug = Path(dataset_path).stem
    return Path(f"result/{model_slug}/{test_set_slug}/{timestamp}.json")


def load_simpleqa_dataset(path: str, num_tasks: Optional[int] = None) -> list[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if num_tasks:
        rows = rows[:num_tasks]
    normalized: list[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item_id = row.get("original_index") or row.get("id") or str(idx)
        normalized.append(
            {
                "id": str(item_id).strip(),
                "query": row.get("problem", ""),
                "gold_answer": row.get("answer", ""),
                "topic": row.get("topic"),
                "answer_type": row.get("answer_type"),
                "multi_step": row.get("multi_step"),
                "requires_reasoning": row.get("requires_reasoning"),
                "urls": row.get("urls"),
            }
        )
    return normalized


def load_simpleqa_output(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"results": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"results": []}
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data
    return {"results": []}


def read_existing_generated_ids_simpleqa(path: str | Path) -> set[str]:
    payload = load_simpleqa_output(path)
    ids: set[str] = set()
    for item in payload.get("results", []):
        if isinstance(item, dict) and item.get("id") is not None and item.get("llm_answer"):
            ids.add(str(item.get("id")))
    return ids


def upsert_simpleqa_results(existing: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index_by_id: dict[str, int] = {}
    for idx, item in enumerate(existing):
        if isinstance(item, dict) and item.get("id") is not None:
            index_by_id[str(item.get("id"))] = idx
    for item in updates:
        sid = str(item.get("id"))
        if sid in index_by_id:
            existing[index_by_id[sid]] = item
        else:
            index_by_id[sid] = len(existing)
            existing.append(item)
    return existing


def strip_sensitive_config(
    config: Optional[Dict[str, Any]],
    sensitive_keys: tuple[str, ...] = ("api_key",),
) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    return {k: v for k, v in config.items() if k not in sensitive_keys}


def build_simpleqa_output_payload(
    model_config: Optional[Dict[str, Any]],
    results: list[dict[str, Any]],
    base_payload: Optional[Dict[str, Any]] = None,
    evaluator_config: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_config": model_config or {},
    }

    if evaluator_config is not None:
        payload["evaluator_config"] = evaluator_config
    elif isinstance(base_payload, dict) and isinstance(base_payload.get("evaluator_config"), dict):
        payload["evaluator_config"] = base_payload["evaluator_config"]

    if summary is not None:
        payload["summary"] = summary
    elif isinstance(base_payload, dict) and isinstance(base_payload.get("summary"), dict):
        payload["summary"] = base_payload["summary"]

    payload["results"] = results
    return payload


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def extract_text_from_file_content(content_obj: Any) -> str:
    text_attr = getattr(content_obj, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if callable(text_attr):
        return text_attr()
    read_method = getattr(content_obj, "read", None)
    if callable(read_method):
        raw = read_method()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)
    return str(content_obj)


def coerce_token_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.lstrip("+-").isdigit():
        return int(value)
    return default


def custom_id_for_key(key: int) -> str:
    return f"key-{key}"


def parse_key_from_custom_id(custom_id: str) -> Optional[int]:
    if not isinstance(custom_id, str) or not custom_id.startswith("key-"):
        return None
    rest = custom_id[len("key-") :]
    if not rest:
        return None
    try:
        return int(rest, 10)
    except ValueError:
        return None


def parse_batch_output(
    output_text: str,
    key_payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    key_results: dict[str, dict[str, Any]] = {}
    failed: set[str] = set()

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue

        k_int = parse_key_from_custom_id(data.get("custom_id", ""))
        if k_int is None:
            continue
        sk = str(k_int)
        if sk not in key_payloads:
            continue

        parsed = _parse_batch_success_payload(data)
        if parsed is None:
            failed.add(sk)
            continue
        key_results[sk] = parsed
    return key_results, failed


def _parse_batch_success_payload(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    err = data.get("error")
    response = data.get("response") or {}
    status_code = response.get("status_code")
    body = response.get("body") or {}
    if err is not None or status_code not in {0, 200}:
        return None

    choices = body.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        return None

    usage = body.get("usage") or {}
    prompt_tokens = coerce_token_int(usage.get("prompt_tokens"), 0)
    completion_tokens = coerce_token_int(usage.get("completion_tokens"), 0)
    total_tokens = coerce_token_int(usage.get("total_tokens"), 0) or (prompt_tokens + completion_tokens)
    return {
        "response": content,
        "total_tokens": int(total_tokens),
    }
