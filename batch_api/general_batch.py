"""SimpleQA batch generation via OpenAI-compatible Batch API (chat completions).

Unified entrypoint for Moonshot (Kimi) and Alibaba DashScope (Qwen).
Provider is auto-detected from the model id keyword ('kimi' or 'qwen').
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_src = REPO_ROOT / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from common_config import (
    build_default_result_path,
    build_simpleqa_output_payload,
    custom_id_for_key,
    ensure_parent_dir,
    extract_text_from_file_content,
    load_json,
    load_simpleqa_dataset,
    load_simpleqa_output,
    load_yaml_config,
    make_chat_completion_create_kwargs,
    parse_batch_output,
    sanitize_path_component,
    read_existing_generated_ids_simpleqa,
    save_json,
    save_text,
    strip_sensitive_config,
    upsert_simpleqa_results,
)
from prompts import QUERY_TEMPLATE  # noqa: E402

TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}

# Per-provider settings. Provider is selected by a keyword in model_id.
PROVIDER_MOONSHOT = "moonshot"
PROVIDER_QWEN = "qwen"

KIMI_BATCH_FORBIDDEN_PARAMS = {
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "n",
    "presence_penalty",
    "frequency_penalty",
}

PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    PROVIDER_MOONSHOT: {
        "artifacts_dir": "batch_api/moonshot/artifacts",
        "max_tasks_per_batch": 1000,
    },
    PROVIDER_QWEN: {
        "artifacts_dir": "batch_api/qwen/artifacts",
        "max_tasks_per_batch": 5000,
    },
}


def detect_provider(model_id: str) -> str:
    """Pick provider branch from a keyword in the model id."""
    m = str(model_id).lower()
    if "kimi" in m:
        return PROVIDER_MOONSHOT
    if "qwen" in m:
        return PROVIDER_QWEN
    raise SystemExit(
        f"cannot detect provider from model id {model_id!r}: "
        "expected the id to contain 'kimi' or 'qwen'"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IFBench: generate with Batch API; output JSONL for run_eval.py evaluation."
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "prepare", "upload", "create", "wait", "collect", "submit", "poll"],
        help="Pipeline step. Full flow: prepare / upload / create / wait / collect.",
    )
    parser.add_argument(
        "--model-id", type=str, default="kimi-k2.6", help="Model id in models.yaml (name field)"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="dataset/simpleqa_verified.csv",
        help="SimpleQA CSV path.",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        default=None,
        help="Output JSONL path. Default matches run_eval: generation/<model-id>/<timestamp>.jsonl",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="If set, only the first k rows of the parquet are considered.",
    )
    parser.add_argument(
        "--models-yaml", type=str, default="models.yaml", help="Path to models.yaml (repo root or absolute)."
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help="Batch completion window (e.g. 24h). Range 24h-336h.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=10,
        help="Interval between batch status polls.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory; required for upload / create / wait / collect (after prepare).",
    )
    parser.add_argument(
        "--batch-id", type=str, default=None, help="Override batch id for wait / collect if needed."
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="For split steps: which chunk 0,1,...(required if run has >1 chunk). Omitted = 0 when 1 chunk.",
    )
    return parser.parse_args()


def _models_yaml_path(args: argparse.Namespace) -> str:
    p = args.models_yaml
    if not Path(p).is_file() and (REPO_ROOT / p).is_file():
        return str(REPO_ROOT / p)
    return p


def _input_csv_path(args: argparse.Namespace) -> str:
    p = args.input_csv
    if not Path(p).is_file() and (REPO_ROOT / p).is_file():
        return str(REPO_ROOT / p)
    return p


def _resolve_responses_path(args: argparse.Namespace) -> str:
    p = str(
        build_default_result_path(
            model_id=str(args.model_id),
            dataset_path=_input_csv_path(args),
            save_to=args.save_to,
        ).resolve()
    )
    ensure_parent_dir(p)
    return p


def merge_create_kwargs_to_batch_body(create_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map OpenAI client kwargs to a flat JSON body for the Batch /v1/chat/completions call."""
    body = {k: v for k, v in create_kwargs.items() if k != "extra_body"}
    extra = create_kwargs.get("extra_body")
    if isinstance(extra, dict):
        body = {**body, **extra}
    return body


def build_batch_request_body(
    model_id_for_filter: str,
    create_kwargs: dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = dict(merge_create_kwargs_to_batch_body(create_kwargs))
    if model_id_for_filter in {"kimi-k2.5", "kimi-k2.6"}:
        for k in KIMI_BATCH_FORBIDDEN_PARAMS:
            body.pop(k, None)
    return body


def validate_completion_window(value: str, *, enforce_openai_range: bool = True) -> str:
    """Validate completion_window: integer + 'h' or 'd'; optional OpenAI batch 24h–336h rule."""
    s = str(value).strip().lower()
    m = re.fullmatch(r"(\d+)([hd])", s)
    if not m:
        raise ValueError(
            "completion_window must be an integer followed by 'h' or 'd' (e.g. 24h, 14d)"
        )
    if enforce_openai_range:
        amount = int(m.group(1))
        unit = m.group(2)
        hours = amount if unit == "h" else amount * 24
        if not (24 <= hours <= 336):
            raise ValueError("completion_window must be between 24h and 336h")
    return s


def normalize_pipeline_step(step: str) -> str:
    if step == "submit":
        return "upload"
    if step == "poll":
        return "wait"
    return step


def iter_simpleqa_batch_items(
    args: argparse.Namespace,
    responses_output: str,
) -> tuple[list[tuple[int, dict[str, Any], list[dict[str, Any]]]], int]:
    """Load dataset rows pending generation. Returns (items, resume_skipped)."""
    items: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    dataset = load_simpleqa_dataset(_input_csv_path(args), args.num_tasks)
    existing_ids = read_existing_generated_ids_simpleqa(responses_output)
    resume_skipped = sum(1 for row in dataset if row["id"] in existing_ids)
    if existing_ids:
        print(
            f"resume: {len(existing_ids)} id(s) already in {responses_output} will be skipped"
        )
    for key, row in enumerate(dataset):
        if row["id"] in existing_ids:
            continue
        prompt = str(QUERY_TEMPLATE.format(question=row["query"]))
        messages = [{"role": "user", "content": prompt}]
        items.append((int(key), row, messages))
    return items, resume_skipped


def build_job_batch_body(
    model_cfg: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ark Job batch JSONL body (custom_id + body); model comes from ModelReference."""
    create_kwargs = make_chat_completion_create_kwargs(model_cfg, messages)
    body = merge_create_kwargs_to_batch_body(create_kwargs)
    body.pop("model", None)
    return body


def build_batch_input_file_simpleqa_job(
    items: list[tuple[int, dict[str, Any], list[dict[str, Any]]]],
    model_cfg: dict[str, Any],
    input_path: Path,
) -> dict[str, dict[str, Any]]:
    """Write Ark Job batch input JSONL: {custom_id, body} per line."""
    key_payloads: dict[str, dict[str, Any]] = {}
    with open(input_path, "w", encoding="utf-8", newline="\n") as f:
        for key, payload, messages in items:
            sk = str(int(key))
            body = build_job_batch_body(model_cfg, messages)
            request_obj: dict[str, Any] = {
                "custom_id": custom_id_for_key(int(key)),
                "body": body,
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            key_payloads[sk] = {
                "key": int(key),
                "id": str(payload.get("id", "")),
                "query": str(payload.get("query", "")),
                "gold_answer": str(payload.get("gold_answer", "")),
                "topic": payload.get("topic"),
                "answer_type": payload.get("answer_type"),
                "multi_step": payload.get("multi_step"),
                "requires_reasoning": payload.get("requires_reasoning"),
                "urls": payload.get("urls"),
            }
    return key_payloads


def apply_collect_results(
    metadata: dict[str, Any],
    chunk_i: int,
    key_payloads: dict[str, dict[str, Any]],
    key_results: dict[str, dict[str, Any]],
    submitted: set[str],
) -> tuple[int, int, int]:
    """Merge batch output into save_to JSON; update meta. Returns (written, not_returned, empty_skipped)."""
    not_returned = submitted - set(key_results.keys())
    success_keys_last_chunk: list[str] = []
    to_write: list[dict[str, Any]] = []
    empty_response_skipped: list[str] = []
    for sk, res in key_results.items():
        if sk not in submitted:
            continue
        text = res.get("response", "")
        if not isinstance(text, str) or text.strip() == "":
            empty_response_skipped.append(sk)
            continue
        success_keys_last_chunk.append(sk)
        meta_row = key_payloads[sk]
        to_write.append(
            {
                "id": str(meta_row.get("id", "")),
                "query": str(meta_row.get("query", "")),
                "llm_answer": text,
                "tokens": int(res.get("tokens") or 0),
                "gold_answer": str(meta_row.get("gold_answer", "")),
                "topic": meta_row.get("topic"),
                "answer_type": meta_row.get("answer_type"),
                "multi_step": meta_row.get("multi_step"),
                "requires_reasoning": meta_row.get("requires_reasoning"),
                "urls": meta_row.get("urls"),
            }
        )

    results_path = str(metadata["save_to"])
    if to_write:
        payload = load_simpleqa_output(results_path)
        existing = payload.get("results", [])
        if not isinstance(existing, list):
            existing = []
        merged_results = upsert_simpleqa_results(existing, to_write)
        ordered_payload = build_simpleqa_output_payload(
            model_config=metadata.get("output_model_config", {}),
            results=merged_results,
            base_payload=payload if isinstance(payload, dict) else None,
        )
        save_json(Path(results_path), ordered_payload)

    done = set(metadata.get("completed_chunk_indices", []))
    done.add(chunk_i)
    metadata["completed_chunk_indices"] = sorted(done)
    metadata["success_keys_last_chunk"] = sorted(int(x) for x in success_keys_last_chunk)
    metadata["not_returned_in_output_keys"] = sorted(int(x) for x in not_returned)
    metadata["empty_response_skipped_keys"] = sorted(int(x) for x in empty_response_skipped)
    return len(to_write), len(not_returned), len(empty_response_skipped)


def make_client(model_id: str, models_yaml: str) -> OpenAI:
    cfg = load_yaml_config(models_yaml, model_id)
    if not cfg:
        raise ValueError(f"model not found in {models_yaml}: {model_id}")
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def build_run_dir(artifacts_dir: Path, model_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = sanitize_path_component(str(model_id))
    run_dir = Path(artifacts_dir) / safe_model / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_batch_input_file(
    items: list[tuple[int, str, list[dict[str, Any]]]],
    model_id: str,
    model_cfg: dict[str, Any],
    input_path: Path,
) -> dict[str, dict[str, Any]]:
    """items: (key, prompt, messages) per row. key_payloads keyed by str(key)."""
    key_payloads: dict[str, dict[str, Any]] = {}

    with open(input_path, "w", encoding="utf-8", newline="\n") as f:
        for key, prompt, messages in items:
            sk = str(int(key))
            create_kwargs = make_chat_completion_create_kwargs(model_cfg, messages)
            body = build_batch_request_body(model_id, create_kwargs)
            custom_id = custom_id_for_key(int(key))
            request_obj: dict[str, Any] = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            key_payloads[sk] = {
                "key": int(key),
                "prompt": str(prompt),
            }
    return key_payloads


def build_batch_input_file_simpleqa(
    items: list[tuple[int, dict[str, Any], list[dict[str, Any]]]],
    model_id: str,
    model_cfg: dict[str, Any],
    input_path: Path,
) -> dict[str, dict[str, Any]]:
    key_payloads: dict[str, dict[str, Any]] = {}
    with open(input_path, "w", encoding="utf-8", newline="\n") as f:
        for key, payload, messages in items:
            sk = str(int(key))
            create_kwargs = make_chat_completion_create_kwargs(model_cfg, messages)
            body = build_batch_request_body(model_id, create_kwargs)
            custom_id = custom_id_for_key(int(key))
            request_obj: dict[str, Any] = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            key_payloads[sk] = {
                "key": int(key),
                "id": str(payload.get("id", "")),
                "query": str(payload.get("query", "")),
                "gold_answer": str(payload.get("gold_answer", "")),
                "topic": payload.get("topic"),
                "answer_type": payload.get("answer_type"),
                "multi_step": payload.get("multi_step"),
                "requires_reasoning": payload.get("requires_reasoning"),
                "urls": payload.get("urls"),
            }
    return key_payloads


def upload_input_file(client: OpenAI, input_path: Path) -> str:
    with open(input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"input file uploaded: {file_obj.id}")
    return file_obj.id


def create_batch(client: OpenAI, input_file_id: str, completion_window: str) -> str:
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    print(f"batch job created: {batch.id}")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str, poll_interval_seconds: int) -> Any:
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = getattr(batch, "request_counts", None)
        completed = getattr(counts, "completed", 0) if counts else 0
        total = getattr(counts, "total", 0) if counts else 0
        print(f"status: {batch.status} ({completed}/{total})")

        if batch.status in TERMINAL_BATCH_STATES:
            return batch
        time.sleep(max(1, int(poll_interval_seconds)))


def stage_prepare(args: argparse.Namespace) -> Path:
    model_id = str(args.model_id)
    provider = detect_provider(model_id)
    defaults = PROVIDER_DEFAULTS[provider]
    models_yaml = _models_yaml_path(args)
    responses_output = _resolve_responses_path(args)

    model_cfg = load_yaml_config(models_yaml, model_id)
    if not model_cfg:
        raise ValueError(f"model not found in {models_yaml}: {model_id}")
    ensure_parent_dir(responses_output)

    items, resume_skipped = iter_simpleqa_batch_items(args, responses_output)

    if not items:
        print("all tasks are already in output; nothing to do.")
        raise SystemExit(0)

    max_per = int(defaults["max_tasks_per_batch"])
    if max_per < 1:
        raise ValueError("max_tasks_per_batch must be at least 1")

    artifacts_root = Path(str(defaults["artifacts_dir"]))
    if not artifacts_root.is_absolute():
        artifacts_root = REPO_ROOT / artifacts_root
    run_dir = build_run_dir(artifacts_root, model_id)
    meta_json = run_dir / "meta.json"

    chunks: list[dict[str, Any]] = []
    all_submitted: list[int] = []
    for off in range(0, len(items), max_per):
        chunk_idx = len(chunks)
        chunk_items = items[off : off + max_per]
        input_jsonl = run_dir / f"batch_input_c{chunk_idx:03d}.jsonl"
        output_jsonl = run_dir / f"batch_output_c{chunk_idx:03d}.jsonl"
        error_jsonl = run_dir / f"batch_error_c{chunk_idx:03d}.jsonl"
        key_payloads_path = run_dir / f"key_payloads_c{chunk_idx:03d}.json"
        key_payloads = build_batch_input_file_simpleqa(
            items=chunk_items,
            model_id=model_id,
            model_cfg=model_cfg,
            input_path=input_jsonl,
        )
        submitted = sorted(int(k) for k in key_payloads.keys())
        all_submitted.extend(submitted)
        save_json(
            key_payloads_path,
            {str(k): key_payloads[str(k)] for k in submitted},
        )
        chunks.append(
            {
                "index": chunk_idx,
                "input_jsonl": str(input_jsonl),
                "output_jsonl": str(output_jsonl),
                "error_jsonl": str(error_jsonl),
                "key_payloads_json": str(key_payloads_path),
                "submitted_keys": submitted,
                "input_file_id": None,
                "batch_id": None,
                "batch_status": None,
            }
        )

    input_csv_resolved = str(Path(_input_csv_path(args)).resolve())
    metadata: dict[str, Any] = {
        "version": 3,
        "task_type": "simpleqa_verified",
        "provider": provider,
        "model": model_id,
        "input_csv": input_csv_resolved,
        "save_to": responses_output,
        "models_yaml": str(Path(models_yaml).resolve()) if Path(models_yaml).is_file() else models_yaml,
        "num_tasks": args.num_tasks,
        "max_tasks_per_batch": max_per,
        "run_dir": str(run_dir),
        "chunks": chunks,
        "chunk_count": len(chunks),
        "completed_chunk_indices": [],
        "completion_window": str(args.completion_window),
        "poll_interval_seconds": int(args.poll_interval_seconds),
        "submitted_keys": sorted(set(all_submitted)),
        "resume_skipped": resume_skipped,
        "output_model_config": strip_sensitive_config(model_cfg),
    }
    save_json(meta_json, metadata)
    n_ch = len(chunks)
    print(
        f"prepare done, run_dir={run_dir} "
        f"({n_ch} batch file(s), {max_per} lines max per file, "
        f"{len(set(all_submitted))} key(s) total, "
        f"resume skipped in benchmark slice: {metadata['resume_skipped']})"
    )
    print(
        f"chunk index for --chunk-index / split steps: 0 .. {n_ch - 1} "
        f"(files: batch_input_c000.jsonl, ...; see meta.json -> chunk_count, chunks)"
    )
    return run_dir


def load_meta_or_fail(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return meta_path, load_json(meta_path)


def _chunk_count(metadata: dict[str, Any]) -> int:
    if metadata.get("chunks") and isinstance(metadata["chunks"], list) and len(metadata["chunks"]) > 0:
        return len(metadata["chunks"])
    return 1


def _resolve_chunk_index(metadata: dict[str, Any], args: argparse.Namespace) -> int:
    n = _chunk_count(metadata)
    ci = args.chunk_index
    if ci is not None:
        if not (0 <= ci < n):
            raise ValueError(f"--chunk-index must be in [0, {n - 1}] (this run has {n} chunk(s))")
        return int(ci)
    if n > 1:
        raise SystemExit(
            f"this run has {n} batch chunks; pass e.g. --chunk-index 0, then 1, ... for upload/create/wait/collect, "
            "or use --step all to run every chunk in order"
        )
    return 0


def _mut_chunk(metadata: dict[str, Any], chunk_index: int) -> dict[str, Any]:
    """Object to set input_file_id / batch_id on (same ref as in meta)."""
    if metadata.get("chunks") and len(metadata.get("chunks", [])) > chunk_index:
        return metadata["chunks"][chunk_index]
    if chunk_index != 0:
        raise ValueError(f"only chunk 0 in legacy (v2) meta, got {chunk_index}")
    return metadata


def stage_upload(args: argparse.Namespace, run_dir: Path) -> None:
    meta_path, metadata = load_meta_or_fail(run_dir)
    chunk_i = _resolve_chunk_index(metadata, args)
    ch = _mut_chunk(metadata, chunk_i)
    model_id = str(metadata["model"])
    models_yaml = _models_yaml_path(args)
    input_jsonl = Path(ch["input_jsonl"])
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"batch input missing: {input_jsonl}")
    client = make_client(model_id, models_yaml)
    input_file_id = upload_input_file(client, input_jsonl)
    ch["input_file_id"] = input_file_id
    ch["batch_id"] = None
    if "batch_status" in ch:
        ch["batch_status"] = None
    save_json(meta_path, metadata)
    print(f"upload done, chunk={chunk_i}, input_file_id={input_file_id}")


def stage_create(args: argparse.Namespace, run_dir: Path) -> str:
    meta_path, metadata = load_meta_or_fail(run_dir)
    chunk_i = _resolve_chunk_index(metadata, args)
    ch = _mut_chunk(metadata, chunk_i)
    model_id = str(metadata["model"])
    models_yaml = _models_yaml_path(args)
    input_file_id = ch.get("input_file_id")
    if not input_file_id:
        raise ValueError("input_file_id missing; run upload first for this chunk.")
    completion_window = validate_completion_window(
        str(args.completion_window or metadata.get("completion_window") or "24h")
    )
    client = make_client(model_id, models_yaml)
    batch_id = create_batch(client, str(input_file_id), completion_window=completion_window)
    ch["batch_id"] = batch_id
    ch["batch_status"] = "validating"
    metadata["completion_window"] = completion_window
    save_json(meta_path, metadata)
    print(f"create done, chunk={chunk_i}, batch_id={batch_id}")
    return batch_id


def stage_wait(args: argparse.Namespace, run_dir: Path) -> Any:
    meta_path, metadata = load_meta_or_fail(run_dir)
    chunk_i = _resolve_chunk_index(metadata, args)
    ch = _mut_chunk(metadata, chunk_i)
    model_id = str(metadata["model"])
    models_yaml = _models_yaml_path(args)
    batch_id = args.batch_id or ch.get("batch_id")
    if not batch_id:
        raise ValueError("batch_id missing; run create first or set --batch-id for this chunk")
    interval = int(args.poll_interval_seconds or metadata.get("poll_interval_seconds") or 10)
    client = make_client(model_id, models_yaml)
    batch = poll_batch(client, str(batch_id), poll_interval_seconds=interval)
    ch["batch_id"] = str(batch_id)
    ch["batch_status"] = batch.status
    save_json(meta_path, metadata)
    print(f"wait done, chunk={chunk_i}, final status: {batch.status}")
    return batch


def _append_generation_lines(path: str, lines: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


def stage_collect(
    args: argparse.Namespace, run_dir: Path, batch_obj: Any | None = None
) -> None:
    meta_path, metadata = load_meta_or_fail(run_dir)
    chunk_i = _resolve_chunk_index(metadata, args)
    ch = _mut_chunk(metadata, chunk_i)
    model_id = str(metadata["model"])
    models_yaml = _models_yaml_path(args)
    batch_id = args.batch_id or ch.get("batch_id")
    if not batch_id:
        raise ValueError("batch_id missing; use create or --batch-id for this chunk")

    client = make_client(model_id, models_yaml)
    batch = batch_obj if batch_obj is not None else client.batches.retrieve(str(batch_id))
    ch["batch_id"] = str(batch_id)
    ch["batch_status"] = batch.status

    if batch.status != "completed":
        save_json(meta_path, metadata)
        raise RuntimeError(f"batch not completed, status={batch.status}")

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        save_json(meta_path, metadata)
        raise RuntimeError("batch completed but output_file_id is empty")

    output_jsonl = Path(ch["output_jsonl"])
    error_jsonl = Path(ch["error_jsonl"])
    results_path = str(metadata["save_to"])
    key_payloads_path = Path(ch["key_payloads_json"])
    if not key_payloads_path.is_file():
        raise FileNotFoundError(f"key_payloads for chunk: {key_payloads_path}")
    key_payloads = load_json(key_payloads_path)
    for k in list(key_payloads.keys()):
        key_payloads[k]["key"] = int(key_payloads[k]["key"])

    output_content = client.files.content(output_file_id)
    output_text = extract_text_from_file_content(output_content)
    save_text(output_jsonl, output_text)
    ch["output_file_id"] = output_file_id

    err_file = getattr(batch, "error_file_id", None)
    if err_file:
        err_c = client.files.content(err_file)
        save_text(error_jsonl, extract_text_from_file_content(err_c))
        ch["error_file_id"] = err_file
    else:
        ch["error_file_id"] = None

    key_results, _ = parse_batch_output(output_text, key_payloads)
    ch_sub = ch.get("submitted_keys", [])
    submitted = {str(x) for x in ch_sub} or set(key_payloads.keys())

    written, not_returned, empty_response_skipped = apply_collect_results(
        metadata, chunk_i, key_payloads, key_results, submitted
    )
    ntot = _chunk_count(metadata)
    save_json(meta_path, metadata)
    print(f"updated generation JSON: {results_path} (chunk {chunk_i} / {ntot})")
    print(f"evaluate next: python main.py --evaluate-file \"{results_path}\" --evaluator deepseek-v4-flash")
    print(
        f"this collect: {written} line(s) written; "
        f"not returned in batch output: {len(not_returned)}; "
        f"empty response skipped: {len(empty_response_skipped)}"
    )
    print(f"run_dir={run_dir}")


def _run_dir_from_arg(run_dir: str) -> Path:
    p = Path(run_dir)
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


def main() -> None:
    args = parse_args()
    detect_provider(str(args.model_id))
    args.completion_window = validate_completion_window(str(args.completion_window))
    step = normalize_pipeline_step(str(args.step))

    if step in {"upload", "create", "wait", "collect"} and not args.run_dir:
        raise SystemExit(f"--step {step} requires --run-dir")

    if step == "prepare":
        stage_prepare(args)
        return
    if step == "upload":
        stage_upload(args, _run_dir_from_arg(args.run_dir))
        return
    if step == "create":
        stage_create(args, _run_dir_from_arg(args.run_dir))
        return
    if step == "wait":
        stage_wait(args, _run_dir_from_arg(args.run_dir))
        return
    if step == "collect":
        stage_collect(args, _run_dir_from_arg(args.run_dir), batch_obj=None)
        return

    run_path = stage_prepare(args)
    _meta_p, run_meta = load_meta_or_fail(run_path)
    n = _chunk_count(run_meta)
    done = set(run_meta.get("completed_chunk_indices", []))
    for i in range(n):
        if i in done:
            print(f"skipping chunk {i} (already completed per meta.json)")
            continue
        args.chunk_index = i
        print(f"--- batch chunk {i + 1}/{n} ---")
        stage_upload(args, run_path)
        stage_create(args, run_path)
        batch = stage_wait(args, run_path)
        stage_collect(args, run_path, batch_obj=batch)
    args.chunk_index = None
    if n > 1:
        print("all batch chunks for this run finished (see generation JSONL).")


if __name__ == "__main__":
    main()
