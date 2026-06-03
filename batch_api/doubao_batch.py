"""SimpleQA batch generation for Volcengine Ark (Doubao) via Batch Job + TOS.

https://www.volcengine.com/docs/82379/1399517
https://www.volcengine.com/docs/82379/1339603
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common_config import (
    load_json,
    load_yaml_config,
    parse_batch_output,
    sanitize_path_component,
    save_json,
    save_text,
    strip_sensitive_config,
)
from batch_api.doubao_volc import (
    DEFAULT_PROJECT_NAME,
    DEFAULT_TOS_BUCKET,
    build_model_reference,
    create_batch_inference_job,
    download_results_jsonl,
    get_batch_job,
    make_ark_api,
    make_tos_client,
    poll_batch_job,
    upload_file_to_tos,
)
from batch_api.general_batch import (
    REPO_ROOT,
    _chunk_count,
    _input_csv_path,
    _models_yaml_path,
    _mut_chunk,
    _resolve_chunk_index,
    _resolve_responses_path,
    _run_dir_from_arg,
    apply_collect_results,
    build_batch_input_file_simpleqa_job,
    build_run_dir,
    iter_simpleqa_batch_items,
    load_meta_or_fail,
    normalize_pipeline_step,
    validate_completion_window,
)

PROVIDER_DOUBAO = "doubao"

DOUBAO_DEFAULTS: dict[str, Any] = {
    "artifacts_dir": "batch_api/doubao/artifacts",
    "max_tasks_per_batch": 5000,
    "tos_input_prefix": "batch-inference-job/dataset",
    "tos_output_prefix": "batch-inference-job/output",
}

TOS_BUCKET = DEFAULT_TOS_BUCKET
ARK_PROJECT_NAME = DEFAULT_PROJECT_NAME


def validate_doubao_model(model_id: str) -> None:
    m = str(model_id).lower()
    if "doubao" in m or m.startswith("ep-"):
        return
    raise SystemExit(
        f"model id {model_id!r} is not a Doubao/Ark model: "
        "expected 'doubao' in the name or an 'ep-' endpoint id"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SimpleQA: Doubao batch inference (Ark Job API + TOS)."
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "prepare", "upload", "create", "wait", "collect", "submit", "poll"],
    )
    parser.add_argument("--model-id", type=str, default="doubao-seed-2-0-pro-260215")
    parser.add_argument("--input-csv", type=str, default="dataset/simpleqa_verified.csv")
    parser.add_argument("--save-to", type=str, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--models-yaml", type=str, default="models.yaml")
    parser.add_argument("--completion-window", type=str, default="1d")
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--batch-id", type=str, default=None)
    parser.add_argument("--chunk-index", type=int, default=None)
    return parser.parse_args()


def _tos_dataset_prefix(run_dir: Path, safe_model: str) -> str:
    return f"{DOUBAO_DEFAULTS['tos_input_prefix']}/{safe_model}/{run_dir.name}"


def _tos_output_prefix(run_dir: Path, safe_model: str, chunk_idx: int) -> str:
    return f"{DOUBAO_DEFAULTS['tos_output_prefix']}/{safe_model}/{run_dir.name}/c{chunk_idx:03d}/"


def stage_prepare(args: argparse.Namespace) -> Path:
    model_id = str(args.model_id)
    validate_doubao_model(model_id)
    defaults = DOUBAO_DEFAULTS
    models_yaml = _models_yaml_path(args)
    responses_output = _resolve_responses_path(args)

    model_cfg = load_yaml_config(models_yaml, model_id)
    if not model_cfg:
        raise ValueError(f"model not found in {models_yaml}: {model_id}")

    items, resume_skipped = iter_simpleqa_batch_items(args, responses_output)
    if not items:
        print("all tasks are already in output; nothing to do.")
        raise SystemExit(0)

    max_per = int(defaults["max_tasks_per_batch"])
    artifacts_root = Path(str(defaults["artifacts_dir"]))
    if not artifacts_root.is_absolute():
        artifacts_root = REPO_ROOT / artifacts_root
    run_dir = build_run_dir(artifacts_root, model_id)
    safe_model = sanitize_path_component(model_id)
    tos_bucket = TOS_BUCKET

    chunks: list[dict[str, Any]] = []
    all_submitted: list[int] = []
    for off in range(0, len(items), max_per):
        chunk_idx = len(chunks)
        chunk_items = items[off : off + max_per]
        input_jsonl = run_dir / f"batch_input_c{chunk_idx:03d}.jsonl"
        output_jsonl = run_dir / f"batch_output_c{chunk_idx:03d}.jsonl"
        error_jsonl = run_dir / f"batch_error_c{chunk_idx:03d}.jsonl"
        key_payloads_path = run_dir / f"key_payloads_c{chunk_idx:03d}.json"
        key_payloads = build_batch_input_file_simpleqa_job(
            items=chunk_items,
            model_cfg=model_cfg,
            input_path=input_jsonl,
        )
        submitted = sorted(int(k) for k in key_payloads.keys())
        all_submitted.extend(submitted)
        save_json(key_payloads_path, {str(k): key_payloads[str(k)] for k in submitted})
        chunks.append(
            {
                "index": chunk_idx,
                "input_jsonl": str(input_jsonl),
                "output_jsonl": str(output_jsonl),
                "error_jsonl": str(error_jsonl),
                "key_payloads_json": str(key_payloads_path),
                "submitted_keys": submitted,
                "tos_bucket": tos_bucket,
                "tos_input_object_key": (
                    f"{_tos_dataset_prefix(run_dir, safe_model)}/batch_input_c{chunk_idx:03d}.jsonl"
                ),
                "tos_output_prefix": _tos_output_prefix(run_dir, safe_model, chunk_idx),
                "job_id": None,
                "job_phase": None,
            }
        )

    metadata: dict[str, Any] = {
        "version": 4,
        "task_type": "simpleqa_verified",
        "provider": PROVIDER_DOUBAO,
        "batch_mode": "job",
        "model": model_id,
        "input_csv": str(Path(_input_csv_path(args)).resolve()),
        "save_to": responses_output,
        "models_yaml": str(Path(models_yaml).resolve()) if Path(models_yaml).is_file() else models_yaml,
        "num_tasks": args.num_tasks,
        "max_tasks_per_batch": max_per,
        "tos_bucket": tos_bucket,
        "project_name": ARK_PROJECT_NAME,
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
    save_json(run_dir / "meta.json", metadata)
    n_ch = len(chunks)
    print(
        f"prepare done, run_dir={run_dir} ({n_ch} file(s), {max_per} lines max, "
        f"{len(set(all_submitted))} key(s), resume skipped: {resume_skipped})"
    )
    print(f"TOS bucket: {tos_bucket}")
    return run_dir


def stage_upload(args: argparse.Namespace, run_dir: Path) -> None:
    meta_path, metadata = load_meta_or_fail(run_dir)
    chunk_i = _resolve_chunk_index(metadata, args)
    ch = _mut_chunk(metadata, chunk_i)
    input_jsonl = Path(ch["input_jsonl"])
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"batch input missing: {input_jsonl}")

    bucket = TOS_BUCKET
    object_key = str(ch["tos_input_object_key"])
    upload_file_to_tos(make_tos_client(), bucket, object_key, input_jsonl)
    ch["tos_uploaded"] = True
    ch["job_id"] = None
    ch["job_phase"] = None
    save_json(meta_path, metadata)
    print(f"upload done, chunk={chunk_i}, tos://{bucket}/{object_key}")


def stage_create(args: argparse.Namespace, run_dir: Path) -> str:
    meta_path, metadata = load_meta_or_fail(run_dir)
    chunk_i = _resolve_chunk_index(metadata, args)
    ch = _mut_chunk(metadata, chunk_i)
    model_id = str(metadata["model"])
    model_cfg = load_yaml_config(_models_yaml_path(args), model_id)
    if not model_cfg:
        raise ValueError(f"model not found for {model_id!r}")

    if not ch.get("tos_uploaded"):
        stage_upload(args, run_dir)
        meta_path, metadata = load_meta_or_fail(run_dir)
        ch = _mut_chunk(metadata, chunk_i)

    completion_window = validate_completion_window(
        str(args.completion_window or metadata.get("completion_window") or "1d"),
        enforce_openai_range=False,
    )
    bucket = TOS_BUCKET

    job_id = create_batch_inference_job(
        make_ark_api(),
        name=f"simpleqa-{model_id}-c{chunk_i:03d}-{run_dir.name}"[:128],
        model_reference=build_model_reference(model_id, model_cfg),
        input_bucket=bucket,
        input_object_key=str(ch["tos_input_object_key"]),
        output_bucket=bucket,
        output_prefix=str(ch["tos_output_prefix"]),
        completion_window=completion_window,
        project_name=ARK_PROJECT_NAME,
        description=f"simpleqa chunk {chunk_i}",
    )
    ch["job_id"] = job_id
    ch["job_phase"] = "Queued"
    metadata["completion_window"] = completion_window
    save_json(meta_path, metadata)
    print(f"create done, chunk={chunk_i}, job_id={job_id}")
    return job_id


def stage_wait(args: argparse.Namespace, run_dir: Path) -> Any:
    meta_path, metadata = load_meta_or_fail(run_dir)
    chunk_i = _resolve_chunk_index(metadata, args)
    ch = _mut_chunk(metadata, chunk_i)
    job_id = args.batch_id or ch.get("job_id")
    if not job_id:
        raise ValueError("job_id missing; run create first or set --batch-id")

    interval = int(args.poll_interval_seconds or metadata.get("poll_interval_seconds") or 30)
    job = poll_batch_job(make_ark_api(), str(job_id), ARK_PROJECT_NAME, interval)
    phase = job.status.phase if job.status else ""
    ch["job_id"] = str(job_id)
    ch["job_phase"] = str(phase)
    save_json(meta_path, metadata)
    print(f"wait done, chunk={chunk_i}, final phase: {phase}")
    return job


def stage_collect(args: argparse.Namespace, run_dir: Path, job_obj: Any | None = None) -> None:
    meta_path, metadata = load_meta_or_fail(run_dir)
    chunk_i = _resolve_chunk_index(metadata, args)
    ch = _mut_chunk(metadata, chunk_i)
    job_id = args.batch_id or ch.get("job_id")
    if not job_id:
        raise ValueError("job_id missing; use create or --batch-id")

    job = job_obj if job_obj is not None else get_batch_job(
        make_ark_api(), str(job_id), ARK_PROJECT_NAME
    )
    phase = str(job.status.phase) if job.status and job.status.phase else ""
    ch["job_id"] = str(job_id)
    ch["job_phase"] = phase

    if phase != "Finished":
        save_json(meta_path, metadata)
        raise RuntimeError(f"batch job not finished, phase={phase}")

    bucket = TOS_BUCKET
    output_jsonl = Path(ch["output_jsonl"])
    key_payloads_path = Path(ch["key_payloads_json"])
    key_payloads = load_json(key_payloads_path)
    for k in list(key_payloads.keys()):
        key_payloads[k]["key"] = int(key_payloads[k]["key"])

    download_results_jsonl(
        make_tos_client(), bucket, str(ch["tos_output_prefix"]), output_jsonl
    )
    output_text = output_jsonl.read_text(encoding="utf-8")
    save_text(output_jsonl, output_text)

    key_results, _ = parse_batch_output(output_text, key_payloads)
    ch_sub = ch.get("submitted_keys", [])
    submitted = {str(x) for x in ch_sub} or set(key_payloads.keys())
    written, not_returned, empty_skipped = apply_collect_results(
        metadata, chunk_i, key_payloads, key_results, submitted
    )

    results_path = str(metadata["save_to"])
    ntot = _chunk_count(metadata)
    save_json(meta_path, metadata)
    print(f"updated generation JSON: {results_path} (chunk {chunk_i} / {ntot})")
    print(f"evaluate next: python main.py --evaluate-file \"{results_path}\" --evaluator deepseek-v4-flash")
    print(
        f"this collect: {written} line(s); not returned: {not_returned}; empty skipped: {empty_skipped}"
    )
    print(f"run_dir={run_dir}")


def main() -> None:
    args = parse_args()
    validate_doubao_model(str(args.model_id))
    args.completion_window = validate_completion_window(
        str(args.completion_window), enforce_openai_range=False
    )
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
        stage_collect(args, _run_dir_from_arg(args.run_dir), job_obj=None)
        return

    run_path = stage_prepare(args)
    _, run_meta = load_meta_or_fail(run_path)
    done = set(run_meta.get("completed_chunk_indices", []))
    for i in range(_chunk_count(run_meta)):
        if i in done:
            print(f"skipping chunk {i} (already completed)")
            continue
        args.chunk_index = i
        print(f"--- batch chunk {i + 1}/{_chunk_count(run_meta)} ---")
        stage_upload(args, run_path)
        stage_create(args, run_path)
        job = stage_wait(args, run_path)
        stage_collect(args, run_path, job_obj=job)
    args.chunk_index = None


if __name__ == "__main__":
    main()
