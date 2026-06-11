"""SimpleQA batch generation for Volcengine Ark (Doubao) via Batch Job + TOS.

https://www.volcengine.com/docs/82379/1399517
https://www.volcengine.com/docs/82379/1339603
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tos
import volcenginesdkark.models as ark_models
import volcenginesdkcore
from volcenginesdkark.api.ark_api import ARKApi

from common_config import save_json
from batch_api.general_batch import (
    PROVIDER_DEFAULTS,
    PROVIDER_DOUBAO,
    _models_yaml_path,
    _mut_chunk,
    _resolve_chunk_index,
    add_simpleqa_batch_arguments,
    build_batch_input_file_simpleqa_job,
    detect_provider,
    finish_simpleqa_collect,
    load_meta_or_fail,
    load_yaml_config,
    poll_until_terminal,
    prepare_simpleqa_run,
    run_simpleqa_pipeline,
    save_chunk_meta,
    validate_completion_window,
)

ARK_REGION = "cn-beijing"
DEFAULT_TOS_ENDPOINT = "tos-cn-beijing.volces.com"
DEFAULT_TOS_REGION = "cn-beijing"
TOS_BUCKET = "batch-doubao"
ARK_PROJECT_NAME = "default"

TERMINAL_JOB_PHASES = {
    "Finished",
    "Failed",
    "Cancelled",
    "Expired",
    "Terminated",
    "Completed",
    "Success",
}

SUCCESS_JOB_PHASES = {"Finished", "Completed", "Success"}


def volc_credentials() -> tuple[str, str]:
    ak = os.environ.get("VOLC_ACCESSKEY", "").strip()
    sk = os.environ.get("VOLC_SECRETKEY", "").strip()
    if not ak or not sk:
        raise SystemExit(
            "VOLC_ACCESSKEY and VOLC_SECRETKEY must be set (see Volcengine IAM docs)."
        )
    return ak, sk


def make_tos_client(
    endpoint: str | None = None,
    region: str | None = None,
) -> tos.TosClientV2:
    ak, sk = volc_credentials()
    return tos.TosClientV2(
        ak,
        sk,
        endpoint or os.environ.get("TOS_ENDPOINT", DEFAULT_TOS_ENDPOINT),
        region or os.environ.get("TOS_REGION", DEFAULT_TOS_REGION),
    )


def make_ark_api() -> ARKApi:
    ak, sk = volc_credentials()
    configuration = volcenginesdkcore.Configuration()
    configuration.ak = ak
    configuration.sk = sk
    configuration.region = ARK_REGION
    return ARKApi(volcenginesdkcore.ApiClient(configuration))


def upload_file_to_tos(
    client: tos.TosClientV2,
    bucket: str,
    object_key: str,
    local_path: Path,
) -> str:
    client.put_object_from_file(bucket, object_key, str(local_path))
    print(f"TOS upload done: tos://{bucket}/{object_key}")
    return object_key


def ensure_tos_output_prefix(
    client: tos.TosClientV2,
    bucket: str,
    output_prefix: str,
) -> str:
    """Create a placeholder object so Ark can resolve OutputDirTosLocation."""
    prefix = str(output_prefix).lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    placeholder_key = f"{prefix}.keep"
    client.put_object(bucket, placeholder_key, content=b"")
    print(f"TOS output prefix ready: tos://{bucket}/{prefix}")
    return prefix


def download_results_jsonl(
    client: tos.TosClientV2,
    bucket: str,
    output_prefix: str,
    dest_path: Path,
) -> str:
    prefix = str(output_prefix).lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    chosen_key: str | None = None
    truncated = True
    continuation = ""
    while truncated:
        resp = client.list_objects_type2(
            bucket,
            prefix=prefix,
            continuation_token=continuation or None,
        )
        for obj in resp.contents or []:
            key = getattr(obj, "key", "") or ""
            if key.endswith("results.jsonl"):
                chosen_key = key
                break
        truncated = bool(getattr(resp, "is_truncated", False))
        continuation = getattr(resp, "next_continuation_token", "") or ""
        if chosen_key:
            break

    if not chosen_key:
        raise FileNotFoundError(
            f"results.jsonl not found under tos://{bucket}/{prefix} "
            "(job may still be running or output path differs)"
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    client.get_object_to_file(bucket, chosen_key, str(dest_path))
    print(f"TOS download done: tos://{bucket}/{chosen_key} -> {dest_path}")
    return chosen_key


def build_model_reference(
    model_id: str,
    model_cfg: dict[str, Any],
) -> ark_models.ModelReferenceForCreateBatchInferenceJobInput:
    """Map models.yaml entry to CreateBatchInferenceJob ModelReference."""
    batch_fm = model_cfg.get("batch_foundation_model")
    if isinstance(batch_fm, dict) and batch_fm.get("name"):
        fm = ark_models.FoundationModelForCreateBatchInferenceJobInput(
            name=str(batch_fm["name"]),
            model_version=str(
                batch_fm.get("model_version") or batch_fm.get("version") or ""
            ),
        )
        return ark_models.ModelReferenceForCreateBatchInferenceJobInput(
            foundation_model=fm
        )

    endpoint_id = str(model_cfg.get("batch_endpoint_id") or model_id)
    if endpoint_id.startswith("ep-"):
        return ark_models.ModelReferenceForCreateBatchInferenceJobInput(
            custom_model_id=endpoint_id
        )

    raise ValueError(
        f"batch job model reference: set batch_endpoint_id / use ep- id, or add "
        f"batch_foundation_model {{name, model_version}} in models.yaml for {model_id!r}"
    )


def create_batch_inference_job(
    ark: ARKApi,
    *,
    name: str,
    model_reference: ark_models.ModelReferenceForCreateBatchInferenceJobInput,
    input_bucket: str,
    input_object_key: str,
    output_bucket: str,
    output_prefix: str,
    completion_window: str,
    project_name: str,
    description: str = "",
) -> str:
    req = ark_models.CreateBatchInferenceJobRequest(
        name=name,
        description=description or name,
        model_reference=model_reference,
        input_file_tos_location=ark_models.InputFileTosLocationForCreateBatchInferenceJobInput(
            bucket_name=input_bucket,
            object_key=input_object_key,
        ),
        output_dir_tos_location=ark_models.OutputDirTosLocationForCreateBatchInferenceJobInput(
            bucket_name=output_bucket,
            object_key=output_prefix,
        ),
        project_name=project_name,
        completion_window=completion_window,
    )
    resp = ark.create_batch_inference_job(req)
    job_id = resp.id
    if not job_id:
        raise RuntimeError("CreateBatchInferenceJob returned empty id")
    print(f"batch inference job created: {job_id}")
    return str(job_id)


def get_batch_job(
    ark: ARKApi, job_id: str, project_name: str
) -> ark_models.ItemForListBatchInferenceJobsOutput:
    req = ark_models.ListBatchInferenceJobsRequest(
        project_name=project_name,
        page_number=1,
        page_size=10,
        filter=ark_models.FilterForListBatchInferenceJobsInput(ids=[job_id]),
    )
    resp = ark.list_batch_inference_jobs(req)
    items = resp.items or []
    for item in items:
        if getattr(item, "id", None) == job_id:
            return item
    if items:
        return items[0]
    raise RuntimeError(f"batch job not found: {job_id}")


def poll_batch_job(
    ark: ARKApi,
    job_id: str,
    project_name: str,
    poll_interval_seconds: int,
) -> ark_models.ItemForListBatchInferenceJobsOutput:
    def _counts(job: Any) -> tuple[int | None, int | None]:
        counts = job.request_counts
        if not counts:
            return None, None
        return getattr(counts, "completed", None), getattr(counts, "total", None)

    def _phase(job: Any) -> str:
        if job.status and getattr(job.status, "phase", None):
            return str(job.status.phase)
        return ""

    return poll_until_terminal(
        lambda: get_batch_job(ark, job_id, project_name),
        get_status=_phase,
        get_counts=_counts,
        terminal_states=TERMINAL_JOB_PHASES,
        poll_interval_seconds=poll_interval_seconds,
        status_label="job status",
        hide_zero_total=True,
    )


def _tos_dataset_prefix(run_dir: Path, safe_model: str) -> str:
    prefix = PROVIDER_DEFAULTS[PROVIDER_DOUBAO]["tos_input_prefix"]
    return f"{prefix}/{safe_model}/{run_dir.name}"


def _tos_output_prefix(run_dir: Path, safe_model: str, chunk_idx: int) -> str:
    prefix = PROVIDER_DEFAULTS[PROVIDER_DOUBAO]["tos_output_prefix"]
    return f"{prefix}/{safe_model}/{run_dir.name}/c{chunk_idx:03d}/"


def _print_doubao_prepare_done(
    run_dir: Path, metadata: dict[str, Any], n_ch: int, max_per: int
) -> None:
    all_submitted = metadata.get("submitted_keys", [])
    resume_skipped = metadata.get("resume_skipped", 0)
    print(
        f"prepare done, run_dir={run_dir} ({n_ch} file(s), {max_per} lines max, "
        f"{len(all_submitted)} key(s), resume skipped: {resume_skipped})"
    )
    print(f"TOS bucket: {metadata.get('tos_bucket', TOS_BUCKET)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SimpleQA: Doubao batch inference (Ark Job API + TOS)."
    )
    add_simpleqa_batch_arguments(
        parser,
        default_model_id="doubao-seed-2-0-pro-260215",
        default_completion_window="1d",
        default_poll_interval_seconds=30,
    )
    return parser.parse_args()


def stage_prepare(args: argparse.Namespace) -> Path:
    provider = detect_provider(str(args.model_id))
    defaults = PROVIDER_DEFAULTS[provider]

    def chunk_extra(chunk_idx: int, run_dir: Path, safe_model: str) -> dict[str, Any]:
        return {
            "tos_bucket": TOS_BUCKET,
            "tos_input_object_key": (
                f"{_tos_dataset_prefix(run_dir, safe_model)}/batch_input_c{chunk_idx:03d}.jsonl"
            ),
            "tos_output_prefix": _tos_output_prefix(run_dir, safe_model, chunk_idx),
            "job_id": None,
            "job_phase": None,
        }

    return prepare_simpleqa_run(
        args,
        provider=provider,
        defaults=defaults,
        build_chunk_input=build_batch_input_file_simpleqa_job,
        chunk_extra=chunk_extra,
        metadata_version=4,
        metadata_extra={
            "batch_mode": "job",
            "tos_bucket": TOS_BUCKET,
            "project_name": ARK_PROJECT_NAME,
        },
        after_prepare_print=_print_doubao_prepare_done,
    )


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
    save_chunk_meta(
        meta_path,
        metadata,
        chunk_i,
        {"tos_uploaded": True, "job_id": None, "job_phase": None},
    )
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
    tos_client = make_tos_client()
    output_prefix = str(ch["tos_output_prefix"])
    ensure_tos_output_prefix(tos_client, bucket, output_prefix)

    job_id = create_batch_inference_job(
        make_ark_api(),
        name=f"simpleqa-{model_id}-c{chunk_i:03d}-{run_dir.name}"[:128],
        model_reference=build_model_reference(model_id, model_cfg),
        input_bucket=bucket,
        input_object_key=str(ch["tos_input_object_key"]),
        output_bucket=bucket,
        output_prefix=output_prefix,
        completion_window=completion_window,
        project_name=ARK_PROJECT_NAME,
        description=f"simpleqa chunk {chunk_i}",
    )
    metadata["completion_window"] = completion_window
    save_chunk_meta(
        meta_path,
        metadata,
        chunk_i,
        {"job_id": job_id, "job_phase": "Queued"},
    )
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
    save_chunk_meta(
        meta_path,
        metadata,
        chunk_i,
        {"job_id": str(job_id), "job_phase": str(phase)},
    )
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

    if phase not in SUCCESS_JOB_PHASES:
        save_json(meta_path, metadata)
        if phase in TERMINAL_JOB_PHASES:
            raise RuntimeError(f"batch job ended without success, phase={phase}")
        raise RuntimeError(f"batch job not finished, phase={phase}")

    bucket = TOS_BUCKET
    output_jsonl = Path(ch["output_jsonl"])
    key_payloads_path = Path(ch["key_payloads_json"])

    download_results_jsonl(
        make_tos_client(), bucket, str(ch["tos_output_prefix"]), output_jsonl
    )
    output_text = output_jsonl.read_text(encoding="utf-8")

    finish_simpleqa_collect(
        meta_path, metadata, chunk_i, output_text, key_payloads_path, run_dir
    )


def main() -> None:
    args = parse_args()
    detect_provider(str(args.model_id))
    args.completion_window = validate_completion_window(
        str(args.completion_window), enforce_openai_range=False
    )
    run_simpleqa_pipeline(
        args,
        stage_prepare=stage_prepare,
        stage_upload=stage_upload,
        stage_create=stage_create,
        stage_wait=stage_wait,
        stage_collect=stage_collect,
    )


if __name__ == "__main__":
    main()
