"""Volcengine Ark batch (Job) + TOS helpers for doubao_batch.py."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tos
import volcenginesdkark.models as ark_models
import volcenginesdkcore
from volcenginesdkark.api.ark_api import ARKApi

ARK_REGION = "cn-beijing"
DEFAULT_TOS_ENDPOINT = "tos-cn-beijing.volces.com"
DEFAULT_TOS_REGION = "cn-beijing"
DEFAULT_TOS_BUCKET = "batch-doubao"
DEFAULT_PROJECT_NAME = "default"

TERMINAL_JOB_PHASES = {
    "Finished",
    "Failed",
    "Cancelled",
    "Expired",
    "Terminated",
    "Completed",
    "Success",
}


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


def get_batch_job(ark: ARKApi, job_id: str, project_name: str) -> ark_models.ItemForListBatchInferenceJobsOutput:
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
    import time

    while True:
        job = get_batch_job(ark, job_id, project_name)
        phase = ""
        if job.status and getattr(job.status, "phase", None):
            phase = str(job.status.phase)
        counts = job.request_counts
        completed = getattr(counts, "completed", None) if counts else None
        total = getattr(counts, "total", None) if counts else None
        if completed is not None and total is not None:
            print(f"job status: {phase} ({completed}/{total})")
        else:
            print(f"job status: {phase}")

        if phase in TERMINAL_JOB_PHASES:
            return job
        time.sleep(max(1, int(poll_interval_seconds)))
