from __future__ import annotations

import os
from typing import Dict

from .config import S3ArtifactConfig


def build_training_s3_key(
    *,
    prefix: str,
    experiment_name: str,
    run_id: str,
    filename: str,
) -> str:
    parts = [
        prefix.strip("/"),
        experiment_name.strip("/"),
        run_id.strip("/"),
        os.path.basename(filename),
    ]
    return "/".join(part for part in parts if part)


def upload_file_to_s3(
    *,
    local_path: str,
    bucket: str,
    key: str,
    region: str,
) -> Dict[str, str]:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for S3 uploads. Install it in the training environment first."
        ) from exc
    
    from boto3.session import Session
    session = Session(region_name=region)
    s3_client = session.client("s3")
    s3_client.upload_file(local_path, bucket, key)

    return {
        "local_path": local_path,
        "bucket": bucket,
        "key": key,
        "s3_uri": f"s3://{bucket}/{key}",
        "region": region,
    }


def upload_training_outputs(
    *,
    s3_cfg: S3ArtifactConfig,
    experiment_name: str,
    run_id: str,
    artifact_bundle_path: str,
    best_checkpoint_path: str,
    fit_summary_path: str,
) -> Dict[str, Dict[str, str]]:
    if not s3_cfg.enabled:
        return {}

    uploads: Dict[str, Dict[str, str]] = {}

    artifact_key = build_training_s3_key(
        prefix=s3_cfg.prefix,
        experiment_name=experiment_name,
        run_id=run_id,
        filename=artifact_bundle_path,
    )
    uploads["artifact_bundle"] = upload_file_to_s3(
        local_path=artifact_bundle_path,
        bucket=s3_cfg.bucket,
        key=artifact_key,
        region=s3_cfg.region,
    )

    if s3_cfg.upload_best_checkpoint:
        checkpoint_key = build_training_s3_key(
            prefix=s3_cfg.prefix,
            experiment_name=experiment_name,
            run_id=run_id,
            filename=best_checkpoint_path,
        )
        uploads["best_checkpoint"] = upload_file_to_s3(
            local_path=best_checkpoint_path,
            bucket=s3_cfg.bucket,
            key=checkpoint_key,
            region=s3_cfg.region,
        )

    if s3_cfg.upload_fit_summary:
        summary_key = build_training_s3_key(
            prefix=s3_cfg.prefix,
            experiment_name=experiment_name,
            run_id=run_id,
            filename=fit_summary_path,
        )
        uploads["fit_summary"] = upload_file_to_s3(
            local_path=fit_summary_path,
            bucket=s3_cfg.bucket,
            key=summary_key,
            region=s3_cfg.region,
        )

    return uploads
