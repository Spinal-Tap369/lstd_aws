from __future__ import annotations

import json
import os
import time
import urllib.parse
from typing import Any

import boto3
from botocore.exceptions import ClientError


"""
Lambda: artifact promoter

Recommended trigger:
- S3 ObjectCreated on bucket my-lstd-data
- prefix: training-artifacts/
- suffix: artifact_bundle.pt

What it does:
- copies the newly uploaded artifact bundle into a stable promoted location
- tries to copy fit_summary.json from the same training run directory
- writes promoted-artifacts/latest/latest.json manifest

This keeps promotion simple and deterministic.
The inference EC2 can pull from:
  s3://<bucket>/promoted-artifacts/latest/artifact_bundle.pt
"""


s3 = boto3.client("s3")

SOURCE_PREFIX = os.environ.get("SOURCE_PREFIX", "training-artifacts").strip("/")
DEST_PREFIX = os.environ.get("DEST_PREFIX", "promoted-artifacts").strip("/")
ARTIFACT_FILENAME = os.environ.get("ARTIFACT_FILENAME", "artifact_bundle.pt").strip()
FIT_SUMMARY_FILENAME = os.environ.get("FIT_SUMMARY_FILENAME", "fit_summary.json").strip()
COPY_FIT_SUMMARY = os.environ.get("COPY_FIT_SUMMARY", "true").lower() in {"1", "true", "yes", "on"}
DEST_BUCKET_OVERRIDE = os.environ.get("DEST_BUCKET", "").strip()


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    del context

    promoted: list[dict[str, Any]] = []
    skipped: list[str] = []

    records = event.get("Records", [])
    for record in records:
        event_name = str(record.get("eventName", ""))
        if not event_name.startswith("ObjectCreated"):
            skipped.append(f"skip eventName={event_name}")
            continue

        src_bucket = str(record["s3"]["bucket"]["name"])
        src_key = urllib.parse.unquote_plus(str(record["s3"]["object"]["key"]))

        if not src_key.startswith(f"{SOURCE_PREFIX}/"):
            skipped.append(f"skip prefix key={src_key}")
            continue

        if not src_key.endswith(f"/{ARTIFACT_FILENAME}"):
            skipped.append(f"skip filename key={src_key}")
            continue

        dest_bucket = DEST_BUCKET_OVERRIDE or src_bucket
        run_dir = src_key.rsplit("/", 1)[0]
        run_id = run_dir.split("/")[-1]

        promoted_artifact_run_key = f"{DEST_PREFIX}/runs/{run_id}/{ARTIFACT_FILENAME}"
        promoted_artifact_latest_key = f"{DEST_PREFIX}/latest/{ARTIFACT_FILENAME}"

        _copy_object(
            src_bucket=src_bucket,
            src_key=src_key,
            dest_bucket=dest_bucket,
            dest_key=promoted_artifact_run_key,
        )
        _copy_object(
            src_bucket=src_bucket,
            src_key=src_key,
            dest_bucket=dest_bucket,
            dest_key=promoted_artifact_latest_key,
        )

        fit_summary_source_key = f"{run_dir}/{FIT_SUMMARY_FILENAME}"
        fit_summary_promoted_run_key = f"{DEST_PREFIX}/runs/{run_id}/{FIT_SUMMARY_FILENAME}"
        fit_summary_promoted_latest_key = f"{DEST_PREFIX}/latest/{FIT_SUMMARY_FILENAME}"
        fit_summary_copied = False

        if COPY_FIT_SUMMARY and _object_exists(src_bucket, fit_summary_source_key):
            _copy_object(
                src_bucket=src_bucket,
                src_key=fit_summary_source_key,
                dest_bucket=dest_bucket,
                dest_key=fit_summary_promoted_run_key,
            )
            _copy_object(
                src_bucket=src_bucket,
                src_key=fit_summary_source_key,
                dest_bucket=dest_bucket,
                dest_key=fit_summary_promoted_latest_key,
            )
            fit_summary_copied = True

        manifest = {
            "promoted_at_epoch": int(time.time()),
            "source_bucket": src_bucket,
            "source_artifact_key": src_key,
            "source_fit_summary_key": fit_summary_source_key if fit_summary_copied else None,
            "destination_bucket": dest_bucket,
            "destination_artifact_key": promoted_artifact_latest_key,
            "destination_fit_summary_key": fit_summary_promoted_latest_key if fit_summary_copied else None,
            "run_id": run_id,
        }

        manifest_key = f"{DEST_PREFIX}/latest/latest.json"
        s3.put_object(
            Bucket=dest_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        promoted.append(
            {
                "run_id": run_id,
                "src_bucket": src_bucket,
                "src_key": src_key,
                "dest_bucket": dest_bucket,
                "dest_key": promoted_artifact_latest_key,
                "manifest_key": manifest_key,
                "fit_summary_copied": fit_summary_copied,
            }
        )

    return {
        "promoted": promoted,
        "skipped": skipped,
    }


def _copy_object(*, src_bucket: str, src_key: str, dest_bucket: str, dest_key: str) -> None:
    s3.copy_object(
        Bucket=dest_bucket,
        Key=dest_key,
        CopySource={"Bucket": src_bucket, "Key": src_key},
    )


def _object_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
