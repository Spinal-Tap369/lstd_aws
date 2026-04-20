# orchestration/cli.py

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import PipelineConfig
from .run import run_pipeline, write_default_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lstd-aws",
        description="Run the end-to-end LSTD AWS orchestration pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init-config",
        help="Write a default real-run BTCUSDT 1m pipeline config JSON.",
    )
    init_parser.add_argument(
        "--output",
        default="configs/btc_1m_real.json",
        help="Path to the JSON config file to write.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run download + training + live inference from a config JSON.",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to a pipeline config JSON.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "init-config":
        path = write_default_config(args.output)
        print(f"Wrote default config -> {path}")
        return 0

    if args.command == "run":
        cfg = PipelineConfig.load_json(args.config)
        summary = run_pipeline(cfg)
        print(json.dumps(summary, indent=2))
        return 0

    parser.error("Unknown command")
    return 2