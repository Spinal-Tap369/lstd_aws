# binanace_history/pipeline

import os
from typing import Dict, Any

from .client import BinanceHistoricalKlinesClient
from .config import HistoricalDownloadConfig
from datasets.utils import (
    ensure_dir,
    resolve_time_range_ms,
    compact_time_str,
    validate_kline_contiguity,
)


def download_historical_klines(cfg: HistoricalDownloadConfig) -> Dict[str, Any]:
    ensure_dir(cfg.output_dir)

    start_ms, end_ms = resolve_time_range_ms(
        start=cfg.start,
        end=cfg.end,
        lookback_days=cfg.lookback_days,
        interval=cfg.interval,
    )

    client = BinanceHistoricalKlinesClient(
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )

    raw_df = client.fetch_historical_klines(
        symbol=cfg.symbol,
        interval=cfg.interval,
        start_ms=start_ms,
        end_ms=end_ms,
        request_limit=cfg.request_limit,
        sleep_seconds=cfg.sleep_seconds,
    )

    if raw_df.empty:
        raise RuntimeError("No kline data returned. Check symbol/interval/date range.")

    quality = validate_kline_contiguity(raw_df, cfg.interval)
    if cfg.validate_contiguity and (not quality["ok"]) and (not cfg.allow_missing_candles):
        raise RuntimeError(f"Kline continuity check failed: {quality}")

    stem = f"{cfg.symbol}_{cfg.interval}_{compact_time_str(start_ms)}_{compact_time_str(end_ms)}"
    raw_path = os.path.join(cfg.output_dir, f"{stem}_raw.csv")
    raw_df.to_csv(raw_path, index=False)

    return {
        "raw_path": raw_path,
        "raw_rows": len(raw_df),
        "quality": quality,
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }