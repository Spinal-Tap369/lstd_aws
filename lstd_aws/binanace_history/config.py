# binanace_history/config

from dataclasses import dataclass
from typing import Optional


@dataclass
class HistoricalDownloadConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1m"

    # Choose one style:
    start: Optional[str] = None
    end: Optional[str] = None
    lookback_days: Optional[int] = None

    base_url: str = "https://api.binance.com"
    request_limit: int = 1000
    sleep_seconds: float = 0.15
    timeout: int = 20
    max_retries: int = 5
    output_dir: str = "data"

    validate_contiguity: bool = True
    allow_missing_candles: bool = False