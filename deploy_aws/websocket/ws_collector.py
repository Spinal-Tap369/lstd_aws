# /opt/pipeline/ws_collector.py

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import Any, Optional, cast

import boto3
import requests
import websockets
from botocore.exceptions import ClientError


"""
Standalone Binance 1m websocket collector for the collector EC2 instance.

What it does:
- subscribes to one Binance 1m kline websocket stream
- accepts only CLOSED 1m candles
- stores each raw candle in S3
- pushes each candle to a raw SQS FIFO queue
- tracks collector state / heartbeat in DynamoDB
- can catch up short collector-side outages via Binance REST using the collector's
  own last_raw_open_time in DynamoDB

What it does NOT do:
- no startup bootstrap of historical candles
- no training-to-inference state transfer work
- no warmup candle backfill for the inference instance

That startup catch-up belongs in the inference worker now.

Expected env vars:
- DATA_BUCKET
- RAW_QUEUE_URL
- STATE_TABLE
- AWS_REGION

Optional env vars:
- LIVE_SYMBOL=BTCUSDT
- STREAMS_JSON=[{"symbol":"BTCUSDT","interval":"1m"}]
- BINANCE_WS_BASE=wss://stream.binance.com:9443
- BINANCE_REST_BASE=https://api.binance.com
- LOG_LEVEL=INFO
- RECONNECT_SECONDS=82800
- IDLE_TIMEOUT_SECONDS=90
- HEARTBEAT_FLUSH_SECONDS=30
- SAFETY_LAG_MS=3000
- ENABLE_REST_CATCHUP=true
- S3_PREFIX=raw
"""


ALLOWED_INTERVAL = "1m"
STEP_MS = 60_000


def _env_required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


AWS_REGION = os.environ.get("AWS_REGION", "eu-central-1").strip()
DATA_BUCKET = _env_required("DATA_BUCKET")
RAW_QUEUE_URL = _env_required("RAW_QUEUE_URL")
STATE_TABLE_NAME = _env_required("STATE_TABLE")
S3_PREFIX = os.environ.get("S3_PREFIX", "raw").strip().strip("/") or "raw"

BINANCE_WS_BASE = os.environ.get("BINANCE_WS_BASE", "wss://stream.binance.com:9443").strip()
BINANCE_REST_BASE = os.environ.get("BINANCE_REST_BASE", "https://api.binance.com").strip()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper().strip()
RECONNECT_SECONDS = int(os.environ.get("RECONNECT_SECONDS", "82800"))
IDLE_TIMEOUT_SECONDS = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "90"))
HEARTBEAT_FLUSH_SECONDS = int(os.environ.get("HEARTBEAT_FLUSH_SECONDS", "30"))
SAFETY_LAG_MS = int(os.environ.get("SAFETY_LAG_MS", "3000"))
ENABLE_REST_CATCHUP = os.environ.get("ENABLE_REST_CATCHUP", "true").lower() in {"1", "true", "yes", "on"}


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ws_collector")


s3 = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)
dynamodb = cast(Any, boto3.resource("dynamodb", region_name=AWS_REGION))
state_table = dynamodb.Table(STATE_TABLE_NAME)

http = requests.Session()
http.headers.update({"User-Agent": "lstd-ws-collector/1.0"})

stop_requested = False
_last_heartbeat_flush: dict[tuple[str, str], int] = {}


def utc_iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_live_symbol() -> str:
    live_symbol = os.environ.get("LIVE_SYMBOL")
    if live_symbol:
        return live_symbol.upper().strip()

    streams_json = os.environ.get("STREAMS_JSON")
    if not streams_json:
        raise ValueError("Set LIVE_SYMBOL or STREAMS_JSON")

    parsed = json.loads(streams_json)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("STREAMS_JSON must be a non-empty JSON list")

    if len(parsed) != 1:
        raise ValueError(
            "This collector supports exactly one live stream. "
            "Set STREAMS_JSON to a single {symbol, interval} entry."
        )

    entry = parsed[0]
    if not isinstance(entry, dict):
        raise ValueError("STREAMS_JSON entries must be objects")

    symbol = str(entry.get("symbol", "")).upper().strip()
    interval = str(entry.get("interval", "")).strip()

    if not symbol:
        raise ValueError("STREAMS_JSON entry must include symbol")
    if interval != ALLOWED_INTERVAL:
        raise ValueError(
            f"This collector only supports interval={ALLOWED_INTERVAL}, got interval={interval}"
        )

    return symbol


LIVE_SYMBOL = _resolve_live_symbol()


def raw_key(symbol: str, interval: str, open_time: int) -> str:
    return f"{S3_PREFIX}/symbol={symbol}/interval={interval}/open_time={open_time}.json"


def _rest_get(path: str, params: Optional[dict[str, Any]] = None) -> Any:
    url = f"{BINANCE_REST_BASE.rstrip('/')}{path}"
    last_err: Optional[Exception] = None

    for attempt in range(5):
        try:
            resp = http.get(url, params=params, timeout=20)
            if resp.status_code in (418, 429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else (1.0 + attempt)
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.0 + attempt)

    raise RuntimeError(f"REST request failed for {path}: {last_err}")


def fetch_server_time_ms() -> int:
    payload = _rest_get("/api/v3/time")
    if not isinstance(payload, dict) or "serverTime" not in payload:
        raise RuntimeError(f"Unexpected /time response: {payload}")
    return int(payload["serverTime"])


def latest_safe_closed_open_time_ms() -> int:
    server_time_ms = fetch_server_time_ms()
    safe_ms = max(0, server_time_ms - SAFETY_LAG_MS)
    current_minute_open = (safe_ms // STEP_MS) * STEP_MS
    latest_closed_open = current_minute_open - STEP_MS
    return max(0, latest_closed_open)


def parse_rest_kline_row(symbol: str, row: list[Any]) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "interval": ALLOWED_INTERVAL,
        "open_time": int(row[0]),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "close_time": int(row[6]),
        "quote_asset_volume": float(row[7]),
        "number_of_trades": int(row[8]),
        "taker_buy_base_asset_volume": float(row[9]),
        "taker_buy_quote_asset_volume": float(row[10]),
        "date": utc_iso_from_ms(int(row[0])),
    }


def fetch_closed_klines_range(
    symbol: str,
    start_open_time_ms: int,
    end_open_time_ms: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    current_start = int(start_open_time_ms)

    while current_start <= int(end_open_time_ms):
        payload = _rest_get(
            "/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": ALLOWED_INTERVAL,
                "limit": 1000,
                "startTime": current_start,
                "endTime": int(end_open_time_ms + STEP_MS),
            },
        )

        if not isinstance(payload, list) or not payload:
            break

        chunk = [parse_rest_kline_row(symbol, row) for row in payload]
        chunk = [
            candle
            for candle in chunk
            if int(candle["open_time"]) >= int(start_open_time_ms)
            and int(candle["open_time"]) <= int(end_open_time_ms)
        ]
        if not chunk:
            break

        out.extend(chunk)

        last_open_time = int(chunk[-1]["open_time"])
        next_start = last_open_time + STEP_MS
        if next_start <= current_start:
            break
        current_start = next_start

    dedup: dict[int, dict[str, Any]] = {}
    for candle in out:
        dedup[int(candle["open_time"])] = candle

    return [dedup[k] for k in sorted(dedup.keys())]


def load_last_raw_open_time(symbol: str, interval: str) -> Optional[int]:
    item = state_table.get_item(
        Key={"symbol": symbol, "interval": interval},
        ConsistentRead=True,
    ).get("Item")

    if not item or "last_raw_open_time" not in item:
        return None

    return int(item["last_raw_open_time"])


def touch_ws_heartbeat(symbol: str, interval: str, event_time_ms: int) -> None:
    now_epoch = int(time.time())
    key = (symbol, interval)
    last_flush = _last_heartbeat_flush.get(key, 0)

    if now_epoch - last_flush < HEARTBEAT_FLUSH_SECONDS:
        return

    state_table.update_item(
        Key={"symbol": symbol, "interval": interval},
        UpdateExpression=(
            "SET ws_last_event_time_ms = :event_ms, "
            "ws_last_heartbeat_epoch = :heartbeat_epoch, "
            "updated_at = :updated_at"
        ),
        ExpressionAttributeValues={
            ":event_ms": int(event_time_ms),
            ":heartbeat_epoch": now_epoch,
            ":updated_at": now_epoch,
        },
    )
    _last_heartbeat_flush[key] = now_epoch


def advance_last_raw_open_time(symbol: str, interval: str, open_time: int, event_time_ms: int) -> None:
    now_epoch = int(time.time())
    state_table.update_item(
        Key={"symbol": symbol, "interval": interval},
        UpdateExpression=(
            "SET last_raw_open_time = :open_time, "
            "ws_last_event_time_ms = :event_ms, "
            "ws_last_heartbeat_epoch = :heartbeat_epoch, "
            "updated_at = :updated_at"
        ),
        ConditionExpression="attribute_not_exists(last_raw_open_time) OR last_raw_open_time < :open_time",
        ExpressionAttributeValues={
            ":open_time": int(open_time),
            ":event_ms": int(event_time_ms),
            ":heartbeat_epoch": now_epoch,
            ":updated_at": now_epoch,
        },
    )


def save_json_s3(bucket: str, key: str, payload: dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        ContentType="application/json",
    )


def send_raw_message(bucket: str, candle: dict[str, Any], source: str) -> None:
    symbol = str(candle["symbol"])
    interval = str(candle["interval"])
    open_time = int(candle["open_time"])
    key = raw_key(symbol, interval, open_time)

    body = {
        "source": source,
        "symbol": symbol,
        "interval": interval,
        "open_time": open_time,
        "bucket": bucket,
        "raw_key": key,
        "candle": candle,
    }

    sqs.send_message(
        QueueUrl=RAW_QUEUE_URL,
        MessageBody=json.dumps(body, separators=(",", ":")),
        MessageGroupId=f"{symbol}-{interval}",
        MessageDeduplicationId=f"{symbol}-{interval}-{open_time}",
    )


def publish_closed_candle(candle: dict[str, Any], event_time_ms: int, source: str) -> bool:
    symbol = str(candle["symbol"])
    interval = str(candle["interval"])
    open_time = int(candle["open_time"])

    if symbol != LIVE_SYMBOL:
        logger.info("skip foreign symbol %s", symbol)
        return False

    if interval != ALLOWED_INTERVAL:
        logger.info("skip foreign interval %s", interval)
        return False

    last_raw = load_last_raw_open_time(symbol, interval)
    if last_raw is not None and open_time <= last_raw:
        logger.info(
            "skip stale/duplicate candle symbol=%s interval=%s open_time=%s last_raw=%s",
            symbol,
            interval,
            open_time,
            last_raw,
        )
        return False

    key = raw_key(symbol, interval, open_time)
    save_json_s3(DATA_BUCKET, key, candle)
    send_raw_message(DATA_BUCKET, candle, source=source)

    try:
        advance_last_raw_open_time(symbol, interval, open_time, event_time_ms)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code != "ConditionalCheckFailedException":
            raise
        return False

    logger.info(
        "published closed candle symbol=%s interval=%s open_time=%s source=%s",
        symbol,
        interval,
        open_time,
        source,
    )
    return True


def catch_up_gap_from_state(symbol: str) -> int:
    if not ENABLE_REST_CATCHUP:
        return 0

    last_raw = load_last_raw_open_time(symbol, ALLOWED_INTERVAL)
    if last_raw is None:
        logger.info("no collector-side catchup: no last_raw_open_time in DynamoDB yet")
        return 0

    latest_safe_open_time = latest_safe_closed_open_time_ms()
    start_open_time = int(last_raw) + STEP_MS

    if start_open_time > latest_safe_open_time:
        logger.info("no catchup needed start=%s latest_safe=%s", start_open_time, latest_safe_open_time)
        return 0

    candles = fetch_closed_klines_range(
        symbol=symbol,
        start_open_time_ms=start_open_time,
        end_open_time_ms=latest_safe_open_time,
    )

    logger.info(
        "catchup start symbol=%s interval=%s from=%s to=%s candles=%d",
        symbol,
        ALLOWED_INTERVAL,
        start_open_time,
        latest_safe_open_time,
        len(candles),
    )

    published = 0
    for candle in candles:
        if publish_closed_candle(
            candle=candle,
            event_time_ms=int(time.time() * 1000),
            source="collector_rest_catchup",
        ):
            published += 1

    logger.info(
        "catchup done symbol=%s interval=%s published=%d",
        symbol,
        ALLOWED_INTERVAL,
        published,
    )
    return published


def parse_kline_event(message: dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[int], Optional[dict[str, Any]]]:
    payload = message.get("data", message)
    if payload.get("e") != "kline":
        return None, None, None, None

    kline = payload.get("k", {})
    symbol = str(kline.get("s", "")).upper().strip()
    interval = str(kline.get("i", "")).strip()
    event_time_raw = payload.get("E")
    if event_time_raw is None:
        return None, None, None, None
    event_time_ms = int(event_time_raw)

    if symbol != LIVE_SYMBOL:
        return None, None, None, None
    if interval != ALLOWED_INTERVAL:
        return None, None, None, None

    if not bool(kline.get("x", False)):
        return symbol, interval, event_time_ms, None

    candle = {
        "symbol": symbol,
        "interval": interval,
        "open_time": int(kline["t"]),
        "open": float(kline["o"]),
        "high": float(kline["h"]),
        "low": float(kline["l"]),
        "close": float(kline["c"]),
        "volume": float(kline["v"]),
        "close_time": int(kline["T"]),
        "quote_asset_volume": float(kline["q"]),
        "number_of_trades": int(kline["n"]),
        "taker_buy_base_asset_volume": float(kline["V"]),
        "taker_buy_quote_asset_volume": float(kline["Q"]),
        "date": utc_iso_from_ms(int(kline["t"])),
    }
    return symbol, interval, event_time_ms, candle


def build_stream_url(symbol: str) -> str:
    stream_name = f"{symbol.lower()}@kline_{ALLOWED_INTERVAL}"
    return f"{BINANCE_WS_BASE}/stream?streams={stream_name}"


def handle_closed_candle(candle: dict[str, Any], event_time_ms: int) -> None:
    publish_closed_candle(
        candle=candle,
        event_time_ms=event_time_ms,
        source="ws_collector",
    )


async def run_once(symbol: str) -> None:
    url = build_stream_url(symbol)
    logger.info("connecting to %s", url)

    async with websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=10,
        max_queue=1000,
    ) as ws:
        try:
            catch_up_gap_from_state(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.exception("catchup error after websocket connect: %s", exc)

        started = asyncio.get_running_loop().time()

        while not stop_requested:
            if asyncio.get_running_loop().time() - started > RECONNECT_SECONDS:
                logger.info("planned reconnect")
                return

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=IDLE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("idle timeout; reconnecting")
                return

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            message = json.loads(raw)
            symbol_out, interval, event_time_ms, candle = parse_kline_event(message)
            if symbol_out is None or interval is None or event_time_ms is None:
                continue

            touch_ws_heartbeat(symbol_out, interval, event_time_ms)

            if candle is not None:
                handle_closed_candle(candle, event_time_ms)


async def main() -> None:
    logger.info(
        "collector start symbol=%s interval=%s rest_catchup=%s safety_lag_ms=%d",
        LIVE_SYMBOL,
        ALLOWED_INTERVAL,
        ENABLE_REST_CATCHUP,
        SAFETY_LAG_MS,
    )

    while not stop_requested:
        try:
            await run_once(LIVE_SYMBOL)
        except Exception as exc:  # noqa: BLE001
            logger.exception("collector loop error: %s", exc)

        if not stop_requested:
            try:
                catch_up_gap_from_state(LIVE_SYMBOL)
            except Exception as exc:  # noqa: BLE001
                logger.exception("catchup error before reconnect: %s", exc)

            await asyncio.sleep(5)


def _request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global stop_requested
    stop_requested = True
    logger.info("stop requested")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    asyncio.run(main())

