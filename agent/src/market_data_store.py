"""Persistent SQLite cache for normalized market-data frames.

Frames are keyed by source, symbol, interval, exact request window, fields and
loader revision. The cache is a performance aid: corrupt or unsupported entries
are treated as misses and the authoritative loader is called again.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.paths import get_runtime_root

logger = logging.getLogger(__name__)
_SCHEMA_VERSION = 1


class MarketDataStore:
    """SQLite-backed cache and inventory for normalized market frames."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (get_runtime_root() / "market-data" / "market-data.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _init_schema(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS market_data_frames (
                    cache_key TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    source_revision TEXT,
                    dataset_type TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    first_timestamp TEXT,
                    last_timestamp TEXT,
                    updated_at TEXT NOT NULL,
                    payload BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_market_data_inventory
                    ON market_data_frames(source, symbol, timeframe, updated_at DESC);
            """)

    @staticmethod
    def _key(source, symbol, timeframe, start_date, end_date, fields, revision) -> tuple[str, str]:
        fields_json = json.dumps(sorted(fields or []), separators=(",", ":"))
        data = json.dumps(
            [source, symbol, timeframe, start_date, end_date, fields_json, revision],
            ensure_ascii=False, separators=(",", ":"),
        ).encode()
        return hashlib.sha256(data).hexdigest(), fields_json

    def get(self, *, source, symbol, timeframe, start_date, end_date, fields=None,
            source_revision=None) -> pd.DataFrame | None:
        # Do not pin today's still-forming candle or future request windows.
        if end_date >= date.today().isoformat():
            return None
        key, _ = self._key(source, symbol, timeframe, start_date, end_date, fields, source_revision)
        try:
            with self._connect() as db:
                row = db.execute(
                    "SELECT schema_version, payload FROM market_data_frames WHERE cache_key=?",
                    (key,),
                ).fetchone()
            if row is None or row["schema_version"] != _SCHEMA_VERSION:
                return None
            doc = json.loads(gzip.decompress(row["payload"]))
            # Construct from the split payload directly. ``read_json`` treats
            # a JSON string as a file path in newer pandas releases.
            split = json.loads(doc["data"])
            frame = pd.DataFrame(
                data=split["data"], index=split["index"], columns=split["columns"],
            )
            if doc.get("index_type") == "datetime":
                frame.index = pd.to_datetime(frame.index, utc=bool(doc.get("index_utc")))
                if not doc.get("index_utc"):
                    frame.index = frame.index.tz_localize(None)
                if doc.get("index_dtype"):
                    frame.index = frame.index.astype(doc["index_dtype"])
            frame.index.name = doc.get("index_name")
            frame.columns.name = doc.get("columns_name")
            for column, dtype in doc.get("datetime_columns", {}).items():
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column], errors="coerce", utc="UTC" in dtype)
                    if "UTC" not in dtype and getattr(frame[column].dt, "tz", None) is not None:
                        frame[column] = frame[column].dt.tz_localize(None)
                    frame[column] = frame[column].astype(dtype)
            frame.attrs.update(doc.get("attrs", {}))
            return frame
        except Exception as exc:  # cache data is disposable; loader remains authoritative
            logger.warning("market data SQLite cache read failed for %s %s: %s", source, symbol, exc)
            return None

    def put(self, *, source, symbol, timeframe, start_date, end_date, fields=None,
            source_revision=None, frame: pd.DataFrame | None) -> None:
        if (
            frame is None or frame.empty or not isinstance(frame, pd.DataFrame)
            or frame.index.nlevels != 1
        ):
            return
        # A request that ends today often includes a still-forming candle. Keep
        # the settled history it returned, but key that payload only through
        # yesterday so it can never masquerade as a complete current-day range.
        today = date.today()
        requested_end = date.fromisoformat(end_date)
        stored_end = min(requested_end, today - timedelta(days=1))
        if requested_end >= today:
            if not isinstance(frame.index, pd.DatetimeIndex):
                return
            cutoff = pd.Timestamp(today.isoformat())
            if frame.index.tz is not None:
                cutoff = cutoff.tz_localize(frame.index.tz)
            frame = frame.loc[frame.index < cutoff].copy()
            if frame.empty:
                return
        if date.fromisoformat(start_date) > stored_end:
            return
        end_date = stored_end.isoformat()
        key, fields_json = self._key(source, symbol, timeframe, start_date, end_date, fields, source_revision)
        try:
            index_type = "datetime" if isinstance(frame.index, pd.DatetimeIndex) else "other"
            index_utc = bool(isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None)
            datetime_columns: dict[str, str] = {}
            for column in frame.columns:
                if pd.api.types.is_datetime64_any_dtype(frame[column].dtype):
                    datetime_columns[str(column)] = str(frame[column].dtype)
            doc = {
                "data": frame.to_json(orient="split", date_format="iso", date_unit="ns"),
                "index_type": index_type,
                "index_utc": index_utc,
                "index_dtype": str(frame.index.dtype) if isinstance(frame.index, pd.DatetimeIndex) else None,
                "index_name": frame.index.name,
                "columns_name": frame.columns.name,
                "datetime_columns": datetime_columns,
                "attrs": {k: v for k, v in frame.attrs.items() if isinstance(v, (str, int, float, bool))},
            }
            payload = gzip.compress(json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode(), compresslevel=6)
            index = frame.index
            first = str(index.min()) if len(index) else None
            last = str(index.max()) if len(index) else None
            kind = "perpetual_bundle" if any(str(c).startswith("mark_") for c in frame.columns) else "ohlcv"
            with self._connect() as db:
                db.execute("""
                    INSERT INTO market_data_frames (
                        cache_key, schema_version, source, symbol, timeframe, start_date,
                        end_date, fields_json, source_revision, dataset_type, row_count,
                        first_timestamp, last_timestamp, updated_at, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        schema_version=excluded.schema_version,
                        dataset_type=excluded.dataset_type,
                        row_count=excluded.row_count,
                        first_timestamp=excluded.first_timestamp,
                        last_timestamp=excluded.last_timestamp,
                        updated_at=excluded.updated_at,
                        payload=excluded.payload
                """, (
                    key, _SCHEMA_VERSION, source, symbol, timeframe, start_date, end_date,
                    fields_json, source_revision, kind, len(frame), first, last,
                    pd.Timestamp.now(tz="UTC").isoformat(), payload,
                ))
        except Exception as exc:
            logger.warning("market data SQLite cache write failed for %s %s: %s", source, symbol, exc)

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("""
                SELECT source, symbol, timeframe, dataset_type, start_date, end_date,
                       row_count, first_timestamp, last_timestamp, updated_at,
                       length(payload) AS stored_bytes
                FROM market_data_frames ORDER BY updated_at DESC, symbol, timeframe
            """).fetchall()
        return [dict(row) for row in rows]
