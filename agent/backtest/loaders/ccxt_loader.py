"""CCXT loader: unified crypto exchange data (100+ exchanges).

Uses the CCXT library to fetch OHLCV candles from any supported exchange.
Defaults to Binance; configurable via CCXT_EXCHANGE env var.
No API key required for public market data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from backtest.loaders.base import (
    cached_loader_fetch,
    check_budget,
    positive_env_float,
    positive_env_int,
    retry_with_budget,
    validate_date_range,
)
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1H": "1h", "1h": "1h", "4H": "4h", "4h": "4h",
    "1D": "1d", "1d": "1d", "1W": "1w", "1w": "1w", "1M": "1M",
}

_TIMEFRAME_DELTA = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),
    "1M": pd.Timedelta(days=31),
}

# P12-b: ccxt had no request timeout and an unbounded paginated fetch with
# no retry budget, so a transient disconnect hung get_market_data for 10+
# minutes. Cap each HTTP call, bound transient retries, and enforce a hard
# wall-clock budget so the fetch fails fast instead of hanging. Retry
# scheduling is delegated to :mod:`backtest.loaders.base`.
_CCXT_TIMEOUT_MS = positive_env_int("CCXT_TIMEOUT_MS", 15_000)
_CCXT_FETCH_BUDGET_S = positive_env_float("CCXT_FETCH_BUDGET_S", 60.0)
_FUNDING_HOURS = {0, 8, 16}  # Binance legacy schedule; OKX uses recorded events.


def _parse_ccxt_symbol(code: str) -> tuple[str, str]:
    """Return the canonical CCXT symbol and instrument type for ``code``."""
    normalized = code.strip().upper()
    if normalized.endswith(("-PERP", "-SWAP")):
        suffix = "-SWAP" if normalized.endswith("-SWAP") else "-PERP"
        match = re.fullmatch(rf"([A-Z0-9]+)-USDT{suffix}", normalized)
        if match is None:
            raise ValueError(
                "USD-M perpetual symbol must use BASE-USDT-PERP or BASE-USDT-SWAP"
            )
        return f"{match.group(1)}/USDT:USDT", "okx_swap" if suffix == "-SWAP" else "swap"
    return normalized.replace("-", "/"), "spot"


def _first_proxy_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()  # noqa: env-gate — system proxy vars
        if value:
            return value
    return ""


def _ccxt_proxy_config() -> dict[str, str]:
    """Build CCXT proxy settings from conventional proxy environment variables."""
    all_proxy = _first_proxy_env("ALL_PROXY", "all_proxy")
    http_proxy = _first_proxy_env("HTTP_PROXY", "http_proxy") or all_proxy
    https_proxy = _first_proxy_env("HTTPS_PROXY", "https_proxy") or all_proxy or http_proxy

    proxies: dict[str, str] = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    return proxies


_BRACKET_SCHEMA_VERSION = 1


def _validate_bracket_artifact(artifact: dict, *, expected_symbol: str) -> tuple[list[dict], str]:
    """Validate a caller-supplied maintenance-bracket artifact.

    ``binanceusdm.fetch_leverage_tiers()`` routes to Binance's signed
    ``GET /fapi/v1/leverageBracket`` USER_DATA endpoint — it requires an API
    key even though the response carries no account-specific data. This
    loader no longer calls it: maintenance brackets are supplied out of band
    as a versioned artifact and validated here, never fetched live. Fails
    closed (raises ``ValueError``) on any schema, symbol, provenance,
    ordering, or content-hash problem.
    """
    if not isinstance(artifact, dict):
        raise ValueError("bracket artifact must be a dict")

    schema_version = artifact.get("schema_version")
    if schema_version != _BRACKET_SCHEMA_VERSION:
        raise ValueError(
            f"bracket artifact schema_version must be {_BRACKET_SCHEMA_VERSION}, "
            f"got {schema_version!r}"
        )

    symbol = artifact.get("symbol")
    if symbol != expected_symbol:
        raise ValueError(
            f"bracket artifact symbol mismatch: expected {expected_symbol!r}, got {symbol!r}"
        )

    provenance_timestamp = artifact.get("provenance_timestamp")
    if not provenance_timestamp:
        raise ValueError("bracket artifact is missing provenance_timestamp")
    try:
        pd.Timestamp(provenance_timestamp)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"bracket artifact provenance_timestamp is not a valid timestamp: "
            f"{provenance_timestamp!r}"
        ) from exc

    brackets = artifact.get("brackets")
    if not brackets:
        raise ValueError("bracket artifact has no brackets")

    normalized: list[dict] = []
    for tier in brackets:
        try:
            record = {
                "bracket_tier": int(tier["bracket_tier"]),
                "notional_cap": float(tier["notional_cap"]),
                "maintenance_rate": float(tier["maintenance_rate"]),
                "cumulative_maintenance_amount": float(tier["cumulative_maintenance_amount"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"bracket artifact tier for {expected_symbol} is missing a required field: {exc}"
            ) from exc
        # Optional: reserved for future risk-model calibration, not part of
        # Binance's own bracket schema. Validated only if present.
        coefficient = tier.get("notional_coefficient")
        if coefficient is not None:
            try:
                record["notional_coefficient"] = float(coefficient)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"bracket artifact notional_coefficient must be numeric: {coefficient!r}"
                ) from exc
        normalized.append(record)

    normalized.sort(key=lambda row: row["bracket_tier"])
    caps = [row["notional_cap"] for row in normalized]
    if caps != sorted(caps) or len(caps) != len(set(caps)):
        raise ValueError(
            f"bracket artifact notional caps for {expected_symbol} are not strictly increasing"
        )

    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    computed_hash = hashlib.sha256(blob).hexdigest()[:16]
    content_hash = artifact.get("content_hash")
    if content_hash != computed_hash:
        raise ValueError(
            f"bracket artifact content_hash mismatch for {expected_symbol}: "
            f"expected {computed_hash}, artifact declares {content_hash!r}"
        )

    return normalized, computed_hash


def _validate_perpetual_candles(
    frame: pd.DataFrame, *, symbol: str, timeframe: str, label: str,
    require_volume: bool = True,
) -> None:
    """Fail closed on malformed or discontinuous perpetual candle series.

    Trade-price candles must include a finite non-negative volume. OKX
    mark-price candles intentionally carry no volume field; for those frames
    only OHLC validity is checked.
    """
    if frame.empty:
        raise ValueError(f"{label} candles are empty for {symbol}")
    if frame.index.has_duplicates:
        raise ValueError(f"duplicate {label} candle timestamps for {symbol}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} candle timestamps are not increasing for {symbol}")
    delta = _TIMEFRAME_DELTA.get(timeframe)
    if delta is None:
        raise ValueError(f"unsupported CCXT timeframe: {timeframe}")
    gaps = frame.index.to_series().diff().dropna()
    if not gaps.eq(delta).all():
        first = gaps[gaps.ne(delta)].index[0]
        raise ValueError(
            f"{label} candles are not continuous for {symbol}; unexpected spacing at {first}"
        )
    columns = ["open", "high", "low", "close"]
    if require_volume:
        columns.append("volume")
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} candles contain missing or non-finite OHLCV values for {symbol}")
    invalid = (
        (values["high"] < values[["open", "low", "close"]].max(axis=1))
        | (values["low"] > values[["open", "high", "close"]].min(axis=1))
        | (values[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    if require_volume:
        invalid |= values["volume"].lt(0)
    if invalid.any():
        raise ValueError(f"{label} candles contain invalid OHLCV rows for {symbol}")


def _completed_candle_end(
    end_ms: int, timeframe: str, *, now: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Return the exclusive end of the completed portion of a request.

    OKX includes the currently forming candle in its public response. A
    request ending on today must therefore stop at the current UTC interval
    boundary; otherwise a full-day coverage check incorrectly demands bars
    from the future. Requests ending in the past retain their requested end.
    """
    delta = _TIMEFRAME_DELTA.get(timeframe)
    if delta is None:
        raise ValueError(f"unsupported CCXT timeframe: {timeframe}")
    requested_end = pd.Timestamp(end_ms, unit="ms")
    if now is None:
        now = pd.Timestamp.utcnow().tz_localize(None)
    elif now.tzinfo is not None:
        now = now.tz_localize(None)
    return min(requested_end, now.floor(delta))


@register
class DataLoader:
    """CCXT-backed crypto OHLCV loader (100+ exchanges)."""

    name = "ccxt"
    markets = {"crypto"}
    requires_auth = False

    def is_available(self) -> bool:
        """Available if ccxt is installed."""
        try:
            import ccxt  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self) -> None:
        pass

    def _get_exchange(self, instrument_type: str = "spot"):
        """Create an exchange instance for spot or configured USDT swaps."""
        import ccxt
        from src.config.accessor import get_env_config

        exchange_id = get_env_config().data.ccxt_exchange.lower()
        if instrument_type == "okx_swap":
            exchange_id = "okx"
        elif instrument_type == "swap" and exchange_id in {"binance", "binanceusdm"}:
            exchange_id = "binanceusdm"
        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            logger.warning("Unknown CCXT exchange %s, falling back to binance", exchange_id)
            exchange_cls = ccxt.binance

        config = {"enableRateLimit": True, "timeout": _CCXT_TIMEOUT_MS}
        proxies = _ccxt_proxy_config()
        if proxies:
            config["proxies"] = proxies
        return exchange_cls(config)

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
        bracket_artifacts: Optional[Dict[str, dict]] = None,
        require_brackets: bool = False,
        margin_mode: str = "cross",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch crypto OHLCV via CCXT.

        Args:
            codes: Symbols like ``["BTC-USDT", "ETH-USDT"]``.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            interval: Bar size.
            fields: Ignored.
            bracket_artifacts: Optional ``{code: artifact}`` map of caller-supplied,
                versioned maintenance-bracket artifacts (see
                ``_validate_bracket_artifact``) for ``-PERP`` codes. Normal
                execution/mark/funding data never needs this — it stays
                zero-credential regardless. OKX strict swap runs fetch the
                current public position-tier snapshot without credentials.
            require_brackets: When True, a ``-PERP`` code without a matching,
                valid artifact in ``bracket_artifacts`` fails closed instead of
                silently returning data without bracket columns. OKX strict
                swaps use the public position-tiers endpoint.
            margin_mode: OKX risk-tier mode, ``cross`` or ``isolated``.

        Returns:
            Mapping symbol -> OHLCV DataFrame.
        """
        validate_date_range(start_date, end_date)
        if margin_mode not in {"cross", "isolated"}:
            raise ValueError("margin_mode must be 'cross' or 'isolated'")

        timeframe = _INTERVAL_MAP.get(interval, "1d")
        # Build the exchange lazily so a full cache hit never imports ccxt or
        # opens an exchange object.
        exchange_holder: Dict[str, object] = {}

        def get_exchange(instrument_type: str):
            if instrument_type not in exchange_holder:
                exchange_holder[instrument_type] = self._get_exchange(instrument_type)
            return exchange_holder[instrument_type]

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            instrument_type = "spot"
            try:
                ccxt_symbol, instrument_type = _parse_ccxt_symbol(code)
                exchange = get_exchange(instrument_type)
                artifact = (bracket_artifacts or {}).get(code)

                def fetch_window(window_start: str, window_end: str):
                    since_ms = int(pd.Timestamp(window_start, tz="UTC").timestamp() * 1000)
                    end_ms = int((pd.Timestamp(window_end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)
                    if instrument_type in {"swap", "okx_swap"}:
                        return self._fetch_perpetual(
                            exchange, ccxt_symbol, timeframe, since_ms, end_ms,
                            is_okx=instrument_type == "okx_swap",
                            bracket_artifact=artifact, require_brackets=require_brackets,
                        )
                    return self._fetch_one(
                        exchange, ccxt_symbol, timeframe, since_ms, end_ms
                    )

                df = cached_loader_fetch(
                    source=self.name,
                    symbol=code,
                    timeframe=interval,
                    start_date=start_date,
                    end_date=end_date,
                    fields=None,
                    fetch=lambda: fetch_window(start_date, end_date),
                    fetch_range=fetch_window,
                )
                if df is not None and not df.empty:
                    if instrument_type == "okx_swap" and require_brackets:
                        df = self._attach_okx_position_tiers(
                            exchange,
                            ccxt_symbol,
                            code,
                            df,
                            margin_mode=margin_mode,
                        )
                    result[code] = df
            except Exception as exc:
                if instrument_type in {"swap", "okx_swap"} or code.strip().upper().endswith(("-PERP", "-SWAP")):
                    raise
                logger.warning("CCXT failed for %s: %s", code, exc)
        return result

    @staticmethod
    def _attach_okx_position_tiers(
        exchange,
        symbol: str,
        instrument_id: str,
        frame: pd.DataFrame,
        *,
        margin_mode: str,
    ) -> pd.DataFrame:
        """Attach a fresh public OKX maintenance-tier snapshot to cached bars.

        OKX ``maxSz`` is an instrument position-size limit in contracts. The
        engine stores position size in base-asset units, so the normalized
        schedule retains the contract cap and base amount per contract; the
        risk model selects tiers in contract units.
        """
        markets = exchange.load_markets()
        market = markets.get(symbol)
        raw_market = (market or {}).get("info") or {}
        if (
            not market
            or not market.get("swap")
            or not market.get("linear")
            or market.get("settle") != "USDT"
            or raw_market.get("instId") != instrument_id
        ):
            raise ValueError(f"current OKX contract metadata does not match {instrument_id}")
        try:
            current_metadata = {
                "contract_size_base": float(raw_market["ctVal"]),
                "contract_amount_step": float(raw_market["lotSz"]),
                "contract_min_amount": float(raw_market["minSz"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"current OKX quantity metadata is incomplete for {instrument_id}"
            ) from exc
        if (
            raw_market.get("settleCcy") != "USDT"
            or raw_market.get("ctValCcy") != symbol.split("/")[0]
        ):
            raise ValueError(f"current OKX contract units are invalid for {instrument_id}")
        for field, current_value in current_metadata.items():
            values = pd.to_numeric(frame[field], errors="coerce")
            if values.isna().any() or values.nunique() != 1 or not math.isclose(
                float(values.iloc[0]), current_value, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"cached {field} differs from current OKX contract metadata for {instrument_id}"
                )
        contract_size = current_metadata["contract_size_base"]
        if not all(pd.notna(value) and value > 0 for value in current_metadata.values()):
            raise ValueError(f"invalid OKX contract quantity metadata for {instrument_id}")

        family = symbol.split("/")[0] + "-USDT"
        params = {
            "instType": "SWAP",
            "tdMode": margin_mode,
            "instFamily": family,
        }
        request = next(
            (
                getattr(exchange, name, None)
                for name in (
                    "public_get_public_position_tiers",
                    "publicGetPublicPositionTiers",
                )
                if callable(getattr(exchange, name, None))
            ),
            None,
        )
        if request is None:
            raise ValueError("OKX public position-tiers endpoint is unavailable")
        response = request(params)
        if not isinstance(response, dict) or str(response.get("code")) != "0":
            code = response.get("code") if isinstance(response, dict) else None
            message = response.get("msg") if isinstance(response, dict) else None
            raise ValueError(
                f"OKX public position-tiers request failed for {instrument_id}: "
                f"code={code!r}, message={message!r}"
            )
        rows = response.get("data")
        if not isinstance(rows, list):
            raise ValueError(f"OKX position-tiers response is malformed for {instrument_id}")
        matched = [
            row for row in rows
            if isinstance(row, dict)
            and (
                row.get("instId") == instrument_id
                or (
                    not row.get("instId")
                    and row.get("instFamily") == family
                    and row.get("uly") == family
                )
            )
        ]
        if not matched:
            raise ValueError(
                f"OKX position-tiers response contains no exact instrument or family row "
                f"for {instrument_id}"
            )

        parsed: list[dict] = []
        for row in matched:
            try:
                tier = int(row["tier"])
                minimum = float(row["minSz"])
                maximum = float(row["maxSz"])
                rate = float(row["mmr"])
                max_leverage = float(row["maxLever"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"OKX position tier for {instrument_id} is missing a numeric field: {exc}"
                ) from exc
            if (
                tier <= 0
                or not all(pd.notna(value) and value >= 0 for value in (minimum, rate))
                or not all(pd.notna(value) and value > 0 for value in (maximum, max_leverage))
                or maximum <= minimum
            ):
                raise ValueError(f"OKX position tier contains invalid limits for {instrument_id}")
            parsed.append(
                {
                    "bracket_tier": tier,
                    "min_contracts": minimum,
                    "max_contracts": maximum,
                    "maintenance_rate": rate,
                    "max_leverage": max_leverage,
                }
            )

        parsed.sort(key=lambda item: item["bracket_tier"])
        if [item["bracket_tier"] for item in parsed] != list(range(1, len(parsed) + 1)):
            raise ValueError(f"OKX position tier sequence is incomplete for {instrument_id}")
        if any(
            current["max_contracts"] <= previous["max_contracts"]
            for previous, current in zip(parsed, parsed[1:])
        ):
            raise ValueError(f"OKX position tier caps are not strictly increasing for {instrument_id}")
        if any(
            current["maintenance_rate"] < previous["maintenance_rate"]
            for previous, current in zip(parsed, parsed[1:])
        ):
            raise ValueError(f"OKX maintenance rates decrease across tiers for {instrument_id}")

        cumulative_base = 0.0
        for index, item in enumerate(parsed):
            if index:
                previous = parsed[index - 1]
                cumulative_base += (
                    previous["max_contracts"]
                    * contract_size
                    * (item["maintenance_rate"] - previous["maintenance_rate"])
                )
            item["cumulative_maintenance_base"] = cumulative_base

        normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        version_payload = json.dumps(
            {
                "instrument_id": instrument_id,
                "margin_mode": margin_mode,
                "contract_size_base": contract_size,
                "brackets": parsed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        version = hashlib.sha256(version_payload).hexdigest()[:16]
        observed_at = pd.Timestamp.now(tz="UTC").isoformat()

        response_text = getattr(exchange, "last_http_response", None)
        if isinstance(response_text, str) and response_text:
            response_capture_mode = "ccxt_http_response_text"
        else:
            response_text = json.dumps(response, sort_keys=True, separators=(",", ":"))
            response_capture_mode = "normalized_ccxt_response_json"
        response_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()

        result = frame.copy()
        result["maintenance_brackets"] = normalized
        result["maintenance_bracket_version"] = version
        result["maintenance_bracket_source"] = "okx_public_position_tiers_current_snapshot"
        result["maintenance_bracket_observed_at"] = observed_at
        result["maintenance_bracket_margin_mode"] = margin_mode
        result.attrs["maintenance_bracket_provenance"] = {
            "source": "okx_public_position_tiers_current_snapshot",
            "endpoint": "/api/v5/public/position-tiers",
            "instrument_id": instrument_id,
            "inst_family": family,
            "margin_mode": margin_mode,
            "observed_at": observed_at,
            "version": version,
            "response_text": response_text,
            "response_sha256": response_hash,
            "response_capture_mode": response_capture_mode,
            "contract_size_base": contract_size,
            "contract_amount_step": current_metadata["contract_amount_step"],
            "contract_min_amount": current_metadata["contract_min_amount"],
        }
        return result

    @classmethod
    def _fetch_perpetual(
        cls, exchange, symbol: str, timeframe: str, since_ms: int, end_ms: int,
        *, is_okx: bool = False, bracket_artifact: dict | None = None, require_brackets: bool = False,
    ) -> pd.DataFrame:
        """Fetch aligned trade-price and mark-price candles for one USD-M swap.

        Non-OKX maintenance brackets are attached only when the caller
        supplies a validated artifact. Strict OKX callers attach the current
        public position-tier snapshot after the market-data cache lookup.
        """
        if require_brackets and bracket_artifact is None and not is_okx:
            raise ValueError(
                f"strict margin-risk fetch for {symbol} requires a maintenance-bracket "
                "artifact, but none was supplied. This loader does not perform a live "
                "authenticated bracket fetch (binanceusdm.fetch_leverage_tiers requires "
                "a Binance API key this backtest path does not carry) — pass a validated "
                "artifact via DataLoader.fetch(bracket_artifacts={code: artifact})."
            )

        if not callable(getattr(exchange, "load_markets", None)):
            raise ValueError(f"exchange metadata endpoint is unavailable for {symbol}")
        markets = exchange.load_markets()
        market = markets.get(symbol)
        if not market or not market.get("swap") or not market.get("linear") or market.get("settle") != "USDT":
            raise ValueError(f"{symbol} is not a verified linear USDT-settled perpetual swap")
        if is_okx:
            raw = market.get("info") or {}
            if raw.get("instId") != symbol.replace("/", "-").replace(":USDT", "-SWAP"):
                raise ValueError(f"OKX instrument identity does not match {symbol}")
            try:
                contract_size = float(raw["ctVal"])
                amount_step = float(raw["lotSz"])
                minimum_contracts = float(raw["minSz"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"OKX contract value, lot size, or minimum size is unavailable for {symbol}"
                ) from exc
            if not all(pd.notna(v) and v > 0 for v in (contract_size, amount_step, minimum_contracts)):
                raise ValueError(f"invalid OKX contract quantity metadata for {symbol}")
            if raw.get("settleCcy") != "USDT" or raw.get("ctValCcy") != symbol.split("/")[0]:
                raise ValueError(f"OKX contract unit is not a linear {symbol.split('/')[0]}-settled amount")
        else:
            contract_size = float(market.get("contractSize") or 0)
            amount_step = float(market.get("precision", {}).get("amount") or 0)
            minimum_contracts = float(market.get("limits", {}).get("amount", {}).get("min") or 0)
            if contract_size <= 0 or amount_step <= 0 or minimum_contracts <= 0:
                raise ValueError(f"contract quantity metadata is incomplete for {symbol}")

        trade = cls._fetch_one(exchange, symbol, timeframe, since_ms, end_ms)
        mark = cls._fetch_one(
            exchange,
            symbol,
            timeframe,
            since_ms,
            end_ms,
            params={"price": "mark"},
        )
        if trade is None or mark is None or not trade.index.equals(mark.index):
            raise ValueError(f"mark-price timestamps are incomplete or unsynchronized for {symbol}")

        delta = _TIMEFRAME_DELTA[timeframe]
        effective_end = _completed_candle_end(end_ms, timeframe)
        trade = trade.loc[trade.index < effective_end]
        mark = mark.loc[mark.index < effective_end]
        if trade.empty or mark.empty:
            raise ValueError(f"no completed perpetual candles available for {symbol}")
        _validate_perpetual_candles(trade, symbol=symbol, timeframe=timeframe, label="trade-price")
        _validate_perpetual_candles(
            mark,
            symbol=symbol,
            timeframe=timeframe,
            label="mark-price",
            require_volume=False,
        )
        expected_index = pd.date_range(
            start=pd.Timestamp(since_ms, unit="ms"),
            end=effective_end - delta,
            freq=delta,
            name=trade.index.name,
        )
        if not trade.index.equals(expected_index):
            raise ValueError(
                f"incomplete perpetual candle coverage for {symbol}: requested "
                f"[{expected_index[0]}, {expected_index[-1]}], received "
                f"[{trade.index[0]}, {trade.index[-1]}] ({len(trade)}/{len(expected_index)} bars)"
            )

        result = trade.copy()
        result["market_risk_source"] = f"ccxt:{'okx' if is_okx else getattr(exchange, 'id', 'binanceusdm')}"
        result["contract_size_base"] = contract_size
        result["contract_amount_step"] = amount_step
        result["contract_min_amount"] = minimum_contracts
        result["contract_unit"] = "contracts"
        suffix = "-SWAP" if is_okx else "-PERP"
        result["instrument_id"] = symbol.split("/")[0] + "-USDT" + suffix
        result["execution_open"] = trade["open"]
        for column in ("open", "high", "low", "close"):
            result[f"mark_{column}"] = mark[column]

        funding = cls._fetch_funding_history(exchange, symbol, since_ms, end_ms, is_okx=is_okx)
        if funding.empty:
            start_label = pd.Timestamp(since_ms, unit="ms").strftime("%Y-%m-%d")
            end_label = (pd.Timestamp(end_ms, unit="ms") - pd.Timedelta(milliseconds=1)).strftime("%Y-%m-%d")
            venue = "OKX public funding-rate history" if is_okx else "exchange funding-rate history"
            raise ValueError(
                f"{venue} returned no settlement records for {symbol} in "
                f"[{start_label}, {end_label}]; perpetual backtest stopped because "
                "funding costs cannot be verified for the requested period. "
                "Choose a period covered by the venue's funding history or provide "
                "a validated historical funding dataset."
            )
        if funding.index.has_duplicates:
            raise ValueError(f"duplicate funding settlement for {symbol}")
        if not is_okx:
            required = result.index[result.index.hour.isin(_FUNDING_HOURS)]
            missing = required.difference(funding.index)
            if not missing.empty:
                raise ValueError(
                    f"funding settlement data is missing for {symbol}: "
                    f"{', '.join(str(ts) for ts in missing)}"
                )
        outside = funding.index.difference(result.index)
        if not outside.empty:
            raise ValueError(f"funding settlement cannot be aligned to candles for {symbol}")

        result["funding_rate"] = 0.0
        result["funding_settlement_time"] = pd.NaT
        aligned = funding.index.intersection(result.index)
        if not aligned.empty:
            result.loc[aligned, "funding_rate"] = funding.loc[aligned, "funding_rate"]
            result.loc[aligned, "funding_settlement_time"] = aligned

        if bracket_artifact is not None:
            brackets, version = _validate_bracket_artifact(
                bracket_artifact, expected_symbol=symbol
            )
            result["maintenance_brackets"] = json.dumps(brackets)
            result["maintenance_bracket_version"] = version
        return result

    @staticmethod
    def _fetch_funding_history(
        exchange, symbol: str, since_ms: int, end_ms: int, *, is_okx: bool = False,
    ) -> pd.DataFrame:
        """Fetch bounded historical funding settlements for one USD-M swap."""
        import ccxt

        rows: list[dict] = []
        cursor = since_ms
        limit = 100 if is_okx else 1000
        deadline = time.monotonic() + _CCXT_FETCH_BUDGET_S
        label = f"ccxt funding fetch for {symbol}"
        completed = False

        for _ in range(200):
            check_budget(deadline, label, budget_s=_CCXT_FETCH_BUDGET_S)
            page = retry_with_budget(
                lambda: exchange.fetch_funding_rate_history(
                    symbol, since=cursor, limit=limit
                ),
                transient=ccxt.NetworkError,
                deadline=deadline,
                label=label,
            )
            if not page:
                completed = True
                break
            rows.extend(page)
            last_ts = int(page[-1]["timestamp"])
            if last_ts >= end_ms or len(page) < limit:
                completed = True
                break
            if last_ts < cursor:
                raise ValueError(f"funding pagination did not advance for {symbol}")
            cursor = last_ts + 1

        if not completed:
            raise ValueError(
                f"funding history pagination exceeded the safety page limit for {symbol}"
            )

        if not rows:
            return pd.DataFrame(
                {"funding_rate": pd.Series(dtype=float)},
                index=pd.DatetimeIndex([], name="trade_date"),
            )

        frame = pd.DataFrame({
            # Venues may return millisecond jitter around settlement time.
            # Round to the second; the event still must align to an hourly bar.
            "trade_date": pd.to_datetime(
                [row["timestamp"] for row in rows], unit="ms"
            ).round("s"),
            "funding_rate": pd.to_numeric(
                [row["fundingRate"] for row in rows], errors="raise"
            ),
        }).set_index("trade_date").sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"duplicate funding settlement for {symbol}")
        start_dt = pd.Timestamp(since_ms, unit="ms")
        end_dt = pd.Timestamp(end_ms, unit="ms")
        return frame[(frame.index >= start_dt) & (frame.index < end_dt)]

    @staticmethod
    def _fetch_one(
        exchange,
        symbol: str,
        timeframe: str,
        since_ms: int,
        end_ms: int,
        *,
        params: dict[str, str] | None = None,
    ) -> Optional[pd.DataFrame]:
        """Paginated OHLCV fetch for one symbol."""
        import ccxt

        all_rows: list = []
        cursor = since_ms
        # OKX caps candle pages at 100 records. Asking for 1000 makes its
        # short response look like end-of-history and silently truncates runs.
        limit = 100 if getattr(exchange, "id", "") == "okx" else 1000
        deadline = time.monotonic() + _CCXT_FETCH_BUDGET_S
        label = f"ccxt fetch for {symbol}"
        hit_page_cap = True
        page_cap = 2_000 if getattr(exchange, "id", "") == "okx" else 200

        for _ in range(page_cap):
            check_budget(deadline, label, budget_s=_CCXT_FETCH_BUDGET_S)
            # ``ccxt.NetworkError`` covers RequestTimeout / DDoSProtection /
            # ExchangeNotAvailable — the transient family. Anything else
            # (e.g. ``ExchangeError`` for a bad symbol) is not retried.
            def fetch_page():
                kwargs = {"since": cursor, "limit": limit}
                if params is not None:
                    if params.get("price") == "mark" and getattr(exchange, "id", "") == "okx":
                        fetch_mark_ohlcv = getattr(exchange, "fetch_mark_ohlcv", None)
                        if not callable(fetch_mark_ohlcv):
                            raise ValueError(
                                "installed CCXT OKX adapter does not support mark-price candles"
                            )
                        return fetch_mark_ohlcv(symbol, timeframe, **kwargs)
                    kwargs["params"] = params
                return exchange.fetch_ohlcv(symbol, timeframe, **kwargs)

            ohlcv = retry_with_budget(
                fetch_page,
                transient=ccxt.NetworkError,
                deadline=deadline,
                label=label,
            )
            if not ohlcv:
                hit_page_cap = False
                break
            all_rows.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts >= end_ms or len(ohlcv) < limit:
                hit_page_cap = False
                break
            cursor = last_ts + 1

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["trade_date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("trade_date").sort_index()

        start_dt = pd.Timestamp(since_ms, unit="ms")
        end_dt = pd.Timestamp(end_ms, unit="ms")
        df = df[(df.index >= start_dt) & (df.index < end_dt)]

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[["open", "high", "low", "close", "volume"]].dropna(
            subset=["open", "high", "low", "close"]
        )
        if df.empty:
            return None

        tolerance = _TIMEFRAME_DELTA.get(timeframe)
        if tolerance is None:
            raise ValueError(f"unsupported CCXT timeframe: {timeframe}")
        if hit_page_cap and df.index[-1] < end_dt - tolerance:
            raise ValueError(
                f"incomplete CCXT history for {symbol}: requested "
                f"[{start_dt}, {end_dt}), received [{df.index[0]}, {df.index[-1]}]"
            )
        return df
