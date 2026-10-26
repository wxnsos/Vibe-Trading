"""Historical Binance USD-M data-contract tests."""

from __future__ import annotations

import hashlib
import json

import pytest
import pandas as pd

from backtest.loaders.base import make_loader_cache_key
from backtest.loaders.ccxt_loader import (
    _parse_ccxt_symbol,
    _validate_bracket_artifact,
    _validate_perpetual_candles,
)
from tests.loader_contract import assert_loader_contract


def _hourly_rows(opens: list[float]) -> list[list[float]]:
    start = int(pd.Timestamp("2024-01-01 00:00:00").timestamp() * 1000)
    hour = 3_600_000
    opens = (opens + [opens[-1]] * 24)[:24]
    return [
        [start + i * hour, value, value + 2, value - 2, value + 1, 10 + i]
        for i, value in enumerate(opens)
    ]


class _PerpetualExchange:
    """Fake exchange with no ``fetch_leverage_tiers`` — proves the loader
    never calls the authenticated bracket endpoint. If any code path tried
    to, this fake would raise ``AttributeError`` instead of silently
    succeeding.
    """

    def __init__(
        self,
        *,
        mark_rows: list[list[float]] | None = None,
        funding_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.id = "okx"
        self.trade_rows = _hourly_rows([100.0, 101.0])
        self.mark_rows = mark_rows if mark_rows is not None else _hourly_rows([99.0, 100.0])
        start = int(pd.Timestamp("2024-01-01 00:00:00").timestamp() * 1000)
        self.funding_rows = (
            funding_rows
            if funding_rows is not None
            else [{"timestamp": start, "fundingRate": 0.0}]
        )
        self.calls: list[dict[str, object]] = []

    def load_markets(self):
        return {
            "BTC/USDT:USDT": {
                "swap": True,
                "linear": True,
                "settle": "USDT",
                "contractSize": 0.01,
                "precision": {"amount": 1},
                "limits": {"amount": {"min": 1}},
                "info": {
                    "instId": "BTC-USDT-SWAP",
                    "ctVal": "0.01",
                    "ctValCcy": "BTC",
                    "settleCcy": "USDT",
                    "lotSz": "1",
                    "minSz": "1",
                },
            },
            "ETH/USDT:USDT": {
                "swap": True,
                "linear": True,
                "settle": "USDT",
                "contractSize": 0.1,
                "precision": {"amount": 1},
                "limits": {"amount": {"min": 1}},
                "info": {
                    "instId": "ETH-USDT-SWAP",
                    "ctVal": "0.1",
                    "ctValCcy": "ETH",
                    "settleCcy": "USDT",
                    "lotSz": "1",
                    "minSz": "1",
                },
            },
        }

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        self.calls.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "since": since,
            "limit": limit,
            "params": params,
        })
        return self.mark_rows if params == {"price": "mark"} else self.trade_rows

    def fetch_mark_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return self.fetch_ohlcv(symbol, timeframe, since=since, limit=limit, params={"price": "mark"})

    def fetch_funding_rate_history(self, symbol, since=None, limit=None):
        self.calls.append({
            "funding_symbol": symbol,
            "since": since,
            "limit": limit,
        })
        return self.funding_rows


_DEFAULT_BRACKET_RECORDS = [
    {
        "bracket_tier": 1,
        "notional_cap": 50_000.0,
        "maintenance_rate": 0.004,
        "cumulative_maintenance_amount": 0.0,
    },
    {
        "bracket_tier": 2,
        "notional_cap": 250_000.0,
        "maintenance_rate": 0.005,
        "cumulative_maintenance_amount": 50.0,
    },
]


def _bracket_content_hash(records: list[dict]) -> str:
    """Mirror ``_validate_bracket_artifact``'s canonical hashing exactly."""
    blob = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _make_bracket_artifact(
    *, symbol: str = "BTC/USDT:USDT", brackets: list[dict] | None = None, **overrides,
) -> dict:
    records = _DEFAULT_BRACKET_RECORDS if brackets is None else brackets
    artifact = {
        "schema_version": 1,
        "symbol": symbol,
        "provenance_timestamp": "2024-01-01T00:00:00Z",
        "brackets": records,
        "content_hash": _bracket_content_hash(records),
    }
    artifact.update(overrides)
    return artifact


def test_spot_symbol_keeps_existing_ccxt_contract() -> None:
    assert _parse_ccxt_symbol("BTC-USDT") == ("BTC/USDT", "spot")


def test_perpetual_symbol_maps_to_binance_usdm_contract() -> None:
    assert _parse_ccxt_symbol("BTC-USDT-PERP") == ("BTC/USDT:USDT", "swap")


def test_okx_swap_symbol_maps_to_unified_contract() -> None:
    assert _parse_ccxt_symbol("ETH-USDT-SWAP") == ("ETH/USDT:USDT", "okx_swap")


def test_perpetual_candle_quality_gate_rejects_gaps_and_bad_ohlc() -> None:
    index = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"])
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0], "high": [102.0, 102.0],
            "low": [98.0, 98.0], "close": [101.0, 101.0], "volume": [1.0, 1.0],
        },
        index=index,
    )
    with pytest.raises(ValueError, match="not continuous"):
        _validate_perpetual_candles(frame, symbol="ETH/USDT:USDT", timeframe="1h", label="trade-price")

    frame.index = pd.date_range("2024-01-01", periods=2, freq="h")
    frame.loc[frame.index[0], "high"] = 99.0
    with pytest.raises(ValueError, match="invalid OHLCV"):
        _validate_perpetual_candles(frame, symbol="ETH/USDT:USDT", timeframe="1h", label="trade-price")


def test_mark_price_candles_may_omit_volume() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0], "high": [102.0, 103.0],
            "low": [98.0, 99.0], "close": [101.0, 102.0],
            "volume": [float("nan"), float("nan")],
        },
        index=index,
    )
    _validate_perpetual_candles(
        frame,
        symbol="ETH/USDT:USDT",
        timeframe="1h",
        label="mark-price",
        require_volume=False,
    )


@pytest.mark.parametrize("code", ["BTC-PERP", "-USDT-PERP", "BTC--PERP"])
def test_malformed_perpetual_symbol_is_rejected(code: str) -> None:
    with pytest.raises(ValueError, match="USD-M perpetual symbol"):
        _parse_ccxt_symbol(code)


def test_spot_and_perpetual_cache_keys_cannot_collide() -> None:
    common = {
        "source": "ccxt",
        "timeframe": "1H",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "fields": None,
    }
    spot = make_loader_cache_key(symbol="BTC-USDT", **common)
    perpetual = make_loader_cache_key(symbol="BTC-USDT-PERP", **common)
    assert spot != perpetual


def test_perpetual_fetch_separates_execution_and_mark_prices(monkeypatch) -> None:
    exchange = _PerpetualExchange()
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    frame = DataLoader().fetch(
        ["BTC-USDT-PERP"], "2024-01-01", "2024-01-01", interval="1H"
    )["BTC-USDT-PERP"]

    assert frame["execution_open"].iloc[:2].tolist() == [100.0, 101.0]
    assert frame["mark_open"].iloc[:2].tolist() == [99.0, 100.0]
    assert frame["mark_high"].iloc[:2].tolist() == [101.0, 102.0]
    assert frame["mark_low"].iloc[:2].tolist() == [97.0, 98.0]
    assert frame["mark_close"].iloc[:2].tolist() == [100.0, 101.0]
    assert frame["contract_size_base"].iloc[:2].tolist() == [0.01, 0.01]
    assert frame["contract_amount_step"].iloc[:2].tolist() == [1.0, 1.0]
    assert frame["contract_min_amount"].iloc[:2].tolist() == [1.0, 1.0]
    assert frame["instrument_id"].iloc[:2].tolist() == ["BTC-USDT-SWAP"] * 2
    assert exchange.calls[0]["symbol"] == "BTC/USDT:USDT"
    assert exchange.calls[0]["params"] is None
    assert exchange.calls[1]["params"] == {"price": "mark"}


def test_perpetual_fetch_rejects_unsynchronized_mark_rows(monkeypatch) -> None:
    mark_rows = _hourly_rows([99.0, 100.0])
    mark_rows[1][0] += 60_000
    exchange = _PerpetualExchange(mark_rows=mark_rows)
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    with pytest.raises(ValueError, match="mark-price timestamps"):
        DataLoader().fetch(
            ["BTC-USDT-PERP"], "2024-01-01", "2024-01-01", interval="1H"
        )


def test_perpetual_fetch_aligns_explicit_zero_funding_settlement(monkeypatch) -> None:
    exchange = _PerpetualExchange()
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    frame = DataLoader().fetch(
        ["BTC-USDT-PERP"], "2024-01-01", "2024-01-01", interval="1H"
    )["BTC-USDT-PERP"]

    assert frame["funding_rate"].iloc[:2].tolist() == [0.0, 0.0]
    assert frame["funding_settlement_time"].iloc[0] == pd.Timestamp("2024-01-01")
    assert pd.isna(frame["funding_settlement_time"].iloc[1])


def test_okx_swap_accepts_recorded_dynamic_settlement_time(monkeypatch) -> None:
    settlement = int(pd.Timestamp("2024-01-01 01:00:00").timestamp() * 1000)
    exchange = _PerpetualExchange(funding_rows=[{"timestamp": settlement, "fundingRate": 0.0003}])
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    frame = DataLoader().fetch(
        ["ETH-USDT-SWAP"], "2024-01-01", "2024-01-01", interval="1H"
    )["ETH-USDT-SWAP"]
    assert frame.loc[pd.Timestamp("2024-01-01 01:00:00"), "funding_rate"] == 0.0003
    assert frame.loc[pd.Timestamp("2024-01-01 01:00:00"), "funding_settlement_time"] == pd.Timestamp("2024-01-01 01:00:00")


def test_perpetual_fetch_rejects_missing_required_funding_settlement(monkeypatch) -> None:
    exchange = _PerpetualExchange(funding_rows=[])
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    with pytest.raises(
        ValueError,
        match=r"OKX public funding-rate history returned no settlement records.*\[2024-01-01, 2024-01-01\]",
    ):
        DataLoader().fetch(
            ["ETH-USDT-SWAP"], "2024-01-01", "2024-01-01", interval="1H"
        )


def test_perpetual_fetch_rejects_duplicate_funding_settlement(monkeypatch) -> None:
    start = int(pd.Timestamp("2024-01-01 00:00:00").timestamp() * 1000)
    exchange = _PerpetualExchange(funding_rows=[
        {"timestamp": start, "fundingRate": 0.0001},
        {"timestamp": start, "fundingRate": 0.0002},
    ])
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    with pytest.raises(ValueError, match="duplicate funding settlement"):
        DataLoader().fetch(
            ["BTC-USDT-PERP"], "2024-01-01", "2024-01-01", interval="1H"
        )


def test_perpetual_fetch_has_no_bracket_columns_without_artifact(monkeypatch) -> None:
    """Normal perpetual fetch stays zero-credential: no artifact in, no
    bracket columns out, and no bracket-endpoint call ever attempted (the
    fake exchange has no ``fetch_leverage_tiers`` at all)."""
    exchange = _PerpetualExchange()
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    frame = DataLoader().fetch(
        ["BTC-USDT-PERP"], "2024-01-01", "2024-01-01", interval="1H"
    )["BTC-USDT-PERP"]

    assert_loader_contract(frame, context="ccxt perpetual 1H")
    assert "maintenance_brackets" not in frame.columns
    assert "maintenance_bracket_version" not in frame.columns
    assert all("bracket" not in str(call) for call in exchange.calls)


def test_perpetual_fetch_attaches_valid_bracket_artifact(monkeypatch) -> None:
    exchange = _PerpetualExchange()
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    artifact = _make_bracket_artifact()
    frame = DataLoader().fetch(
        ["BTC-USDT-PERP"], "2024-01-01", "2024-01-01", interval="1H",
        bracket_artifacts={"BTC-USDT-PERP": artifact},
    )["BTC-USDT-PERP"]

    versions = frame["maintenance_bracket_version"].unique().tolist()
    assert versions == [artifact["content_hash"]]
    assert json.loads(frame["maintenance_brackets"].iloc[0]) == _DEFAULT_BRACKET_RECORDS
    assert frame["maintenance_brackets"].nunique() == 1


def test_strict_perpetual_fetch_without_artifact_fails_closed_before_any_call(monkeypatch) -> None:
    exchange = _PerpetualExchange()
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    with pytest.raises(ValueError, match="requires a maintenance-bracket artifact"):
        DataLoader().fetch(
            ["BTC-USDT-PERP"], "2024-01-01", "2024-01-01", interval="1H",
            require_brackets=True,
        )
    assert exchange.calls == []


def test_strict_perpetual_fetch_with_valid_artifact_succeeds(monkeypatch) -> None:
    exchange = _PerpetualExchange()
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    artifact = _make_bracket_artifact()
    frame = DataLoader().fetch(
        ["BTC-USDT-PERP"], "2024-01-01", "2024-01-01", interval="1H",
        bracket_artifacts={"BTC-USDT-PERP": artifact}, require_brackets=True,
    )["BTC-USDT-PERP"]

    assert frame["maintenance_bracket_version"].iloc[0] == artifact["content_hash"]


def test_bracket_artifact_accepts_optional_notional_coefficient() -> None:
    records = [
        {**_DEFAULT_BRACKET_RECORDS[0], "notional_coefficient": 1.5},
        _DEFAULT_BRACKET_RECORDS[1],
    ]
    artifact = _make_bracket_artifact(brackets=records)
    brackets, version = _validate_bracket_artifact(artifact, expected_symbol="BTC/USDT:USDT")
    assert brackets[0]["notional_coefficient"] == 1.5
    assert "notional_coefficient" not in brackets[1]
    assert version == artifact["content_hash"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"symbol": "ETH/USDT:USDT"}, "symbol mismatch"),
        ({"provenance_timestamp": ""}, "provenance_timestamp"),
        ({"provenance_timestamp": "not-a-timestamp"}, "provenance_timestamp"),
        ({"brackets": []}, "no brackets"),
        ({"content_hash": "deadbeefdeadbeef"}, "content_hash mismatch"),
    ],
)
def test_bracket_artifact_rejects_invalid_fields(overrides: dict, match: str) -> None:
    artifact = _make_bracket_artifact(**overrides)
    with pytest.raises(ValueError, match=match):
        _validate_bracket_artifact(artifact, expected_symbol="BTC/USDT:USDT")


def test_bracket_artifact_rejects_tier_missing_required_field() -> None:
    artifact = _make_bracket_artifact(brackets=[
        {"bracket_tier": 1, "notional_cap": 50_000.0, "maintenance_rate": 0.004},
    ])
    with pytest.raises(ValueError, match="missing a required field"):
        _validate_bracket_artifact(artifact, expected_symbol="BTC/USDT:USDT")


def test_bracket_artifact_rejects_non_numeric_notional_coefficient() -> None:
    records = [{**_DEFAULT_BRACKET_RECORDS[0], "notional_coefficient": "not-a-number"}]
    artifact = _make_bracket_artifact(brackets=records)
    with pytest.raises(ValueError, match="notional_coefficient must be numeric"):
        _validate_bracket_artifact(artifact, expected_symbol="BTC/USDT:USDT")


def test_bracket_artifact_rejects_non_monotonic_brackets() -> None:
    artifact = _make_bracket_artifact(brackets=[
        {
            "bracket_tier": 1, "notional_cap": 250_000.0,
            "maintenance_rate": 0.004, "cumulative_maintenance_amount": 0.0,
        },
        {
            "bracket_tier": 2, "notional_cap": 50_000.0,
            "maintenance_rate": 0.005, "cumulative_maintenance_amount": 50.0,
        },
    ])
    with pytest.raises(ValueError, match="not strictly increasing"):
        _validate_bracket_artifact(artifact, expected_symbol="BTC/USDT:USDT")


def test_perpetual_fetch_rounds_funding_settlement_jitter(monkeypatch) -> None:
    """Binance returns settlement timestamps with millisecond jitter
    (e.g. ``08:00:00.011``); the loader must round them to the second so
    they align with bar timestamps instead of raising the
    missing-settlement error."""
    start = int(pd.Timestamp("2024-01-01 00:00:00").timestamp() * 1000)
    exchange = _PerpetualExchange(
        funding_rows=[{"timestamp": start + 11, "fundingRate": 0.0003}]
    )
    monkeypatch.setattr(
        "backtest.loaders.ccxt_loader.DataLoader._get_exchange",
        lambda _self, instrument_type="spot": exchange,
    )

    from backtest.loaders.ccxt_loader import DataLoader

    frame = DataLoader().fetch(
        ["BTC-USDT-PERP"], "2024-01-01", "2024-01-01", interval="1H"
    )["BTC-USDT-PERP"]

    assert frame["funding_rate"].iloc[:2].tolist() == [0.0003, 0.0]
    assert frame["funding_settlement_time"].iloc[0] == pd.Timestamp("2024-01-01")
