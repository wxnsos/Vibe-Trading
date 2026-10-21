from datetime import date, timedelta

import pandas as pd

from src.market_data_store import MarketDataStore
from backtest.loaders.base import cached_loader_fetch


def _past_day() -> str:
    return (date.today() - timedelta(days=10)).isoformat()


def test_store_round_trips_datetime_index_and_columns(tmp_path):
    store = MarketDataStore(tmp_path / "market.sqlite3")
    day = _past_day()
    frame = pd.DataFrame(
        {"close": [100.5, 101.25], "settlement_time": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T08:00Z"])},
        index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC", name="timestamp"),
    )
    frame.attrs["market"] = "perpetual"
    store.put(
        source="okx", symbol="ETH-USDT-SWAP", timeframe="1h",
        start_date=day, end_date=day, fields=["close", "settlement_time"],
        source_revision="test-v1", frame=frame,
    )

    loaded = store.get(
        source="okx", symbol="ETH-USDT-SWAP", timeframe="1h",
        start_date=day, end_date=day, fields=["settlement_time", "close"],
        source_revision="test-v1",
    )
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, frame, check_freq=False)
    assert loaded.attrs["market"] == "perpetual"
    assert store.list_datasets()[0]["row_count"] == 2


def test_cached_fetch_reuses_sqlite_without_calling_fetch(tmp_path, monkeypatch):
    store = MarketDataStore(tmp_path / "market.sqlite3")
    monkeypatch.setattr("src.market_data_store.MarketDataStore", lambda: store)
    day = _past_day()
    expected = pd.DataFrame({"close": [10.0]}, index=pd.DatetimeIndex(["2024-01-01"], name="date"))
    store.put(
        source="fixture", symbol="ABC", timeframe="1d", start_date=day,
        end_date=day, fields=["close"], source_revision="v1", frame=expected,
    )

    def fail_fetch():
        raise AssertionError("cache hit must not fetch")

    actual = cached_loader_fetch(
        source="fixture", symbol="ABC", timeframe="1d", start_date=day,
        end_date=day, fields=["close"], source_revision="v1", fetch=fail_fetch,
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_current_date_is_not_cached(tmp_path):
    store = MarketDataStore(tmp_path / "market.sqlite3")
    today = date.today().isoformat()
    frame = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp.today()]))
    store.put(
        source="fixture", symbol="ABC", timeframe="1d", start_date=today,
        end_date=today, fields=["close"], frame=frame,
    )
    assert store.get(
        source="fixture", symbol="ABC", timeframe="1d", start_date=today,
        end_date=today, fields=["close"],
    ) is None
    assert store.list_datasets() == []
