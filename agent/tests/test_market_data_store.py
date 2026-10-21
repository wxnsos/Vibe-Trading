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


def test_current_date_request_caches_only_completed_bars(tmp_path):
    store = MarketDataStore(tmp_path / "market.sqlite3")
    today = date.today().isoformat()
    start = (date.today() - timedelta(days=1)).isoformat()
    frame = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.DatetimeIndex([
            pd.Timestamp(today) - pd.Timedelta(hours=1),
            pd.Timestamp(today),
        ]),
    )
    store.put(
        source="fixture", symbol="ABC", timeframe="1d", start_date=start,
        end_date=today, fields=["close"], frame=frame,
    )
    datasets = store.list_datasets()
    assert len(datasets) == 1
    assert datasets[0]["end_date"] == (date.today() - timedelta(days=1)).isoformat()
    assert datasets[0]["row_count"] == 1
    cached = store.get(
        source="fixture", symbol="ABC", timeframe="1d", start_date=start,
        end_date=(date.today() - timedelta(days=1)).isoformat(), fields=["close"],
    )
    assert cached is not None
    assert cached["close"].tolist() == [1.0]


def test_current_date_request_stores_only_the_settled_prefix(tmp_path):
    store = MarketDataStore(tmp_path / "market.sqlite3")
    today = date.today()
    yesterday = today - timedelta(days=1)
    frame = pd.DataFrame(
        {"close": [10.0, 11.0]},
        index=pd.DatetimeIndex([
            pd.Timestamp(yesterday.isoformat(), tz="UTC") + pd.Timedelta(hours=23),
            pd.Timestamp(today.isoformat(), tz="UTC"),
        ], name="timestamp"),
    )

    store.put(
        source="okx", symbol="ETH-USDT-SWAP", timeframe="1h",
        start_date=(today - timedelta(days=90)).isoformat(),
        end_date=today.isoformat(), fields=None, frame=frame,
    )

    datasets = store.list_datasets()
    assert len(datasets) == 1
    assert datasets[0]["end_date"] == yesterday.isoformat()
    assert datasets[0]["row_count"] == 1
    loaded = store.get(
        source="okx", symbol="ETH-USDT-SWAP", timeframe="1h",
        start_date=(today - timedelta(days=90)).isoformat(),
        end_date=yesterday.isoformat(), fields=None,
    )
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, frame.iloc[:1], check_freq=False)


def test_current_date_request_reuses_prefix_and_fetches_only_tail(tmp_path, monkeypatch):
    store = MarketDataStore(tmp_path / "market.sqlite3")
    monkeypatch.setattr("src.market_data_store.MarketDataStore", lambda: store)
    today = date.today()
    yesterday = today - timedelta(days=1)
    start = (today - timedelta(days=90)).isoformat()
    prefix = pd.DataFrame(
        {"close": [10.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")], name="timestamp"),
    )
    store.put(
        source="okx", symbol="ETH-USDT-SWAP", timeframe="1h",
        start_date=start, end_date=yesterday.isoformat(), fields=None, frame=prefix,
    )
    tail = pd.DataFrame(
        {"close": [11.0]},
        index=pd.DatetimeIndex([pd.Timestamp(today.isoformat(), tz="UTC")], name="timestamp"),
    )
    fetched_windows = []

    def fetch_range(fetch_start, fetch_end):
        fetched_windows.append((fetch_start, fetch_end))
        return tail

    def fail_full_fetch():
        raise AssertionError("the cached prefix should avoid a full-range download")

    actual = cached_loader_fetch(
        source="okx", symbol="ETH-USDT-SWAP", timeframe="1h",
        start_date=start, end_date=today.isoformat(), fields=None,
        fetch=fail_full_fetch, fetch_range=fetch_range,
    )

    assert fetched_windows == [(today.isoformat(), today.isoformat())]
    assert actual is not None
    assert actual["close"].tolist() == [10.0, 11.0]
    assert store.list_datasets()[0]["end_date"] == yesterday.isoformat()
