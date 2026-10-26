"""Strict perpetual runs preserve fetched data when a risk input is missing."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from backtest import runner


def _strict_config() -> dict:
    return {
        "perpetual_strict": True,
        "source": "ccxt",
        "codes": ["ETH-USDT-SWAP"],
        "interval": "1H",
        "funding_mode": "data",
        "margin_mode": "cross",
        "initial_cash": 10_000,
        "leverage": 5,
        "taker_rate": 0.0005,
        "slippage": 0.0005,
        "stop_loss_pct": 0.01,
    }


def _frame_without_brackets() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2026-10-25T00:00:00Z"], name="trade_date")
    return pd.DataFrame(
        {
            "open": [3800.0],
            "high": [3820.0],
            "low": [3780.0],
            "close": [3810.0],
            "volume": [120.0],
            "instrument_id": ["ETH-USDT-SWAP"],
            "contract_unit": ["contracts"],
            "contract_size_base": [0.1],
            "contract_amount_step": [1.0],
            "contract_min_amount": [1.0],
            "execution_open": [3800.0],
            "mark_open": [3799.0],
            "mark_high": [3821.0],
            "mark_low": [3779.0],
            "mark_close": [3809.0],
            "funding_rate": [0.0001],
            "funding_settlement_time": [pd.Timestamp("2026-10-25T00:00:00Z")],
        },
        index=index,
    )


def test_strict_config_does_not_block_before_missing_bracket_diagnostic() -> None:
    runner._validate_perpetual_config(_strict_config())


def test_strict_fetch_attempts_public_data_without_bracket_artifact(monkeypatch) -> None:
    received: dict = {}
    frame = _frame_without_brackets()

    class RecordingLoader:
        name = "ccxt"

        def fetch(self, codes, start_date, end_date, **kwargs):
            received.update(kwargs)
            return {"ETH-USDT-SWAP": frame}

    monkeypatch.setattr(runner, "_get_loader", lambda _source: RecordingLoader)
    result = runner.fetch_data_map(
        {
            **_strict_config(),
            "start_date": "2026-10-25",
            "end_date": "2026-10-25",
        }
    )

    assert list(result.data_map) == ["ETH-USDT-SWAP"]
    assert received["require_brackets"] is True
    assert received["margin_mode"] == "cross"
    assert received["bracket_artifacts"] == {}


def test_preflight_persists_normalized_data_and_names_missing_brackets(tmp_path) -> None:
    frame = _frame_without_brackets()

    with pytest.raises(ValueError, match="missing fields.*maintenance_brackets"):
        runner._perpetual_preflight(
            _strict_config(), {"ETH-USDT-SWAP": frame}, tmp_path
        )

    report_path = tmp_path / "artifacts" / "perpetual_preflight.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = report["datasets"]["ETH-USDT-SWAP"]
    data_path = tmp_path / dataset["normalized_frame_path"]
    data = data_path.read_bytes()

    assert data
    assert hashlib.sha256(data).hexdigest() == dataset["normalized_frame_sha256"]
    assert dataset["rows"] == 1
    assert "maintenance_brackets" in dataset["missing_required_fields"]
    assert dataset["source_raw_response_hash"] is None
    assert report["status"] == "failed"


def test_preflight_archives_and_validates_okx_position_tier_snapshot(tmp_path) -> None:
    frame = _frame_without_brackets()
    records = [
        {
            "bracket_tier": 1,
            "min_contracts": 0.0,
            "max_contracts": 100.0,
            "maintenance_rate": 0.01,
            "max_leverage": 100.0,
            "cumulative_maintenance_base": 0.0,
        }
    ]
    version_payload = json.dumps(
        {
            "instrument_id": "ETH-USDT-SWAP",
            "margin_mode": "cross",
            "contract_size_base": 0.1,
            "brackets": records,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    version = hashlib.sha256(version_payload).hexdigest()[:16]
    response_text = json.dumps(
        {
            "code": "0",
            "data": [
                {
                    "instId": "",
                    "instFamily": "ETH-USDT",
                    "uly": "ETH-USDT",
                    "tier": "1",
                    "minSz": "0",
                    "maxSz": "100",
                    "mmr": "0.01",
                    "maxLever": "100",
                }
            ],
        },
        separators=(",", ":"),
    )
    response_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
    frame["maintenance_brackets"] = json.dumps(records, sort_keys=True, separators=(",", ":"))
    frame["maintenance_bracket_version"] = version
    frame["maintenance_bracket_source"] = "okx_public_position_tiers_current_snapshot"
    frame["maintenance_bracket_observed_at"] = "2026-10-25T00:00:00+00:00"
    frame["maintenance_bracket_margin_mode"] = "cross"
    frame.attrs["maintenance_bracket_provenance"] = {
        "source": "okx_public_position_tiers_current_snapshot",
        "endpoint": "/api/v5/public/position-tiers",
        "instrument_id": "ETH-USDT-SWAP",
        "inst_family": "ETH-USDT",
        "margin_mode": "cross",
        "observed_at": "2026-10-25T00:00:00+00:00",
        "version": version,
        "response_text": response_text,
        "response_sha256": response_hash,
        "response_capture_mode": "ccxt_http_response_text",
        "contract_size_base": 0.1,
        "contract_amount_step": 1.0,
        "contract_min_amount": 1.0,
    }

    runner._perpetual_preflight(_strict_config(), {"ETH-USDT-SWAP": frame}, tmp_path)

    report = json.loads(
        (tmp_path / "artifacts" / "perpetual_preflight.json").read_text(encoding="utf-8")
    )
    snapshot = report["datasets"]["ETH-USDT-SWAP"]["maintenance_tier_snapshot"]
    raw_copy = tmp_path / snapshot["response_path"]
    assert report["status"] == "passed_with_limitations"
    assert snapshot["history_scope"] == "current snapshot applied to all bars in this run"
    assert raw_copy.read_text(encoding="utf-8") == response_text
    assert hashlib.sha256(raw_copy.read_bytes()).hexdigest() == response_hash
