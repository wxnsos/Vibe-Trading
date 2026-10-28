"""The isolated backtest child must not read private dotenv files."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

from backtest.runner import _load_dotenv_for_execution


def test_isolated_backtest_child_skips_dotenv_before_engine_validation(
    monkeypatch, tmp_path: Path
) -> None:
    """A mode-0600 project .env must not prevent sandbox startup."""
    env_file = tmp_path / ".env"
    env_file.write_text("PRIVATE_TEST_VALUE=must-not-be-read\n", encoding="utf-8")
    env_file.chmod(0o600)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VIBE_TRADING_ISOLATED_BACKTEST_CHILD", "1")

    called = []
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: called.append(True)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    _load_dotenv_for_execution()

    assert called == []
    assert os.environ.get("PRIVATE_TEST_VALUE") is None


def test_direct_backtest_invocation_still_loads_dotenv(monkeypatch) -> None:
    monkeypatch.delenv("VIBE_TRADING_ISOLATED_BACKTEST_CHILD", raising=False)
    called = []
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: called.append(True)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    _load_dotenv_for_execution()

    assert called == [True]
