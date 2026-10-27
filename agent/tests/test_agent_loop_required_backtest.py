"""Explicit backtest requests must reach the backtest tool before success."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.agent.loop import AgentLoop
from src.agent.tools import BaseTool, ToolRegistry


class _BacktestTool(BaseTool):
    name = "backtest"
    description = "Offline backtest test double."
    parameters = {
        "type": "object",
        "properties": {"run_dir": {"type": "string"}},
        "required": ["run_dir"],
    }
    is_readonly = False

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls = 0

    def execute(self, **_: Any) -> str:
        self.calls += 1
        return json.dumps(self.result)


class _ScriptedLLM:
    model_name = "offline-test"

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.seen_messages: list[list[dict[str, Any]]] = []
        self.seen_tools: list[Any] = []

    def stream_chat(self, messages: list[dict[str, Any]], *, tools: Any = None, **_: Any) -> Any:
        self.seen_messages.append(list(messages))
        self.seen_tools.append(tools)
        return self.responses.pop(0)

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return self.stream_chat(messages, **kwargs)


def _answer(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        reasoning_content=None,
        tool_calls=[],
        has_tool_calls=False,
    )


def _backtest_call() -> SimpleNamespace:
    return SimpleNamespace(
        content="",
        reasoning_content=None,
        tool_calls=[
            SimpleNamespace(
                id="backtest-call-1",
                name="backtest",
                arguments={"run_dir": "."},
            )
        ],
        has_tool_calls=True,
    )


def _build_loop(tmp_path: Path, llm: _ScriptedLLM, registry: ToolRegistry) -> AgentLoop:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent = AgentLoop(registry=registry, llm=llm, max_iterations=1)
    agent.memory.run_dir = str(run_dir)
    return agent


def test_premature_final_is_nudged_until_backtest_is_called(tmp_path: Path) -> None:
    tool = _BacktestTool({"status": "ok", "exit_code": 0})
    registry = ToolRegistry()
    registry.register(tool)
    llm = _ScriptedLLM(
        [
            _answer("No research goal was found, so I am stopping."),
            _backtest_call(),
            _answer("The requested backtest finished successfully."),
        ]
    )
    agent = _build_loop(tmp_path, llm, registry)

    result = agent.run("Please run one historical backtest.")

    assert tool.calls == 1
    assert result["status"] == "success"
    assert result["iterations"] == 3
    assert "backtest` tool has not been called" in llm.seen_messages[1][-1]["content"]


def test_backtest_preflight_error_remains_a_failed_run(tmp_path: Path) -> None:
    tool = _BacktestTool(
        {"status": "error", "error": "missing maintenance brackets"}
    )
    registry = ToolRegistry()
    registry.register(tool)
    llm = _ScriptedLLM(
        [
            _answer("No research goal was found, so I am stopping."),
            _backtest_call(),
            _answer("The perpetual preflight failed: missing maintenance brackets."),
        ]
    )
    agent = _build_loop(tmp_path, llm, registry)

    result = agent.run("Please run one historical backtest.")

    assert tool.calls == 1
    assert result["status"] == "failed"
    assert result["reason"] == (
        "requested backtest did not complete successfully: "
        "missing maintenance brackets"
    )
    assert "missing maintenance brackets" in result["content"]


def test_backtest_diagnostic_does_not_force_a_new_run(tmp_path: Path) -> None:
    llm = _ScriptedLLM([_answer("The previous backtest was not executed.")])
    agent = _build_loop(tmp_path, llm, ToolRegistry())

    result = agent.run("Look into why the backtest was not executed.")

    assert result["status"] == "success"
    assert result["max_iterations"] == 1
    assert len(llm.seen_messages) == 1


def test_missing_backtest_tool_cannot_produce_success_for_execution_request(
    tmp_path: Path,
) -> None:
    llm = _ScriptedLLM([_answer("The backtest tool is unavailable.")])
    agent = _build_loop(tmp_path, llm, ToolRegistry())

    result = agent.run("Please run a historical backtest.")

    assert result["status"] == "failed"
    assert result["reason"] == "requested backtest was not invoked"
