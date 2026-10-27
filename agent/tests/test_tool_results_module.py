"""Pin the re-export contract of the tool-results extraction (issue #1624).

``loop.py`` used to define these helpers inline; they now live in
``src.agent.tool_results`` and are re-exported by ``loop.py`` so every
existing ``from src.agent.loop import ...`` keeps resolving. These tests
fail if the two modules ever drift apart, e.g. someone deletes a re-export
or redefines a helper in ``loop.py`` instead of moving it.
"""

import pytest

import src.agent.loop as loop
import src.agent.tool_results as tool_results

MOVED_CONSTANTS = [
    "_FORCED_TEXT_TOOL_CALL_RE",
    "_DSML_BAR_TOOL_CALL_RE",
    "_TARGET_PATH_RE",
    "_TARGET_ACTION_RE",
]

MOVED_FUNCTIONS = [
    "_failure_code",
    "_is_tool_success",
    "_looks_like_tool_call_syntax",
    "_requires_backtest_execution",
    "_named_target_paths",
    "_normalize_tool_run_dir",
    "_previously_archived",
    "_archive_backtest_result",
]


@pytest.mark.parametrize("name", MOVED_CONSTANTS + MOVED_FUNCTIONS)
def test_loop_reexports_moved_name(name):
    assert getattr(loop, name) is getattr(tool_results, name)


def test_moved_surface_is_complete():
    public_helpers = [
        n
        for n in dir(tool_results)
        if not n.startswith("__") and getattr(tool_results, n) is not None
    ]
    # The new module must not silently grow an unmoved consumer surface.
    assert set(MOVED_CONSTANTS + MOVED_FUNCTIONS) <= set(public_helpers)


@pytest.mark.parametrize(
    "text",
    [
        "请运行 ETH 策略的历史回测",
        "帮我执行回测",
        "Run a backtest for this strategy.",
        "请运行回测；如果数据不完整则停止回测并说明原因。",
    ],
)
def test_explicit_backtest_requests_are_detected(text: str) -> None:
    assert tool_results._requires_backtest_execution(text)


@pytest.mark.parametrize(
    "text",
    [
        "看下为什么这次没有成功执行回测",
        "本次不要运行回测，只解释现有日志",
        "Don't run a backtest; explain the existing logs only.",
        "如果数据不完整，请停止回测，不要生成收益结论",
        "帮我调整程序确保回测可以正常进行",
    ],
)
def test_diagnostics_and_negated_requests_do_not_require_execution(text: str) -> None:
    assert not tool_results._requires_backtest_execution(text)
