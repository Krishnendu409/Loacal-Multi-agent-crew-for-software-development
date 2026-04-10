"""Lightweight local state graph for cyclical routing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

NodeFn = Callable[[dict[str, Any]], None]
RouterFn = Callable[[dict[str, Any]], str | None]


class StateGraph:
    DEFAULT_MAX_STEPS = 50  # BUG FIX: was 200, which is unreachably high and hides bugs

    def __init__(self) -> None:
        self._nodes: dict[str, NodeFn] = {}
        self._routers: dict[str, RouterFn] = {}
        self._start: str | None = None

    def add_node(self, name: str, fn: NodeFn, *, router: RouterFn | None = None) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Node name must be a non-empty string.")
        self._nodes[name] = fn
        if router is not None:
            self._routers[name] = router

    def set_start(self, name: str) -> None:
        if name not in self._nodes:
            raise ValueError(f"Unknown start node: {name!r}")
        self._start = name

    def run(self, state: dict[str, Any], *, max_steps: int = DEFAULT_MAX_STEPS) -> dict[str, Any]:
        if self._start is None:
            raise ValueError("StateGraph start node is not set.")
        current: str | None = self._start
        steps = 0
        while current is not None:
            if steps >= max_steps:
                # BUG FIX: log and break instead of crashing the entire pipeline
                logger.warning(
                    "StateGraph reached max_steps=%d at node %r; stopping gracefully.",
                    max_steps,
                    current,
                )
                break
            node = self._nodes.get(current)
            if node is None:
                raise ValueError(f"Unknown node during execution: {current!r}")
            node(state)
            router = self._routers.get(current)
            next_node = router(state) if router else None
            # BUG FIX: detect self-loops that would spin indefinitely
            if next_node == current:
                logger.warning(
                    "StateGraph detected self-loop at node %r; stopping gracefully.", current
                )
                break
            current = next_node
            steps += 1
        return state
