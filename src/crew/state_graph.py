"""Lightweight local state graph for cyclical routing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

NodeFn = Callable[[dict[str, Any]], None]
RouterFn = Callable[[dict[str, Any]], str | None]


class StateGraph:
    DEFAULT_MAX_STEPS = 200

    def __init__(self) -> None:
        self._nodes: dict[str, NodeFn] = {}
        self._routers: dict[str, RouterFn] = {}
        self._start: str | None = None

    def add_node(self, name: str, fn: NodeFn, *, router: RouterFn | None = None) -> None:
        self._nodes[name] = fn
        if router is not None:
            self._routers[name] = router

    def set_start(self, name: str) -> None:
        if name not in self._nodes:
            raise ValueError(f"Unknown start node: {name}")
        self._start = name

    def run(self, state: dict[str, Any], *, max_steps: int = DEFAULT_MAX_STEPS) -> dict[str, Any]:
        if self._start is None:
            raise ValueError("StateGraph start node is not set.")
        current: str | None = self._start
        steps = 0
        while current is not None:
            if steps >= max_steps:
                raise RuntimeError("StateGraph exceeded max steps; possible routing loop.")
            node = self._nodes.get(current)
            if node is None:
                raise ValueError(f"Unknown node during execution: {current}")
            node(state)
            router = self._routers.get(current)
            current = router(state) if router else None
            steps += 1
        return state
