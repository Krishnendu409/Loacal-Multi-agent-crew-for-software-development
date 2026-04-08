"""Thin wrapper around the Ollama Python SDK.

Keeps the rest of the codebase decoupled from the Ollama API so we can
easily swap backends in the future.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any


def _get_ollama():
    """Import ollama lazily so the test suite can mock it easily."""
    try:
        import ollama  # type: ignore[import-untyped]

        return ollama
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'ollama' package is required. Install it with:\n"
            "    pip install ollama\n"
            "and make sure the Ollama daemon is running (https://ollama.com)."
        ) from exc


class OllamaClient:
    """Simple wrapper that sends a single-turn chat message to an Ollama model."""

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        options: dict[str, Any] | None = None,
        retries: int = 1,
        timeout_seconds: int | None = None,
        max_cache_entries: int = 256,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.options: dict[str, Any] = options or {}
        self.retries = max(retries, 0)
        self.timeout_seconds = timeout_seconds
        self._client: Any = None
        self._client_lock = threading.Lock()
        # cache key: (model, system_prompt, user_message, options_signature)
        self._cache: OrderedDict[tuple[str, str, str, str], str] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_cache_entries = max(0, int(max_cache_entries))

    def _get_client(self) -> Any:
        """Return the shared Ollama client, creating it on first use (thread-safe)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    ollama = _get_ollama()
                    kwargs: dict[str, Any] = {"host": self.base_url}
                    if self.timeout_seconds:
                        kwargs["timeout"] = self.timeout_seconds
                    self._client = ollama.Client(**kwargs)
        return self._client

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        fallback_models: list[str] | None = None,
        retries_override: int | None = None,
    ) -> str:
        """Send a chat message and return the assistant's reply as a string."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        merged_options = dict(self.options)
        if options:
            merged_options.update(options)
        options_signature = self._options_signature(merged_options)

        model_candidates = [model or self.model]
        if fallback_models:
            model_candidates.extend([m for m in fallback_models if m])

        errors: list[str] = []
        effective_retries = self.retries if retries_override is None else max(retries_override, 0)
        attempts = effective_retries + 1
        for candidate in model_candidates:
            cache_key = (
                candidate,
                system_prompt,
                user_message,
                options_signature,
            )
            with self._cache_lock:
                if cache_key in self._cache:
                    self._cache.move_to_end(cache_key)
                    return self._cache[cache_key]
            for attempt in range(1, attempts + 1):
                try:
                    response = client.chat(
                        model=candidate,
                        messages=messages,
                        options=merged_options or None,
                    )
                    message = response.get("message", {}) if isinstance(response, dict) else {}
                    content = message.get("content") if isinstance(message, dict) else None
                    if isinstance(content, str):
                        with self._cache_lock:
                            self._cache[cache_key] = content
                            self._cache.move_to_end(cache_key)
                            while len(self._cache) > self._max_cache_entries:
                                self._cache.popitem(last=False)
                        return content
                    raise RuntimeError("Ollama response missing message.content")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{candidate} (attempt {attempt}/{attempts}): {exc}")
                    if attempt < attempts:
                        time.sleep(0.2)
                    continue

        error_text = "; ".join(errors) if errors else "unknown error"
        raise RuntimeError(f"Ollama chat failed across model candidates: {error_text}")

    @staticmethod
    def _options_signature(options: dict[str, Any]) -> str:
        if not options:
            return ""
        try:
            return json.dumps(options, sort_keys=True, default=str, separators=(",", ":"))
        except (TypeError, ValueError):
            return repr(options)
