"""Thin wrapper around the Ollama Python SDK.

Keeps the rest of the codebase decoupled from the Ollama API so we can
easily swap backends in the future.
"""

from __future__ import annotations

import time
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
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.options: dict[str, Any] = options or {}
        self.retries = max(retries, 0)
        self.timeout_seconds = timeout_seconds
        self._client = None
        self._cache: dict[tuple[str, str, str], str] = {}
        self._cache_order: list[tuple[str, str, str]] = []
        self._cache_limit = 128

    def _build_client(self):
        if self._client is not None:
            return self._client
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
    ) -> str:
        """Send a chat message and return the assistant's reply as a string."""
        client = self._build_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        merged_options = dict(self.options)
        if options:
            merged_options.update(options)

        model_candidates = [model or self.model]
        if fallback_models:
            model_candidates.extend([m for m in fallback_models if m])

        errors: list[str] = []
        attempts = self.retries + 1
        for candidate in model_candidates:
            cache_key = (
                candidate,
                system_prompt,
                user_message,
            )
            if cache_key in self._cache:
                return self._cache[cache_key]
            for attempt in range(1, attempts + 1):
                try:
                    response = client.chat(
                        model=candidate,
                        messages=messages,
                        options=merged_options or None,
                    )
                    message = response.get("message") if isinstance(response, dict) else None
                    if not isinstance(message, dict) or not isinstance(
                        message.get("content"), str
                    ):
                        raise RuntimeError(
                            "Ollama response missing expected message.content string."
                        )
                    content = message["content"]
                    self._cache[cache_key] = content
                    self._cache_order.append(cache_key)
                    if len(self._cache_order) > self._cache_limit:
                        oldest = self._cache_order.pop(0)
                        self._cache.pop(oldest, None)
                    return content
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{candidate} (attempt {attempt}/{attempts}): {exc}")
                    if attempt < attempts:
                        time.sleep(0.2)
                    continue

        error_text = "; ".join(errors) if errors else "unknown error"
        raise RuntimeError(f"Ollama chat failed across model candidates: {error_text}")
