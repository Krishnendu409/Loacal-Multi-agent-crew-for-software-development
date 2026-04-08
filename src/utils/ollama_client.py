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

_TIMEOUT_EXTENSION_SECONDS = 120


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
        # cache key: (model, system_prompt, user_message, options+format signature)
        self._cache: OrderedDict[tuple[str, str, str, str], str] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_cache_entries = max(0, int(max_cache_entries))

    def _get_client(self) -> Any:
        """Return the shared Ollama client, creating it on first use (thread-safe)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._build_client(self.timeout_seconds)
        return self._client

    def _build_client(self, timeout_seconds: int | None) -> Any:
        ollama = _get_ollama()
        kwargs: dict[str, Any] = {"host": self.base_url}
        if timeout_seconds:
            kwargs["timeout"] = timeout_seconds
        return ollama.Client(**kwargs)

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        timeout_types: tuple[type[BaseException], ...] = (TimeoutError,)
        try:
            import httpx

            timeout_types = (TimeoutError, httpx.TimeoutException)
        except ImportError:  # pragma: no cover
            pass
        if isinstance(exc, timeout_types):
            return True
        if exc.__cause__ is not None and isinstance(exc.__cause__, timeout_types):
            return True
        text = str(exc).lower()
        return "timed out" in text

    def _extract_content(self, response: Any) -> str:
        message = response.get("message", {}) if isinstance(response, dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        raise RuntimeError("Ollama response missing message.content")

    def _store_cache(self, cache_key: tuple[str, str, str, str], content: str) -> None:
        with self._cache_lock:
            self._cache[cache_key] = content
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        format_schema: dict[str, Any] | None = None,
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
        format_signature = self._options_signature(format_schema or {})

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
                f"{options_signature}|{format_signature}",
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
                        format=format_schema or None,
                    )
                    content = self._extract_content(response)
                    self._store_cache(cache_key, content)
                    return content
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{candidate} (attempt {attempt}/{attempts}): {exc}")
                    if self._is_timeout_error(exc):
                        timeout_base = (
                            self.timeout_seconds
                            if self.timeout_seconds is not None
                            else _TIMEOUT_EXTENSION_SECONDS
                        )
                        extended_timeout = timeout_base + _TIMEOUT_EXTENSION_SECONDS
                        try:
                            extended_client = self._build_client(extended_timeout)
                            response = extended_client.chat(
                                model=candidate,
                                messages=messages,
                                options=merged_options or None,
                                format=format_schema or None,
                            )
                            content = self._extract_content(response)
                            self._store_cache(cache_key, content)
                            return content
                        except Exception as timeout_retry_exc:  # noqa: BLE001
                            errors.append(
                                f"{candidate} (attempt {attempt}/{attempts}, "
                                f"extended-timeout={extended_timeout}s): {timeout_retry_exc}"
                            )
                    if attempt < attempts:
                        time.sleep(0.2)
                    continue

        error_text = "; ".join(errors) if errors else "unknown error"
        raise RuntimeError(f"Ollama chat failed across model candidates: {error_text}")

    def embed(self, text: str, *, model: str = "nomic-embed-text") -> list[float]:
        """Create an embedding vector for *text* using Ollama embeddings API."""
        if not text.strip():
            return []
        client = self._get_client()
        response = client.embeddings(model=model, prompt=text)
        embedding = response.get("embedding") if isinstance(response, dict) else None
        if isinstance(embedding, list) and all(isinstance(v, (float, int)) for v in embedding):
            return [float(v) for v in embedding]
        return []

    @staticmethod
    def _options_signature(options: dict[str, Any]) -> str:
        if not options:
            return ""
        try:
            return json.dumps(options, sort_keys=True, default=str, separators=(",", ":"))
        except (TypeError, ValueError):
            return repr(options)
