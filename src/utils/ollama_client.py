"""Thin wrapper around the Ollama Python SDK.

Keeps the rest of the codebase decoupled from the Ollama API so we can
easily swap backends in the future.
"""

from __future__ import annotations

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
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.options: dict[str, Any] = options or {}

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a chat message and return the assistant's reply as a string."""
        ollama = _get_ollama()

        # Build client pointing at the configured base URL
        client = ollama.Client(host=self.base_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        response = client.chat(
            model=self.model,
            messages=messages,
            options=self.options or None,
        )
        return response["message"]["content"]
