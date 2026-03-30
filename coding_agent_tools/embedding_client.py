"""Ollama embedding client for semantic search."""

import json
import os
import subprocess
import sys
from typing import Optional

import httpx

DEFAULT_MODEL = "qwen3-embedding:8b"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaEmbedder:
    """Wrapper around Ollama's /api/embed endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=120.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of embedding vectors."""
        resp = self._client.post(
            "/api/embed",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return any(
                m.get("name", "").startswith(self.model.split(":")[0])
                for m in models
            )
        except (httpx.ConnectError, httpx.HTTPError):
            return False

    def pull_model(self, proxy: Optional[str] = None) -> None:
        """Pull the embedding model via Ollama CLI.

        Args:
            proxy: SOCKS5 proxy URL (e.g. socks5://localhost:1085)
        """
        env = os.environ.copy()
        if proxy:
            env["ALL_PROXY"] = proxy
            env["all_proxy"] = proxy

        print(f"Pulling {self.model} via Ollama...")
        if proxy:
            print(f"  Using proxy: {proxy}")

        result = subprocess.run(
            ["ollama", "pull", self.model],
            env=env,
        )
        if result.returncode != 0:
            print(f"Error pulling model (exit code {result.returncode})", file=sys.stderr)
            sys.exit(1)
        print(f"Model {self.model} pulled successfully.")

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
