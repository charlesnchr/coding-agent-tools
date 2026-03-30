"""ChromaDB-based vector store for session chunks."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb

from coding_agent_tools.chunker import Chunk

DEFAULT_STORE_PATH = str(Path.home() / ".cache" / "find-session" / "chroma")

COLLECTIONS = ("claude_sessions", "codex_sessions", "opencode_sessions")

AGENT_TO_COLLECTION = {
    "claude": "claude_sessions",
    "codex": "codex_sessions",
    "opencode": "opencode_sessions",
}


@dataclass
class SemanticResult:
    """A single semantic search result."""

    session_id: str
    agent: str
    score: float  # cosine similarity (higher = more similar)
    chunk_text: str
    metadata: dict


class SessionVectorStore:
    """Manages ChromaDB collections for session chunk embeddings."""

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = store_path or DEFAULT_STORE_PATH
        os.makedirs(self.store_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.store_path)
        self._tracker_path = os.path.join(self.store_path, "file_tracker.json")
        self._tracker: Optional[dict] = None

    def _load_tracker(self) -> dict:
        if self._tracker is None:
            if os.path.exists(self._tracker_path):
                with open(self._tracker_path) as f:
                    self._tracker = json.load(f)
            else:
                self._tracker = {}
        return self._tracker

    def _save_tracker(self) -> None:
        if self._tracker is not None:
            with open(self._tracker_path, "w") as f:
                json.dump(self._tracker, f)

    def track_file(self, agent: str, file_path: str, mtime: float, size: int) -> None:
        """Record that a file has been processed (with or without chunks)."""
        tracker = self._load_tracker()
        key = f"{agent}:{file_path}"
        tracker[key] = {"mtime": round(mtime, 2), "size": size}
        self._save_tracker()

    def get_tracked_files(self, agent: str) -> dict[str, tuple[float, int]]:
        """Get all tracked files for an agent as {file_path: (mtime, size)}."""
        tracker = self._load_tracker()
        prefix = f"{agent}:"
        result = {}
        for key, val in tracker.items():
            if key.startswith(prefix):
                fp = key[len(prefix):]
                result[fp] = (val["mtime"], val["size"])
        return result

    def untrack_file(self, agent: str, file_path: str) -> None:
        """Remove a file from the tracker."""
        tracker = self._load_tracker()
        key = f"{agent}:{file_path}"
        tracker.pop(key, None)
        self._save_tracker()

    def clear_tracker(self) -> None:
        """Clear all tracked files."""
        self._tracker = {}
        self._save_tracker()

    def get_collection(self, agent: str) -> chromadb.Collection:
        """Get or create a collection for an agent type."""
        name = AGENT_TO_COLLECTION.get(agent, f"{agent}_sessions")
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        agent: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks with pre-computed embeddings to the store."""
        if not chunks:
            return

        collection = self.get_collection(agent)

        # ChromaDB metadata values must be str, int, float, or bool
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [_sanitize_metadata(c.metadata) for c in chunks]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        agent: Optional[str] = None,
        n_results: int = 50,
    ) -> list[SemanticResult]:
        """Query the vector store with an embedding.

        Args:
            query_embedding: The query vector
            agent: Optional agent filter. If None, searches all collections.
            n_results: Max results per collection

        Returns:
            List of SemanticResult sorted by score descending
        """
        results: list[SemanticResult] = []

        if agent:
            agents = [agent]
        else:
            agents = list(AGENT_TO_COLLECTION.keys())

        for a in agents:
            collection = self.get_collection(a)
            if collection.count() == 0:
                continue

            query_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count()),
            )

            if not query_result["ids"] or not query_result["ids"][0]:
                continue

            ids = query_result["ids"][0]
            distances = query_result["distances"][0] if query_result.get("distances") else [0.0] * len(ids)
            documents = query_result["documents"][0] if query_result.get("documents") else [""] * len(ids)
            metadatas = query_result["metadatas"][0] if query_result.get("metadatas") else [{}] * len(ids)

            for doc_id, distance, doc, meta in zip(ids, distances, documents, metadatas):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity: 1 - distance/2 gives [0, 1]
                similarity = 1.0 - distance / 2.0
                results.append(SemanticResult(
                    session_id=meta.get("session_id", doc_id.split(":")[1] if ":" in doc_id else doc_id),
                    agent=meta.get("agent", a),
                    score=similarity,
                    chunk_text=doc[:300],
                    metadata=meta,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def get_indexed_files(self, agent: str) -> dict[str, tuple[float, int]]:
        """Get map of indexed file_path -> (mtime, size) for incremental updates."""
        collection = self.get_collection(agent)
        if collection.count() == 0:
            return {}

        # Get all metadata - fetch in batches
        indexed: dict[str, tuple[float, int]] = {}
        count = collection.count()
        batch_size = 5000

        for offset in range(0, count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )
            for meta in result["metadatas"]:
                fp = meta.get("file_path", "")
                if fp and fp not in indexed:
                    indexed[fp] = (
                        float(meta.get("file_mtime", 0)),
                        int(meta.get("file_size", 0)),
                    )

        return indexed

    def get_indexed_sessions(self, agent: str) -> dict[str, float]:
        """Get map of session_id -> mtime for OpenCode incremental updates."""
        collection = self.get_collection(agent)
        if collection.count() == 0:
            return {}

        indexed: dict[str, float] = {}
        count = collection.count()
        batch_size = 5000

        for offset in range(0, count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )
            for meta in result["metadatas"]:
                sid = meta.get("session_id", "")
                if sid and sid not in indexed:
                    indexed[sid] = float(meta.get("file_mtime", 0))

        return indexed

    def delete_by_file(self, agent: str, file_path: str) -> None:
        """Delete all chunks for a given file path."""
        collection = self.get_collection(agent)
        # Find IDs with this file_path
        results = collection.get(
            where={"file_path": file_path},
            include=[],
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])

    def delete_session(self, agent: str, session_id: str) -> None:
        """Delete all chunks for a given session."""
        collection = self.get_collection(agent)
        results = collection.get(
            where={"session_id": session_id},
            include=[],
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])

    def get_stats(self) -> dict[str, int]:
        """Get chunk counts per collection."""
        stats = {}
        for agent, coll_name in AGENT_TO_COLLECTION.items():
            try:
                collection = self._client.get_or_create_collection(name=coll_name)
                stats[agent] = collection.count()
            except Exception:
                stats[agent] = 0
        stats["total"] = sum(stats.values())
        return stats

    def clear(self) -> None:
        """Delete all collections."""
        for coll_name in COLLECTIONS:
            try:
                self._client.delete_collection(name=coll_name)
            except Exception:
                pass


def _sanitize_metadata(meta: dict) -> dict:
    """Ensure all metadata values are ChromaDB-compatible types."""
    sanitized = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        elif v is None:
            sanitized[k] = ""
        else:
            sanitized[k] = str(v)
    return sanitized
