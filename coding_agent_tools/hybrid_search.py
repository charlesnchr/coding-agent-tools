"""Hybrid search combining keyword and semantic results via Reciprocal Rank Fusion."""

from typing import Optional


def hybrid_search(
    keywords: list[str],
    keyword_results: list[dict],
    global_search: bool = False,
    agents: Optional[list[str]] = None,
    num_results: int = 50,
) -> list[dict]:
    """Run semantic search and fuse with keyword results using RRF.

    Args:
        keywords: Search keywords (used as semantic query text)
        keyword_results: Results from existing keyword search
        global_search: Whether searching globally
        agents: Agent filter
        num_results: Max results to return

    Returns:
        Merged list of session dicts with hybrid_score added
    """
    from coding_agent_tools.embedding_client import OllamaEmbedder
    from coding_agent_tools.vector_store import SessionVectorStore, SemanticResult

    embedder = OllamaEmbedder()
    if not embedder.is_available():
        # Fall back to keyword-only
        return keyword_results

    store = SessionVectorStore()
    stats = store.get_stats()
    if stats.get("total", 0) == 0:
        # No index built yet
        return keyword_results

    # Embed the query
    query_text = " ".join(keywords)
    query_embedding = embedder.embed_single(query_text)
    embedder.close()

    # Query vector store
    agent_filter = agents[0] if agents and len(agents) == 1 else None
    semantic_results = store.query(
        query_embedding=query_embedding,
        agent=agent_filter,
        n_results=num_results,
    )

    if not semantic_results:
        return keyword_results

    # Aggregate semantic results to session level (best chunk per session)
    semantic_by_session = _aggregate_to_sessions(semantic_results)

    # Fuse using RRF
    fused = reciprocal_rank_fusion(keyword_results, semantic_by_session)

    return fused[:num_results]


def _aggregate_to_sessions(
    results: list,
) -> dict[str, dict]:
    """Group semantic results by (agent, session_id), keep best score per session.

    Returns: dict keyed by "{agent}:{session_id}" -> {score, chunk_text, metadata}
    """
    best: dict[str, dict] = {}
    for r in results:
        key = f"{r.agent}:{r.session_id}"
        if key not in best or r.score > best[key]["score"]:
            best[key] = {
                "session_id": r.session_id,
                "agent": r.agent,
                "score": r.score,
                "chunk_text": r.chunk_text,
                "metadata": r.metadata,
            }
    return best


def reciprocal_rank_fusion(
    keyword_results: list[dict],
    semantic_by_session: dict[str, dict],
    k: int = 60,
) -> list[dict]:
    """Merge keyword and semantic results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) across all rank lists.
    k=60 is the standard constant from the original RRF paper.

    Returns: merged list of session dicts sorted by RRF score descending.
    """
    # Build a unified map of all sessions
    all_sessions: dict[str, dict] = {}

    # Add keyword results
    for rank, session in enumerate(keyword_results):
        key = f"{session['agent']}:{session['session_id']}"
        all_sessions[key] = dict(session)  # copy
        all_sessions[key]["_keyword_rank"] = rank
        all_sessions[key]["_semantic_rank"] = None

    # Add/merge semantic results
    semantic_ranked = sorted(
        semantic_by_session.values(),
        key=lambda x: x["score"],
        reverse=True,
    )
    for rank, sem in enumerate(semantic_ranked):
        key = f"{sem['agent']}:{sem['session_id']}"
        if key in all_sessions:
            # Already have from keyword search - merge semantic info
            all_sessions[key]["_semantic_rank"] = rank
            # Use semantic best_chunk if it's better context
            if sem.get("chunk_text"):
                all_sessions[key]["semantic_chunk"] = sem["chunk_text"]
        else:
            # Semantic-only result - create session dict from metadata
            meta = sem.get("metadata", {})
            all_sessions[key] = {
                "agent": sem["agent"],
                "agent_display": sem["agent"].title(),
                "session_id": sem["session_id"],
                "mod_time": float(meta.get("file_mtime", 0)),
                "create_time": float(meta.get("file_mtime", 0)),
                "lines": 0,
                "project": meta.get("project", ""),
                "first_message": "",
                "last_message": "",
                "match_score": 0.0,
                "best_chunk": sem.get("chunk_text", ""),
                "cwd": meta.get("cwd", ""),
                "branch": meta.get("branch", ""),
                "_keyword_rank": None,
                "_semantic_rank": rank,
                "semantic_chunk": sem.get("chunk_text", ""),
            }

    # Calculate RRF scores
    for key, session in all_sessions.items():
        rrf_score = 0.0
        kr = session.get("_keyword_rank")
        sr = session.get("_semantic_rank")

        if kr is not None:
            rrf_score += 1.0 / (k + kr)
        if sr is not None:
            rrf_score += 1.0 / (k + sr)

        session["hybrid_score"] = rrf_score
        session["match_score"] = rrf_score * 100  # scale for display

        # Set search source indicator
        if kr is not None and sr is not None:
            session["search_source"] = "K+S"
        elif kr is not None:
            session["search_source"] = "K"
        else:
            session["search_source"] = "S"

        # Use semantic chunk as best_chunk when available and keyword didn't have one
        if not session.get("best_chunk") and session.get("semantic_chunk"):
            session["best_chunk"] = session["semantic_chunk"]

        # Clean up internal keys
        session.pop("_keyword_rank", None)
        session.pop("_semantic_rank", None)
        session.pop("semantic_chunk", None)

    # Sort by RRF score
    result = sorted(all_sessions.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return result
