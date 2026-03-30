"""CLI for building and managing the semantic search index."""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

DEFAULT_PROXY = "socks5://localhost:1085"
EMBED_BATCH_SIZE = 32


def _check_deps():
    """Check that semantic dependencies are installed."""
    missing = []
    try:
        import chromadb  # noqa: F401
    except ImportError:
        missing.append("chromadb")
    try:
        import httpx  # noqa: F401
    except ImportError:
        missing.append("httpx")

    if missing:
        print(
            f"Missing dependencies: {', '.join(missing)}\n"
            f"Install with: uv tool install 'coding-agent-tools[semantic]'",
            file=sys.stderr,
        )
        sys.exit(1)


def _iter_claude_files(
    claude_home: Optional[str] = None,
) -> list[tuple[Path, str, str]]:
    """Yield (filepath, project_name, project_path) for all Claude session files."""
    base_dir = Path(claude_home).expanduser() if claude_home else Path.home() / ".claude"
    projects_dir = base_dir / "projects"
    if not projects_dir.exists():
        return []

    results = []
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        dir_name = project_dir.name
        # Reconstruct original path from encoded directory name
        if dir_name.startswith("-"):
            original_path = "/" + dir_name[1:].replace("-", "/")
        else:
            original_path = dir_name
        project_name = original_path.rstrip("/").split("/")[-1] if "/" in original_path else dir_name

        for jsonl_file in project_dir.glob("*.jsonl"):
            results.append((jsonl_file, project_name, original_path))
    return results


def _iter_codex_files(
    codex_home: Optional[str] = None,
) -> list[tuple[Path, str, str, str]]:
    """Yield (filepath, session_id, project_name, cwd, branch) for all Codex session files."""
    import re
    base = Path(codex_home).expanduser() if codex_home else Path.home() / ".codex"
    sessions_dir = base / "sessions"
    if not sessions_dir.exists():
        return []

    results = []
    for year_dir in sessions_dir.iterdir():
        if not year_dir.is_dir():
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
            for day_dir in month_dir.iterdir():
                if not day_dir.is_dir():
                    continue
                for f in day_dir.glob("rollout-*.jsonl"):
                    # Extract session ID from filename
                    match = re.match(
                        r"rollout-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-(.+)\.jsonl",
                        f.name,
                    )
                    session_id = match.group(1) if match else f.stem

                    # Extract metadata from file
                    cwd = ""
                    branch = ""
                    try:
                        with open(f, "r", encoding="utf-8") as fh:
                            for line in fh:
                                if not line.strip():
                                    continue
                                try:
                                    entry = json.loads(line)
                                    if entry.get("type") == "session_meta":
                                        payload = entry.get("payload", {})
                                        cwd = payload.get("cwd", "")
                                        branch = payload.get("git", {}).get("branch", "")
                                        break
                                except json.JSONDecodeError:
                                    continue
                    except (OSError, IOError):
                        pass

                    project_name = Path(cwd).name if cwd else "unknown"
                    results.append((f, session_id, project_name, cwd, branch))
    return results


def _iter_opencode_sessions(
    opencode_home: Optional[str] = None,
) -> list[tuple[str, str, str, float]]:
    """Yield (session_id, project_name, cwd, time_updated) for all OpenCode sessions."""
    base = Path(opencode_home).expanduser() if opencode_home else Path.home() / ".local" / "share" / "opencode"
    db_path = base / "opencode.db"
    if not db_path.exists():
        return []

    results = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT s.id, s.directory, s.time_updated, p.worktree, p.name
            FROM session s
            JOIN project p ON s.project_id = p.id
            """
        ).fetchall()
        for row in rows:
            directory = row["directory"] or ""
            worktree = row["worktree"] or ""
            cwd = directory if directory and directory != "/" else worktree
            project = Path(cwd).name if cwd else row["name"] or "unknown"
            time_updated = row["time_updated"] / 1000.0
            results.append((row["id"], project, cwd, time_updated))
        conn.close()
    except sqlite3.Error:
        pass
    return results


def build_index(
    agents: Optional[list[str]] = None,
    full: bool = False,
    claude_home: Optional[str] = None,
    codex_home: Optional[str] = None,
    opencode_home: Optional[str] = None,
):
    """Build or incrementally update the semantic index."""
    from coding_agent_tools.embedding_client import OllamaEmbedder
    from coding_agent_tools.vector_store import SessionVectorStore
    from coding_agent_tools.chunker import (
        chunk_claude_session,
        chunk_codex_session,
        chunk_opencode_session,
    )

    embedder = OllamaEmbedder()
    if not embedder.is_available():
        print(
            "Error: Ollama is not running or the embedding model is not available.\n"
            "Run: find-session-index --pull-model",
            file=sys.stderr,
        )
        sys.exit(1)

    store = SessionVectorStore()
    console = Console() if RICH_AVAILABLE else None

    if full:
        if console:
            console.print("[yellow]Full re-index requested. Clearing existing index...[/yellow]")
        store.clear()

    target_agents = agents or ["claude", "codex", "opencode"]

    # -- Claude --
    if "claude" in target_agents:
        _index_claude(embedder, store, claude_home, console, full)

    # -- Codex --
    if "codex" in target_agents:
        _index_codex(embedder, store, codex_home, console, full)

    # -- OpenCode --
    if "opencode" in target_agents:
        _index_opencode(embedder, store, opencode_home, console, full)

    # Print stats
    stats = store.get_stats()
    if console:
        console.print(f"\n[green]Index complete.[/green] Total chunks: {stats['total']}")
        for agent, count in stats.items():
            if agent != "total":
                console.print(f"  {agent}: {count} chunks")
    else:
        print(f"\nIndex complete. Total chunks: {stats['total']}")

    embedder.close()


def _index_claude(embedder, store, claude_home, console, full):
    from coding_agent_tools.chunker import chunk_claude_session

    files = _iter_claude_files(claude_home)
    if not files:
        if console:
            console.print("[dim]No Claude sessions found.[/dim]")
        return

    indexed = {} if full else store.get_indexed_files("claude")
    to_process = []
    current_paths = set()

    for filepath, project_name, project_path in files:
        fp = str(filepath)
        current_paths.add(fp)
        stat = filepath.stat()
        prev = indexed.get(fp)
        if prev and prev[0] == stat.st_mtime and prev[1] == stat.st_size:
            continue  # unchanged
        to_process.append((filepath, project_name, project_path))

    # Delete removed files
    for fp in set(indexed.keys()) - current_paths:
        store.delete_by_file("claude", fp)

    if not to_process:
        if console:
            console.print(f"[dim]Claude: {len(files)} files, all up to date.[/dim]")
        return

    if console:
        console.print(f"[cyan]Claude: indexing {len(to_process)}/{len(files)} files...[/cyan]")

    _process_file_batch(
        to_process,
        lambda fp, proj, path: chunk_claude_session(fp, fp.stem, proj, path, ""),
        "claude",
        embedder,
        store,
        console,
    )


def _index_codex(embedder, store, codex_home, console, full):
    from coding_agent_tools.chunker import chunk_codex_session

    files = _iter_codex_files(codex_home)
    if not files:
        if console:
            console.print("[dim]No Codex sessions found.[/dim]")
        return

    indexed = {} if full else store.get_indexed_files("codex")
    to_process = []
    current_paths = set()

    for filepath, session_id, project_name, cwd, branch in files:
        fp = str(filepath)
        current_paths.add(fp)
        stat = filepath.stat()
        prev = indexed.get(fp)
        if prev and prev[0] == stat.st_mtime and prev[1] == stat.st_size:
            continue
        to_process.append((filepath, session_id, project_name, cwd, branch))

    for fp in set(indexed.keys()) - current_paths:
        store.delete_by_file("codex", fp)

    if not to_process:
        if console:
            console.print(f"[dim]Codex: {len(files)} files, all up to date.[/dim]")
        return

    if console:
        console.print(f"[cyan]Codex: indexing {len(to_process)}/{len(files)} files...[/cyan]")

    _process_codex_batch(to_process, embedder, store, console)


def _index_opencode(embedder, store, opencode_home, console, full):
    from coding_agent_tools.chunker import chunk_opencode_session

    sessions = _iter_opencode_sessions(opencode_home)
    if not sessions:
        if console:
            console.print("[dim]No OpenCode sessions found.[/dim]")
        return

    indexed = {} if full else store.get_indexed_sessions("opencode")
    to_process = []

    for session_id, project, cwd, time_updated in sessions:
        prev_mtime = indexed.get(session_id)
        if prev_mtime and prev_mtime == time_updated:
            continue
        to_process.append((session_id, project, cwd, time_updated))

    if not to_process:
        if console:
            console.print(f"[dim]OpenCode: {len(sessions)} sessions, all up to date.[/dim]")
        return

    if console:
        console.print(f"[cyan]OpenCode: indexing {len(to_process)}/{len(sessions)} sessions...[/cyan]")

    # Open DB for chunking
    base = Path(opencode_home).expanduser() if opencode_home else Path.home() / ".local" / "share" / "opencode"
    db_path = base / "opencode.db"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    all_chunks = []
    for session_id, project, cwd, time_updated in to_process:
        # Delete old chunks for this session
        store.delete_session("opencode", session_id)
        chunks = chunk_opencode_session(conn, session_id, project, cwd, time_updated)
        all_chunks.extend(chunks)

    conn.close()

    if all_chunks:
        _embed_and_store(all_chunks, "opencode", embedder, store, console)


def _process_file_batch(file_list, chunker_fn, agent, embedder, store, console):
    """Process a batch of files: chunk, embed, store."""
    all_chunks = []

    if RICH_AVAILABLE and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Chunking...", total=len(file_list))
            for item in file_list:
                filepath = item[0]
                # Delete old chunks
                store.delete_by_file(agent, str(filepath))
                chunks = chunker_fn(*item)
                all_chunks.extend(chunks)
                progress.advance(task)
    else:
        for item in file_list:
            filepath = item[0]
            store.delete_by_file(agent, str(filepath))
            chunks = chunker_fn(*item)
            all_chunks.extend(chunks)

    if all_chunks:
        _embed_and_store(all_chunks, agent, embedder, store, console)


def _process_codex_batch(file_list, embedder, store, console):
    """Process Codex files with their specific signature."""
    from coding_agent_tools.chunker import chunk_codex_session

    all_chunks = []

    if RICH_AVAILABLE and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Chunking...", total=len(file_list))
            for filepath, session_id, project_name, cwd, branch in file_list:
                store.delete_by_file("codex", str(filepath))
                chunks = chunk_codex_session(filepath, session_id, project_name, cwd, branch)
                all_chunks.extend(chunks)
                progress.advance(task)
    else:
        for filepath, session_id, project_name, cwd, branch in file_list:
            store.delete_by_file("codex", str(filepath))
            chunks = chunk_codex_session(filepath, session_id, project_name, cwd, branch)
            all_chunks.extend(chunks)

    if all_chunks:
        _embed_and_store(all_chunks, "codex", embedder, store, console)


def _embed_and_store(chunks, agent, embedder, store, console):
    """Embed chunks in batches and store them."""
    total = len(chunks)

    if RICH_AVAILABLE and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Embedding {total} chunks...", total=total)

            for i in range(0, total, EMBED_BATCH_SIZE):
                batch = chunks[i : i + EMBED_BATCH_SIZE]
                texts = [c.text for c in batch]
                embeddings = embedder.embed(texts)
                store.add_chunks(agent, batch, embeddings)
                progress.advance(task, advance=len(batch))
    else:
        for i in range(0, total, EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c.text for c in batch]
            embeddings = embedder.embed(texts)
            store.add_chunks(agent, batch, embeddings)
            done = min(i + EMBED_BATCH_SIZE, total)
            print(f"  Embedded {done}/{total} chunks", end="\r")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Build and manage the semantic search index for find-session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    find-session-index                          # Incremental index update
    find-session-index --full                   # Full re-index
    find-session-index --pull-model             # Pull embedding model (via proxy)
    find-session-index --stats                  # Show index statistics
    find-session-index --clear                  # Delete the index
    find-session-index --agents claude          # Index only Claude sessions
        """,
    )

    parser.add_argument(
        "--pull-model",
        action="store_true",
        help="Pull the embedding model via Ollama",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=DEFAULT_PROXY,
        help=f"Proxy for model pull (default: {DEFAULT_PROXY})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full re-index (delete and rebuild)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the entire index",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["claude", "codex", "opencode"],
        help="Limit indexing to specific agents",
    )
    parser.add_argument("--claude-home", type=str)
    parser.add_argument("--codex-home", type=str)
    parser.add_argument("--opencode-home", type=str)

    args = parser.parse_args()

    _check_deps()

    if args.pull_model:
        from coding_agent_tools.embedding_client import OllamaEmbedder
        embedder = OllamaEmbedder()
        embedder.pull_model(proxy=args.proxy)
        return

    if args.stats:
        from coding_agent_tools.vector_store import SessionVectorStore
        store = SessionVectorStore()
        stats = store.get_stats()
        console = Console() if RICH_AVAILABLE else None
        if console:
            console.print("[bold]Index Statistics[/bold]")
            for agent, count in stats.items():
                if agent != "total":
                    console.print(f"  {agent}: {count} chunks")
            console.print(f"  [bold]Total: {stats['total']} chunks[/bold]")
        else:
            print("Index Statistics")
            for agent, count in stats.items():
                print(f"  {agent}: {count}")
        return

    if args.clear:
        from coding_agent_tools.vector_store import SessionVectorStore
        store = SessionVectorStore()
        store.clear()
        console = Console() if RICH_AVAILABLE else None
        if console:
            console.print("[green]Index cleared.[/green]")
        else:
            print("Index cleared.")
        return

    build_index(
        agents=args.agents,
        full=args.full,
        claude_home=args.claude_home,
        codex_home=args.codex_home,
        opencode_home=args.opencode_home,
    )


if __name__ == "__main__":
    main()
