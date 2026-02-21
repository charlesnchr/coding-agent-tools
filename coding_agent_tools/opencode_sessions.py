#!/usr/bin/env python3
"""
Find and resume OpenCode sessions by searching keywords in session history.

Usage:
    find-opencode-session "keywords" [OPTIONS]

Examples:
    find-opencode-session "langroid,MCP"           # Current project only
    find-opencode-session "error,debugging" -g     # All projects
    find-opencode-session "keywords" -n 5          # Limit results
"""

import argparse
import json
import os
import shlex
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def get_opencode_home(custom_home: Optional[str] = None) -> Path:
    """Get the OpenCode data directory."""
    if custom_home:
        return Path(custom_home).expanduser()
    return Path.home() / ".local" / "share" / "opencode"


def get_db_path(opencode_home: Path) -> Path:
    """Get the path to the OpenCode SQLite database."""
    return opencode_home / "opencode.db"


def get_project_name(worktree: str, directory: str) -> str:
    """Extract project name from worktree or directory path."""
    # Prefer worktree if it's a real project path (not just /)
    path_str = worktree if worktree and worktree != "/" else directory
    if not path_str:
        return "unknown"
    path = Path(path_str)
    return path.name if path.name else "unknown"


def get_session_cwd(worktree: str, directory: str) -> str:
    """Determine the effective working directory for a session."""
    # Use directory (actual cwd) if available, fall back to worktree
    if directory and directory != "/":
        return directory
    if worktree and worktree != "/":
        return worktree
    return directory or worktree or ""


def format_preview(first_message: Optional[str], last_message: Optional[str]) -> Optional[str]:
    """Format first/last user messages into a compact preview string."""
    if first_message and last_message:
        if first_message == last_message:
            return f"First/Last: {first_message}"
        return f"First: {first_message}\nLast: {last_message}"
    if first_message:
        return f"First/Last: {first_message}"
    if last_message:
        return f"First/Last: {last_message}"
    return None


def find_sessions(
    opencode_home: Path,
    keywords: list[str],
    num_matches: int = 10,
    global_search: bool = False,
) -> list[dict]:
    """
    Find OpenCode sessions matching keywords.

    Args:
        opencode_home: Path to OpenCode data directory
        keywords: List of keywords to search for (AND logic)
        num_matches: Maximum number of results to return
        global_search: If False, filter to current directory only

    Returns list of dicts with: session_id, project, date, mod_time,
                                 lines, preview, cwd, file_path
    """
    db_path = get_db_path(opencode_home)
    if not db_path.exists():
        return []

    current_cwd = os.getcwd() if not global_search else None

    matches = []

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        # Get all sessions with their project info
        sessions = conn.execute(
            """
            SELECT
                s.id, s.title, s.directory, s.time_created, s.time_updated,
                p.worktree, p.name as project_name,
                (SELECT COUNT(*) FROM message m WHERE m.session_id = s.id) as msg_count
            FROM session s
            JOIN project p ON s.project_id = p.id
            ORDER BY s.time_updated DESC
            """
        ).fetchall()

        for session in sessions:
            session_id = session["id"]
            directory = session["directory"] or ""
            worktree = session["worktree"] or ""
            cwd = get_session_cwd(worktree, directory)

            # Filter by current directory if not global search
            if current_cwd:
                # Match if the cwd starts with current_cwd or vice versa
                if not (
                    cwd == current_cwd
                    or cwd.startswith(current_cwd + "/")
                    or current_cwd.startswith(cwd + "/")
                    or worktree == current_cwd
                    or (worktree and current_cwd.startswith(worktree + "/"))
                ):
                    continue

            # Search keywords in message parts
            found, preview = _search_session_keywords(conn, session_id, keywords)
            if not found:
                continue

            # OpenCode timestamps are in milliseconds
            mod_time = session["time_updated"] / 1000.0
            create_time = session["time_created"] / 1000.0

            project = get_project_name(worktree, directory)
            title = session["title"] or ""

            # Use title as preview if we didn't find a good user message
            if not preview and title and title != f"New session - {session_id}":
                preview = title

            matches.append(
                {
                    "session_id": session_id,
                    "project": project,
                    "branch": "",  # OpenCode doesn't store branch in session data
                    "date": _format_date_range(create_time, mod_time),
                    "mod_time": mod_time,
                    "create_time": create_time,
                    "lines": session["msg_count"],
                    "preview": preview or title or "No preview",
                    "cwd": cwd,
                    "title": title,
                }
            )

            # Early exit if we have enough
            if len(matches) >= num_matches * 3:
                break

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        return []

    # Sort by modification time (newest first) and limit
    matches.sort(key=lambda x: x["mod_time"], reverse=True)
    return matches[:num_matches]


def _search_session_keywords(
    conn: sqlite3.Connection, session_id: str, keywords: list[str]
) -> tuple[bool, Optional[str]]:
    """
    Search for keywords in session message parts.

    Returns: (found, preview)
    - found: True if all keywords found (or True if no keywords)
    - preview: first and last user message text
    """
    first_any = None
    first_substantial = None
    last_any = None
    last_substantial = None

    # Get all text parts for this session, joined with message role info
    rows = conn.execute(
        """
        SELECT p.data as part_data, m.data as msg_data
        FROM part p
        JOIN message m ON p.message_id = m.id
        WHERE p.session_id = ?
        ORDER BY p.time_created ASC
        """,
        (session_id,),
    ).fetchall()

    keywords_lower = [k.lower() for k in keywords]
    found_keywords = set()

    for row in rows:
        try:
            part_data_str = row["part_data"]
            msg_data_str = row["msg_data"]

            # Extract user messages for first+last preview
            msg_data = json.loads(msg_data_str)
            if msg_data.get("role") == "user":
                part_data = json.loads(part_data_str)
                if part_data.get("type") == "text":
                    text = part_data.get("text", "").strip()
                    if text and not _is_system_message(text):
                        cleaned = text[:400].replace("\n", " ").strip()
                        if first_any is None:
                            first_any = cleaned
                        last_any = cleaned

                        if len(cleaned) > 20:
                            if first_substantial is None:
                                first_substantial = cleaned
                            last_substantial = cleaned

            # Search for keywords in all text content (both part and message data)
            combined_lower = (part_data_str + msg_data_str).lower()
            if keywords_lower:
                for kw in keywords_lower:
                    if kw in combined_lower:
                        found_keywords.add(kw)

        except (json.JSONDecodeError, KeyError):
            continue

    all_found = True if not keywords_lower else len(found_keywords) == len(keywords_lower)
    if not all_found:
        return False, None

    first_message = first_substantial or first_any
    last_message = last_substantial or last_any
    return True, format_preview(first_message, last_message)


def _is_system_message(text: str) -> bool:
    """Check if text is system-generated."""
    if not text or len(text.strip()) < 5:
        return True
    text = text.strip()
    if text.startswith("<") and ">" in text[:100]:
        return True
    return False


def _format_date_range(create_time: float, mod_time: float) -> str:
    """Format a date range string."""
    create_date = datetime.fromtimestamp(create_time).strftime("%m/%d")
    mod_date = datetime.fromtimestamp(mod_time).strftime("%m/%d %H:%M")
    return f"{create_date} - {mod_date}"


def display_interactive_ui(
    matches: list[dict],
    keywords: list[str] = None,
) -> Optional[dict]:
    """Display matches in interactive UI and get user selection."""
    if not matches:
        print("No matching sessions found.")
        return None

    if RICH_AVAILABLE:
        console = Console()
        title = (
            f"OpenCode Sessions matching: {', '.join(keywords)}"
            if keywords
            else "All OpenCode Sessions"
        )
        table = Table(title=title, show_header=True)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Session ID", style="yellow", no_wrap=True)
        table.add_column("Project", style="green")
        table.add_column("Date-Range", style="blue")
        table.add_column("Msgs", justify="right")
        table.add_column("First + Last User Message", style="dim", max_width=60, overflow="fold")

        for i, match in enumerate(matches, 1):
            table.add_row(
                str(i),
                match["session_id"][:20] + "...",
                match["project"],
                match["date"],
                str(match["lines"]),
                match["preview"],
            )

        console.print(table)
    else:
        print("\nMatching OpenCode Sessions:")
        print("-" * 80)
        for i, match in enumerate(matches, 1):
            print(f"{i}. {match['session_id'][:20]}...")
            print(f"   Project: {match['project']}")
            print(f"   Date: {match['date']}")
            print(f"   Preview: {match['preview'][:60]}...")
            print()

    # Get user selection
    if len(matches) == 1:
        print(f"\nAuto-selecting only match: {matches[0]['session_id'][:20]}...")
        return matches[0]

    try:
        choice = input(
            "\nEnter number to select session (or Enter to cancel): "
        ).strip()
        if not choice:
            print("Cancelled.")
            return None

        idx = int(choice) - 1
        if 0 <= idx < len(matches):
            return matches[idx]
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        print("Invalid input.")
        return None
    except KeyboardInterrupt:
        print("\nCancelled.")
        return None


def show_action_menu(match: dict) -> Optional[str]:
    """Show action menu for selected session."""
    print(f"\n=== Session: {match['session_id'][:20]}... ===")
    print(f"Project: {match['project']}")
    if match.get("title"):
        print(f"Title: {match['title']}")
    print(f"\nWhat would you like to do?")
    print("1. Resume session (default)")
    print("2. Show session ID")
    print("3. Export session (opencode export)")
    print()

    try:
        choice = input("Enter choice [1-3] (or Enter for 1): ").strip()
        if not choice or choice == "1":
            return "resume"
        elif choice == "2":
            return "path"
        elif choice == "3":
            return "export"
        else:
            print("Invalid choice.")
            return None
    except KeyboardInterrupt:
        print("\nCancelled.")
        return None


def copy_session_file(session_id: str) -> None:
    """Export session using opencode export."""
    import subprocess

    try:
        dest = input("\nEnter destination file path (or Enter for stdout): ").strip()
        if dest:
            result = subprocess.run(
                ["opencode", "export", session_id],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                dest_path = Path(dest).expanduser()
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_text(result.stdout)
                print(f"\nExported to: {dest_path}")
            else:
                print(f"\nError exporting: {result.stderr}")
        else:
            os.execlp("opencode", "opencode", "export", session_id)
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"\nError: {e}")


def resume_session(
    session_id: str, cwd: str, shell_mode: bool = False
) -> None:
    """Resume an OpenCode session."""
    if shell_mode:
        if cwd and cwd != os.getcwd():
            print(f"cd {shlex.quote(cwd)}", file=sys.stdout)
        print(f"opencode --session {shlex.quote(session_id)}", file=sys.stdout)
    else:
        if cwd and cwd != os.getcwd():
            response = input(
                f"\nSession is in different directory: {cwd}\n"
                "Change directory and resume? [Y/n]: "
            ).strip()
            if response.lower() in ("", "y", "yes"):
                try:
                    os.chdir(cwd)
                    print(f"Changed to: {cwd}")
                except OSError as e:
                    print(f"Error changing directory: {e}")
                    return

        try:
            os.execvp(
                "opencode", ["opencode", "--session", session_id]
            )
        except OSError as e:
            print(f"Error launching opencode: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find and resume OpenCode sessions by keyword search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  find-opencode-session "langroid,MCP"           # Current project only
  find-opencode-session "error,debugging" -g     # All projects
  find-opencode-session "keywords" -n 5          # Limit results
        """,
    )

    parser.add_argument(
        "keywords",
        nargs="?",
        default="",
        help="Comma-separated keywords to search (AND logic). If omitted, shows all sessions.",
    )
    parser.add_argument(
        "-g",
        "--global",
        dest="global_search",
        action="store_true",
        help="Search all projects (default: current project only)",
    )
    parser.add_argument(
        "-n",
        "--num-matches",
        type=int,
        default=10,
        help="Number of matches to display (default: 10)",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Output shell commands for eval (enables persistent cd)",
    )
    parser.add_argument(
        "--opencode-home",
        help="Custom OpenCode data directory (default: ~/.local/share/opencode)",
    )

    args = parser.parse_args()

    # Parse keywords
    keywords = (
        [k.strip() for k in args.keywords.split(",") if k.strip()]
        if args.keywords
        else []
    )

    # Get OpenCode home
    opencode_home = get_opencode_home(args.opencode_home)
    db_path = get_db_path(opencode_home)
    if not db_path.exists():
        print(f"Error: OpenCode database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Find matching sessions
    matches = find_sessions(
        opencode_home, keywords, args.num_matches, args.global_search
    )

    # Display and get selection
    selected_match = display_interactive_ui(matches, keywords)
    if not selected_match:
        return

    # Show action menu
    action = show_action_menu(selected_match)
    if not action:
        return

    # Perform selected action
    if action == "resume":
        resume_session(
            selected_match["session_id"], selected_match["cwd"], args.shell
        )
    elif action == "path":
        print(f"\nSession ID: {selected_match['session_id']}")
    elif action == "export":
        copy_session_file(selected_match["session_id"])


if __name__ == "__main__":
    main()
