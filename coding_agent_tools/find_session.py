#!/usr/bin/env python3
"""
Unified session finder - search across multiple coding agents (Claude Code, Codex, OpenCode)

Usage:
    find-session [keywords] [OPTIONS]
    fs [keywords] [OPTIONS]  # via shell wrapper

Examples:
    find-session "langroid,MCP"      # Search all agents in current project
    find-session -g                  # Show all sessions across all projects
    find-session "bug" --agents claude  # Search only Claude sessions
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, cast

# Import search functions from agent modules
from coding_agent_tools.claude_sessions import (
    find_sessions as find_claude_sessions,
    resume_session as resume_claude_session,
    get_session_file_path as get_claude_session_file_path,
    copy_session_file as copy_claude_session_file,
)
from coding_agent_tools.codex_sessions import (
    find_sessions as find_codex_sessions,
    resume_session as resume_codex_session,
    get_codex_home,
    copy_session_file as copy_codex_session_file,
)
from coding_agent_tools.opencode_sessions import (
    find_sessions as find_opencode_sessions,
    resume_session as resume_opencode_session,
    get_opencode_home,
    copy_session_file as copy_opencode_session_file,
)

# Textual TUI imports (optional - falls back to plain text if not installed)
try:
    from textual.app import App, ComposeResult
    from textual.containers import VerticalScroll
    from textual.widgets import Footer, Header, ListItem, ListView, Static
    from textual.binding import Binding
    from textual.reactive import reactive

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    App = object  # type: ignore[misc,assignment]
    ComposeResult = None  # type: ignore[misc,assignment]
    ListItem = object  # type: ignore[misc,assignment]
    ListView = object  # type: ignore[misc,assignment]
    Static = object  # type: ignore[misc,assignment]
    Header = object  # type: ignore[misc,assignment]
    Footer = object  # type: ignore[misc,assignment]
    VerticalScroll = object  # type: ignore[misc,assignment]
    Binding = object  # type: ignore[misc,assignment]
    reactive = lambda x: x  # type: ignore[misc,assignment]

# Rich imports (optional - used for fallback messages)
try:
    from rich.console import Console

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = cast(Any, None)  # type: ignore[misc]


@dataclass
class AgentConfig:
    """Configuration for a coding agent."""

    name: str
    display_name: str
    home_dir: Optional[str] = None
    enabled: bool = True


def get_default_agents() -> List[AgentConfig]:
    """Return default agent configurations."""
    return [
        AgentConfig(name="claude", display_name="Claude", home_dir=None),
        AgentConfig(name="codex", display_name="Codex", home_dir=None),
        AgentConfig(name="opencode", display_name="OC", home_dir=None),
    ]


def load_config() -> List[AgentConfig]:
    """Load agent configuration from config file or use defaults."""
    config_path = Path.home() / ".config" / "find-session" / "config.json"

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                agents = []
                for agent_data in data.get("agents", []):
                    agents.append(
                        AgentConfig(
                            name=agent_data["name"],
                            display_name=agent_data.get(
                                "display_name", agent_data["name"].title()
                            ),
                            home_dir=agent_data.get("home_dir"),
                            enabled=agent_data.get("enabled", True),
                        )
                    )
                return agents
        except (json.JSONDecodeError, KeyError, IOError):
            pass

    # Return defaults if config doesn't exist or is invalid
    return get_default_agents()


def search_all_agents(
    keywords: List[str],
    global_search: bool = False,
    num_matches: int = 10,
    agents: Optional[List[str]] = None,
    claude_home: Optional[str] = None,
    codex_home: Optional[str] = None,
    opencode_home: Optional[str] = None,
) -> List[dict]:
    """
    Search sessions across all enabled agents.

    Returns list of dicts with agent metadata added.
    """
    agent_configs = load_config()

    # Filter by requested agents if specified
    if agents:
        agent_configs = [a for a in agent_configs if a.name in agents]

    # Filter by enabled agents
    agent_configs = [a for a in agent_configs if a.enabled]

    all_sessions = []

    for agent_config in agent_configs:
        if agent_config.name == "claude":
            # Search Claude sessions
            home = claude_home or agent_config.home_dir
            sessions = find_claude_sessions(
                keywords, global_search=global_search, claude_home=home
            )

            # Add agent metadata to each session
            for session in sessions:
                if len(session) >= 11:
                    first_message = session[5]
                    last_message = session[6]
                    match_score = session[7]
                    cwd = session[8]
                    branch = session[9]
                    best_chunk = session[10]
                elif len(session) >= 10:
                    first_message = session[5]
                    last_message = session[6]
                    match_score = session[7]
                    cwd = session[8]
                    branch = session[9]
                    best_chunk = None
                else:
                    preview = session[5] if len(session) > 5 else ""
                    first_message = preview
                    last_message = preview
                    match_score = 0.0
                    cwd = session[6] if len(session) > 6 else ""
                    branch = session[7] if len(session) > 7 else ""
                    best_chunk = None
                session_dict = {
                    "agent": "claude",
                    "agent_display": agent_config.display_name,
                    "session_id": session[0],
                    "mod_time": session[1],
                    "create_time": session[2],
                    "lines": session[3],
                    "project": session[4],
                    "first_message": first_message,
                    "last_message": last_message,
                    "match_score": float(match_score or 0.0),
                    "best_chunk": best_chunk,
                    "cwd": cwd,
                    "branch": branch or "",
                    "claude_home": home,
                }
                all_sessions.append(session_dict)

        elif agent_config.name == "codex":
            # Search Codex sessions
            home = codex_home or agent_config.home_dir
            codex_home_path = get_codex_home(home)

            if codex_home_path.exists():
                sessions = find_codex_sessions(
                    codex_home_path,
                    keywords,
                    num_matches=num_matches * 2,  # Get more for merging
                    global_search=global_search,
                )

                # Add agent metadata to each session
                for session in sessions:
                    preview = session.get("preview", "")
                    session_dict = {
                        "agent": "codex",
                        "agent_display": agent_config.display_name,
                        "session_id": session["session_id"],
                        "mod_time": session["mod_time"],
                        "create_time": session.get("mod_time"),
                        "lines": session["lines"],
                        "project": session["project"],
                        "first_message": session.get("first_message", preview),
                        "last_message": session.get("last_message", preview),
                        "match_score": float(session.get("match_score", 0.0) or 0.0),
                        "best_chunk": session.get("best_chunk"),
                        "cwd": session["cwd"],
                        "branch": session.get("branch", ""),
                        "file_path": session.get("file_path", ""),
                    }
                    all_sessions.append(session_dict)

        elif agent_config.name == "opencode":
            # Search OpenCode sessions
            home = opencode_home or agent_config.home_dir
            opencode_home_path = get_opencode_home(home)

            if opencode_home_path.exists():
                sessions = find_opencode_sessions(
                    opencode_home_path,
                    keywords,
                    num_matches=num_matches * 2,  # Get more for merging
                    global_search=global_search,
                )

                # Add agent metadata to each session
                for session in sessions:
                    preview = session.get("preview", "")
                    session_dict = {
                        "agent": "opencode",
                        "agent_display": agent_config.display_name,
                        "session_id": session["session_id"],
                        "mod_time": session["mod_time"],
                        "create_time": session.get("create_time", session["mod_time"]),
                        "lines": session["lines"],
                        "project": session["project"],
                        "first_message": session.get("first_message", preview),
                        "last_message": session.get("last_message", preview),
                        "match_score": float(session.get("match_score", 0.0) or 0.0),
                        "best_chunk": session.get("best_chunk"),
                        "cwd": session["cwd"],
                        "branch": session.get("branch", ""),
                    }
                    all_sessions.append(session_dict)

    if keywords:
        all_sessions.sort(
            key=lambda x: (float(x.get("match_score", 0.0) or 0.0), x["mod_time"]),
            reverse=True,
        )
    else:
        all_sessions.sort(key=lambda x: x["mod_time"], reverse=True)

    return all_sessions[:num_matches]


def format_session_item(session: dict, idx: int, keywords: List[str]) -> str:
    """Format a session for display in the list."""
    mod_time = session["mod_time"]
    date_str = datetime.fromtimestamp(mod_time).strftime("%m/%d %H:%M")
    branch = session.get("branch", "") or ""
    branch_str = f" [{branch}]" if branch else ""

    first_msg = (session.get("first_message", "") or "").replace("\n", " ")[:150]
    last_msg = (session.get("last_message", "") or "").replace("\n", " ")[:150]

    lines = [
        f"[bold yellow]#{idx}[/bold yellow]  [magenta]{session['agent_display']}[/magenta]  "
        f"[dim]{session['session_id']}[/dim]  [green]{session['project']}[/green]{branch_str}  "
        f"[blue]{date_str}[/blue]  [cyan]{session['lines']} lines[/cyan]",
        f"  [white]First:[/white] {first_msg}",
        f"  [white]Last:[/white]  {last_msg}",
    ]

    if keywords:
        score = session.get("match_score", 0.0) or 0.0
        chunk = (session.get("best_chunk", "") or "").replace("\n", " ")[:200]
        lines.append(f"  [yellow]Score: {score:.0f}[/yellow]  [white]Match:[/white] {chunk}")

    return "\n".join(lines)


class SessionItem(ListItem):  # type: ignore[misc]
    """A session list item."""

    def __init__(self, session: dict, idx: int, keywords: List[str]) -> None:
        self.session = session
        self.idx = idx
        self.keywords = keywords
        super().__init__()

    def compose(self) -> "ComposeResult":  # type: ignore[override]
        yield Static(format_session_item(self.session, self.idx, self.keywords))


class SessionPicker(App):  # type: ignore[misc]
    """Textual TUI for selecting a session."""

    CSS = """
    ListView {
        height: 1fr;
    }
    ListItem {
        padding: 0 1;
        margin: 0;
    }
    ListItem:hover {
        background: $surface-lighten-1;
    }
    ListItem.-active {
        background: $primary-background-lighten-2;
    }
    Static {
        padding: 1 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
        Binding("enter", "select", "Select"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
    ]

    selected_session: reactive[Optional[dict]] = reactive(None)

    def __init__(self, sessions: List[dict], keywords: List[str]) -> None:
        self.sessions = sessions
        self.keywords = keywords
        super().__init__()

    def compose(self) -> "ComposeResult":  # type: ignore[override]
        yield Header()
        with VerticalScroll():
            # Reverse so best match is at bottom (nearest prompt)
            reversed_sessions = list(reversed(self.sessions))
            yield ListView(
                *[
                    SessionItem(session, len(self.sessions) - i, self.keywords)
                    for i, session in enumerate(reversed_sessions)
                ],
                id="session-list",
            )
        yield Footer()

    def on_list_view_selected(self, event: "ListView.Selected") -> None:  # type: ignore[name-defined]
        """Handle session selection."""
        item = event.item
        if isinstance(item, SessionItem):
            self.selected_session = item.session
            self.exit()

    def action_select(self) -> None:
        """Select the current item."""
        list_view = self.query_one("#session-list", ListView)
        if list_view.highlighted_child is not None:
            item = list_view.highlighted_child
            if isinstance(item, SessionItem):
                self.selected_session = item.session
                self.exit()


def display_textual_ui(
    sessions: List[dict], keywords: List[str]
) -> Optional[dict]:
    """Display Textual TUI for session selection."""
    if not TEXTUAL_AVAILABLE:
        return None

    app = SessionPicker(sessions, keywords)
    app.run()
    return app.selected_session


def show_action_menu(session: dict, stderr_mode: bool = False) -> Optional[str]:
    """Show action menu for selected session."""
    output = sys.stderr if stderr_mode else sys.stdout

    print(f"\n=== Session: {session['session_id']} ===", file=output)
    print(f"Agent: {session['agent_display']}", file=output)
    print(f"Project: {session['project']}", file=output)
    if session.get("branch"):
        print(f"Branch: {session['branch']}", file=output)
    print(f"\nWhat would you like to do?", file=output)
    print("1. Resume session (default)", file=output)
    print("2. Show session file path", file=output)
    print("3. Copy session file to file (*.jsonl) or directory", file=output)
    print(file=output)

    try:
        if stderr_mode:
            # In stderr mode, prompt to stderr so it's visible
            sys.stderr.write("Enter choice [1-3] (or Enter for 1): ")
            sys.stderr.flush()
            choice = sys.stdin.readline().strip()
        else:
            choice = input("Enter choice [1-3] (or Enter for 1): ").strip()

        if not choice or choice == "1":
            return "resume"
        elif choice == "2":
            return "path"
        elif choice == "3":
            return "copy"
        else:
            print("Invalid choice.", file=output)
            return None
    except KeyboardInterrupt:
        print("\nCancelled.", file=output)
        return None


def handle_action(session: dict, action: str, shell_mode: bool = False) -> None:
    """Handle the selected action based on agent type."""
    agent = session["agent"]

    if action == "resume":
        if agent == "claude":
            resume_claude_session(
                session["session_id"],
                session["cwd"],
                shell_mode=shell_mode,
                claude_home=session.get("claude_home"),
            )
        elif agent == "codex":
            resume_codex_session(
                session["session_id"], session["cwd"], shell_mode=shell_mode
            )
        elif agent == "opencode":
            resume_opencode_session(
                session["session_id"], session["cwd"], shell_mode=shell_mode
            )

    elif action == "path":
        if agent == "claude":
            file_path = get_claude_session_file_path(
                session["session_id"],
                session["cwd"],
                claude_home=session.get("claude_home"),
            )
            print(f"\nSession file path:")
            print(file_path)
        elif agent == "codex":
            print(f"\nSession file path:")
            print(session.get("file_path", "Unknown"))
        elif agent == "opencode":
            print(f"\nSession ID:")
            print(session["session_id"])

    elif action == "copy":
        if agent == "claude":
            file_path = get_claude_session_file_path(
                session["session_id"],
                session["cwd"],
                claude_home=session.get("claude_home"),
            )
            copy_claude_session_file(file_path)
        elif agent == "codex":
            copy_codex_session_file(session.get("file_path", ""))
        elif agent == "opencode":
            copy_opencode_session_file(session["session_id"])


def main():
    parser = argparse.ArgumentParser(
        description="Unified session finder - search across multiple coding agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    find-session "langroid,MCP"           # Search all agents in current project
    find-session -g                       # Show all sessions across all projects
    find-session "bug" --agents claude    # Search only Claude sessions
    find-session "error" --agents codex   # Search only Codex sessions
    find-session "pdf" --agents opencode  # Search only OpenCode sessions
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
        help="Search across all projects, not just the current one",
    )
    parser.add_argument(
        "-n",
        "--num-matches",
        type=int,
        default=20,
        help="Number of matching sessions to display (default: 20)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["claude", "codex", "opencode"],
        help="Limit search to specific agents (default: all)",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Output shell commands for evaluation (for use with shell function)",
    )
    parser.add_argument(
        "--claude-home", type=str, help="Path to Claude home directory (default: ~/.claude)"
    )
    parser.add_argument(
        "--codex-home", type=str, help="Path to Codex home directory (default: ~/.codex)"
    )
    parser.add_argument(
        "--opencode-home", type=str, help="Path to OpenCode data directory (default: ~/.local/share/opencode)"
    )

    args = parser.parse_args()

    # Parse keywords
    keywords = (
        [k.strip() for k in args.keywords.split(",") if k.strip()]
        if args.keywords
        else []
    )

    # Search all agents
    matching_sessions = search_all_agents(
        keywords,
        global_search=args.global_search,
        num_matches=args.num_matches,
        agents=args.agents,
        claude_home=args.claude_home,
        codex_home=args.codex_home,
        opencode_home=args.opencode_home,
    )

    if not matching_sessions:
        scope = "all projects" if args.global_search else "current project"
        keyword_msg = (
            f" containing all keywords: {', '.join(keywords)}" if keywords else ""
        )
        if RICH_AVAILABLE:
            console = Console()
            console.print(f"[yellow]No sessions found{keyword_msg} in {scope}[/yellow]")
        else:
            print(f"No sessions found{keyword_msg} in {scope}", file=sys.stderr)
        sys.exit(0)

    # Display Textual TUI
    if TEXTUAL_AVAILABLE:
        selected_session = display_textual_ui(matching_sessions, keywords)
        if selected_session:
            # Show action menu
            action = show_action_menu(selected_session, stderr_mode=args.shell)
            if action:
                handle_action(selected_session, action, shell_mode=args.shell)
    else:
        # Fallback without Textual
        print("\nMatching sessions:")
        for idx, session in enumerate(matching_sessions[: args.num_matches], 1):
            print(
                f"{idx}. [{session['agent_display']}] {session['session_id']} | "
                f"{session['project']} | {session.get('branch', 'N/A')}"
            )


if __name__ == "__main__":
    main()
