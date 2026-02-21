# coding-agent-tools

Unified CLI for searching sessions and viewing token usage across coding agents
(Claude Code, Codex, OpenCode, OpenClaw, OpenWhispr).

## Install

```bash
uv tool install coding-agent-tools
```

Or via `pipx install coding-agent-tools` / `pip install coding-agent-tools`.

Requires Python 3.11+. The only runtime dependency is [Rich](https://github.com/Textualize/rich).

## Commands

### `find-session`

Search and resume sessions across agents. Shows an interactive table where you
can select a session to resume it, view its file path, or copy it.

```
find-session "langroid,MCP"           # keyword search (AND logic)
find-session -g                       # all sessions across all projects
find-session "bug" --agents claude    # limit to a specific agent
find-session -n 20                    # show more results
```

![find-session](demo/find-session.gif)

### `agent-usage`

Token and cost analytics. Fetches pricing from the
[LiteLLM registry](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
to estimate costs. Claude Code usage is delegated to
[ccusage](https://github.com/ryoppippi/ccusage).

```
agent-usage                           # daily usage (default)
agent-usage weekly --breakdown        # weekly with per-model breakdown
agent-usage monthly codex opencode    # specific sources only
```

![agent-usage](demo/agent-usage.gif)

## Supported agents

| Agent | Session search | Usage analytics | Data location |
|-------|:-:|:-:|---|
| Claude Code | yes | yes (via ccusage) | `~/.claude/projects/*/*.jsonl` |
| Codex | yes | yes | `~/.codex/sessions/**/rollout-*.jsonl` |
| OpenCode | yes | yes | `~/.local/share/opencode/opencode.db` |
| OpenClaw | -- | yes | `~/.openclaw/agents/*/sessions/*.jsonl` |
| OpenWhispr | -- | yes | `~/Library/Application Support/open-whispr/transcriptions.db` |

## Acknowledgements

- [ccusage](https://github.com/ryoppippi/ccusage) -- Claude Code usage tracking
- [claude-code-tools](https://github.com/pchalasani/claude-code-tools) -- prior art

## License

MIT
