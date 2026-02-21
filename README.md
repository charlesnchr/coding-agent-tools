# Coding Agent Tools

Small CLI utilities for searching coding-agent sessions and viewing usage across agents.

## Install

```bash
uv tool install coding-agent-tools
```

Or:

```bash
pipx install coding-agent-tools
# or
pip install coding-agent-tools
```

Installed commands:

- `find-session`
- `agent-usage`

## Data Sources

- **Claude Code**
  - `find-session`: `~/.claude/projects/*/*.jsonl`
  - `agent-usage`: delegates to native `ccusage`
- **Codex**
  - `find-session` and `agent-usage`: `~/.codex/sessions/**/rollout-*.jsonl`
- **OpenCode**
  - `find-session` and `agent-usage`: `~/.local/share/opencode/opencode.db`
- **OpenClaw**
  - `agent-usage`: `~/.openclaw/agents/*/sessions/*.jsonl`
- **OpenWhispr**
  - `agent-usage`: `~/Library/Application Support/open-whispr/transcriptions.db`

## Demos

### `find-session`

![find-session demo](demo/find-session.gif)

### `agent-usage`

![agent-usage demo](demo/agent-usage.gif)

## References

- ccusage: https://github.com/ryoppippi/ccusage
- pchalasani/claude-code-tools: https://github.com/pchalasani/claude-code-tools
- LiteLLM pricing registry: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

## License

MIT
