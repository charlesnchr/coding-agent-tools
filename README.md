# CodingAgentTools

`CodingAgentTools` is a focused toolbox for local AI coding workflows.

It currently ships two high-leverage commands:

- `find-session`: search and resume sessions across Claude Code, Codex, and OpenCode.
- `agent-usage`: unified usage + estimated cost reporting across Claude Code, Codex, OpenClaw, OpenCode, and OpenWhispr.

## Why this repo exists

This repository consolidates tools that were previously split across multiple projects/scripts:

- `find-session` (now embedded here)
- local `allusage` script (now `agent-usage`)

The goal is one installable package, one docs surface, and one place for improvements.

## Install

```bash
uv tool install .
```

or with pip:

```bash
pip install .
```

## Command: find-session

Search and resume sessions across agents from one interface.

```bash
# Search current project
find-session "auth,refactor"

# Search across all projects
find-session -g

# Limit to one agent
find-session "redis" --agents codex
```

Supported agents:

- Claude Code (`~/.claude/projects/*.jsonl`)
- Codex (`~/.codex/sessions/**/rollout-*.jsonl`)
- OpenCode (`~/.local/share/opencode/opencode.db`)

## Command: agent-usage

Show token usage and estimated USD-equivalent cost by period.

```bash
# Daily usage for all sources
agent-usage

# Weekly with per-model breakdown
agent-usage weekly --breakdown

# Only OpenCode + Codex
agent-usage monthly opencode codex
```

### Sources and aliases

- `claude` / `cc`
- `codex` / `cx`
- `openclaw` / `oc` / `claw`
- `opencode` / `oe` / `code`
- `openwhispr` / `ow` / `whispr`

### Pricing model

`agent-usage` estimates cost from token counts using live LiteLLM pricing metadata.

- If a source already reports non-zero cost, estimated pricing is still preferred when model pricing is available.
- This avoids misleading `0` totals for subscription-backed usage (for example OpenCode sessions routed through subscription plans).
- For models not present in the pricing registry, the tool falls back to observed/source-reported cost when available.

## References

- `ccusage` (inspiration for usage/cost reporting style): https://github.com/ryoppippi/ccusage
- `claude-code-tools` (Cloud/Claude Code tools reference and prior consolidation work): https://github.com/charlesnchr/claude-code-tools
- LiteLLM model pricing registry: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

## Development

```bash
python3 -m pip install -e .
find-session --help
agent-usage --help
```

## License

MIT
