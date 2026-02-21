# CodingAgentTools

`coding-agent-tools` is a practical CLI bundle for local coding-agent workflows.

It currently provides:

- `find-session`: search and resume sessions across Claude Code, Codex, and OpenCode
- `agent-usage`: unified usage and estimated USD-equivalent cost across Claude Code, Codex, OpenClaw, OpenCode, and OpenWhispr

## Why this repo exists

This consolidates earlier standalone tools into one package:

- `find-session` (now here)
- local `allusage` script (now `agent-usage`)

The goal is one install, one command surface, one place to improve.

## Install

From PyPI (recommended):

```bash
uv tool install coding-agent-tools
```

Alternative:

```bash
pipx install coding-agent-tools
# or
pip install coding-agent-tools
```

From source:

```bash
uv tool install .
```

## Quickstart

```bash
# Find sessions in current project across all supported agents
find-session "auth,refactor"

# Search all projects
find-session -g

# Usage report (daily default)
agent-usage

# Weekly report with per-model expansion
agent-usage weekly --breakdown
```

## `find-session`

Unified session search with interactive selection, resume, and export/copy actions.

```bash
# Search current project
find-session "redis,bug"

# Search all projects
find-session -g

# Limit to one agent
find-session "checkpoint" --agents codex
```

### Preview behavior

Session previews now include both ends of user intent:

- `First: ...`
- `Last: ...`

If only one meaningful user message is found, it displays as `First/Last: ...`.

## `agent-usage`

Token usage and estimated cost tables by period.

```bash
# Daily usage for all sources
agent-usage

# Weekly with per-model breakdown
agent-usage weekly --breakdown

# Restrict sources
agent-usage monthly opencode codex
```

### Sources and aliases

- `claude` / `cc` / `claudecode` (native `ccusage`)
- `codex` / `cx`
- `openclaw` / `oc` / `claw`
- `opencode` / `oe` / `code`
- `openwhispr` / `ow` / `whispr`

### Claude source = native ccusage

For Claude data, `agent-usage` now delegates directly to `ccusage` (no custom reimplementation).

- Uses `ccusage` if installed
- Otherwise falls back to `npx --yes ccusage@latest`
- Requires Node.js + `npx` unless you already have `ccusage` installed

### Cost model

`agent-usage` estimates costs from token counts using live LiteLLM pricing metadata.

- This avoids misleading `$0` totals for subscription-backed traffic (especially OpenCode).
- If a model is missing in the pricing registry, it falls back to source-observed cost when present.

## Data sources

- Claude Code: native `ccusage`
- Codex: `~/.codex/sessions/**/rollout-*.jsonl`
- OpenCode: `~/.local/share/opencode/opencode.db`
- OpenClaw: `~/.openclaw/agents/*/sessions/*.jsonl`
- OpenWhispr: `~/Library/Application Support/open-whispr/transcriptions.db`

## Beautiful terminal demos (yes, via subprocess recording)

Yes: you can generate polished terminal renders by recording commands as subprocesses.

This repo includes ready-to-run VHS tapes in `demo/`:

```bash
# macOS
brew install vhs

# render animated gifs from deterministic command scripts
./scripts/render-demos.sh
```

How it works:

- `vhs` launches a shell subprocess
- runs scripted commands (sample queries and real command output)
- renders a styled terminal recording directly to GIF

Use these as README assets, release previews, or social clips.

### `find-session` demo

![find-session demo](demo/find-session.gif)

<br/>
<br/>

### `agent-usage` demo

![agent-usage demo](demo/agent-usage.gif)

## Development

```bash
python3 -m pip install -e .
find-session --help
agent-usage --help
```

## References

- `ccusage` (inspiration for usage/cost reporting style): https://github.com/ryoppippi/ccusage
- `claude-code-tools` (Cloud/Claude Code tools reference and prior consolidation work): https://github.com/charlesnchr/claude-code-tools
- LiteLLM model pricing registry: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

## License

MIT
