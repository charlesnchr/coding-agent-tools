#!/usr/bin/env bash
set -euo pipefail

if ! command -v vhs >/dev/null 2>&1; then
  echo "vhs is required. Install with: brew install vhs" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

vhs "$REPO_ROOT/demo/find-session.tape"
vhs "$REPO_ROOT/demo/agent-usage.tape"

echo "Rendered demo/find-session.gif and demo/agent-usage.gif"
