#!/usr/bin/env bash
set -euo pipefail

if ! command -v morpheus >/dev/null 2>&1; then
  echo "morpheus CLI not found. Install the project or run from an activated dev environment."
  exit 1
fi

DEMO_DIR="$(mktemp -d "${TMPDIR:-/tmp}/morpheus-wake-demo.XXXXXX")"
echo "Demo project: ${DEMO_DIR}"
cd "${DEMO_DIR}"

cat > README.md <<'EOF'
# Cold Start Demo

This is a small project used to show what changes when an agent has WAKE.md.

DECISION: The project uses WAKE.md as the agent continuity file.
NOTE: The demo stays local and deterministic.
EOF

cat > SPEC.md <<'EOF'
# Demo Specification

DECISION: Morpheus compiles explicit state markers into WAKE.md.
TODO: Ask the next agent to read WAKE.md before making changes.
NOTE: Receipts make the compiled state verifiable.
EOF

echo
echo "Without Morpheus:"
echo "Agent: I do not have enough context."
echo
echo "With Morpheus:"
morpheus wake .
morpheus verify --all
morpheus stale .
echo
echo "Paste this into an agent: Read WAKE.md and continue."
