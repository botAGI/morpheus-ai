#!/bin/bash
# Morpheus safe daily learning-lab gate.
#
# This script intentionally does not train or activate adapters. It refreshes
# source-backed project state, verifies receipts, builds the autonomous dogfood
# learning-lab dataset, and prints status so a human can decide whether to run a
# separate training/eval pass.

set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M')] $1"
}

log "=== Morpheus safe daily lab gate ==="
cd "$PROJECT_DIR"

log "Step 1: Refresh private WAKE state"
morpheus wake . --private

log "Step 2: Verify receipt chain"
morpheus verify --all

log "Step 3: Build strict dogfood lab artifacts without training"
morpheus learn lab . --dogfood --no-train

log "Step 4: Show effective learning status"
morpheus learn status

log "No adapter was trained or activated by this script."
