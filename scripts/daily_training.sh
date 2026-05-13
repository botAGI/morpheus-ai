#!/bin/bash
# Morpheus Daily Training Pipeline
# Run via launchd/cron at end of day
# Usage: ./daily_training.sh [project_dir]

set -e

PROJECT_DIR="${1:-$HOME/Projects/morpheus}"
LOG_FILE="$HOME/.morpheus/logs/training_$(date +%Y%m%d).log"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M')] $1" | tee -a "$LOG_FILE"
}

log "=== Morpheus Daily Training ==="

cd "$PROJECT_DIR"

# 1. Consolidate yesterday's sessions
log "Step 1: Consolidating sessions..."
morpheus consolidate --days 1 --output dataset.jsonl >> "$LOG_FILE" 2>&1

if [ ! -s dataset.jsonl ]; then
    log "No sessions to train on. Skipping."
    exit 0
fi

# 2. Run training
log "Step 2: Starting QLoRA training..."
morpheus train --base-model qwen2.5:7b --dataset dataset.jsonl --output-dir morpheus_adapters/daily --epochs 3 >> "$LOG_FILE" 2>&1

# 3. Keep only last 7 adapters
log "Step 3: Cleaning old adapters..."
ADAPTERS_DIR="morpheus_adapters/daily"
if [ -d "$ADAPTERS_DIR" ]; then
    cd "$ADAPTERS_DIR"
    ls -t | tail -n +8 | xargs rm -rf 2>/dev/null || true
fi

log "=== Training complete ==="
