# Morpheus Training Pipeline

Phase 3: Daily LoRA fine-tuning for "weights-as-memory" effect.

## Quick Start

```bash
cd morpheus-ai

# 1. Consolidate last 7 days of sessions into dataset
morpheus consolidate --days 7 --output dataset.jsonl

# 2. Train LoRA adapter
morpheus train --base-model qwen2.5:7b --dataset dataset.jsonl --output-dir morpheus_adapters/daily

# 3. Evaluate
morpheus eval --adapter-path morpheus_adapters/daily
```

## Pipeline

```
sessions/*.jsonl  →  consolidate.py  →  dataset.jsonl
                                            ↓
                                       train.py  →  adapter/
                                            ↓
                                       eval.py  →  results.jsonl
```

## How It Works

1. **Consolidate**: Reads OpenClaw sessions, extracts Q&A pairs, filters system/infrastructure messages
2. **Train**: QLoRA fine-tuning via LlamaFactory (4-bit, DoRA)
3. **Eval**: Tests adapter on held-out questions

## Daily Automation

Add to crontab for end-of-day training:

```bash
# Edit crontab
crontab -e

# Add line:
0 23 * * * /path/to/morpheus-ai/scripts/daily_training.sh
```

Or use launchd on macOS:

```bash
cp scripts/com.morpheus.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.morpheus.daily.plist
```

## Requirements

- Python 3.10+
- llamafactory-cli: `pip install llamafactory`
- Ollama with qwen2.5:7b model
- GPU with 8GB+ VRAM (for 7B model training)
