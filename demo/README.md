# Demo

This folder contains a safe terminal demo scaffold for the first public launch.
It does not call cloud APIs and does not require a GPU, Obsidian, OpenClaw, or
Hermes.

## Record

Install `asciinema`, then record:

```bash
asciinema rec demo.cast -- ./demo/record_demo.sh
```

Convert the recording to GIF with `agg`:

```bash
agg demo.cast demo.gif
```

The demo creates a temporary project, writes deterministic source files with
`DECISION:`, `TODO:`, and `NOTE:` markers, then runs:

```bash
morpheus wake .
morpheus verify --all
morpheus stale .
```

It ends with a copyable prompt:

```text
Paste this into an agent: Read WAKE.md and continue.
```
