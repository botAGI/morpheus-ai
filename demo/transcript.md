# Demo Transcript

## Without Morpheus

```text
User: What changed in this project?
Agent: I do not have enough context.
```

## With Morpheus

```bash
uvx --from morpheus-wake morpheus wake .
```

```text
Morpheus compiles project files into WAKE.md, signs a receipt, verifies the
receipt chain, and prints an agent handoff prompt.
```

## Agent Prompt

```text
Read WAKE.md and continue.
```
