# Parametric Memory

Historical research direction:

```text
weights-as-memory via reviewed consolidation datasets
```

The original Morpheus research direction was project/session context becoming a
verified consolidation dataset, then a LoRA or QLoRA adapter, so an agent wakes
up with stable project knowledge in weights.

In v0.2 and later, the truth layer is the data quality gate before weights:

- semantic compile extracts source-backed candidates
- review/check keeps state honest
- accepted candidates become dataset rows
- eval decides whether adapter learning helped
- activation and rollback keep deployment controlled

This direction is active, but adapter output is not the source of truth. Source
spans and local checks remain the authority.

