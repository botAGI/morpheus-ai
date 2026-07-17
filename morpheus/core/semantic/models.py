"""Pydantic models for semantic review candidates."""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


CandidateKind = Literal[
    "current_state",
    "active_decision",
    "open_task",
    "outdated_claim",
    "agent_rule",
    "source_reference",
]
CandidateLabel = Literal["source_backed", "inferred", "needs_review"]
CandidateStatus = Literal["pending", "accepted", "rejected"]
CandidateClass = Literal[
    "architecture",
    "implementation",
    "product",
    "security",
    "command",
    "integration",
    "stale",
    "convention",
    "open_task",
    "temporary",
    "unknown",
]
TrainabilityStatus = Literal[
    "trainable",
    "negative_example",
    "eval_only",
    "retrievable",
    "needs_review",
    "unsafe",
    "excluded",
]
MemoryRoute = Literal[
    "adapter_training",
    "negative_example",
    "eval_only",
    "retrieval",
    "prompt_context",
    "human_review",
    "stale_archive",
    "excluded",
]
SemanticSourceCategory = Literal[
    "docs_state_sources",
    "build_manifest_sources",
    "cli_api_sources",
    "workflow_sources",
]


class SemanticSource(BaseModel):
    path: str
    category: SemanticSourceCategory
    sha256: str
    size_bytes: int
    line_count: int
    modified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: str = ""


class SemanticCandidate(BaseModel):
    id: str
    run_id: str
    kind: CandidateKind
    claim: str
    source_path: str
    source_sha256: str
    source_mtime: datetime
    source_revision: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    evidence_excerpt: str = Field(min_length=1)
    evidence_sha256: str = Field(min_length=64, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)
    label: CandidateLabel
    semantic_class: CandidateClass = "unknown"
    trainability_status: TrainabilityStatus = "needs_review"
    trainability_reason: str = "unrouted"
    memory_route: MemoryRoute = "human_review"
    status: CandidateStatus = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provider: dict
    prompt_sha256: str = Field(min_length=64, max_length=64)
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_reason: str | None = None
