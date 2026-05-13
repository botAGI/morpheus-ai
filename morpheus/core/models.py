"""
Core Pydantic models for Morpheus.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    id: str
    path: str
    kind: str = "file"
    sha256: str = ""
    size_bytes: int = 0
    line_count: int = 0
    modified_at: datetime = Field(default_factory=datetime.utcnow)


class Claim(BaseModel):
    id: str
    source_id: str
    line_start: int
    line_end: int
    excerpt: str
    status: str = "active"
    category: str = "fact"
    inference: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Evidence(BaseModel):
    id: str
    claim_id: str
    source_id: str
    path: str
    line_start: int
    line_end: int
    excerpt: str
    source_sha256: str
    excerpt_sha256: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Receipt(BaseModel):
    schema_version: str = "morpheus-receipt/1"
    receipt_id: str = ""
    project: dict = Field(default_factory=dict)
    wake_md_sha256: str = ""
    state_json_sha256: str = ""
    evidence_jsonl_sha256: str = ""
    sources: list = Field(default_factory=list)
    claim_count: dict = Field(default_factory=dict)
    tool: dict = Field(default_factory=dict)
    issued_at: str = ""
    previous_receipt_sha256: Optional[str] = None
    signature: dict = Field(default_factory=dict)


class ProjectState(BaseModel):
    sources: list[Source] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    compiled_at: datetime = Field(default_factory=datetime.utcnow)
    receipt_id: Optional[str] = None
