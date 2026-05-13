"""
Verify routes
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/verify", tags=["verify"])

class VerifyRequest(BaseModel):
    project_root: Optional[str] = None

# Main verify logic is in server.py
