"""
Compile routes
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/compile", tags=["compile"])

class CompileRequest(BaseModel):
    project_root: Optional[str] = None

# Main compile logic is in server.py
# This file is for additional compile endpoints if needed
