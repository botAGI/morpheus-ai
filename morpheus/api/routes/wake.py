"""
Wake routes
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/wake", tags=["wake"])

class WakeRequest(BaseModel):
    project: str

# Main wake logic is in server.py
