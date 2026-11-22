"""
API Request and Response Models
Pydantic models for input validations and resposne structure
"""

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timezone

class ChatRequest(BaseModel):
    """
    Incoming char request"""
    message: str=Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's message to the agent"
    )
    thread_id:str = Field(
        default="default",
        description="Conversation thread ID"

    )

class ChatResponse(BaseModel):
    """Chat response returned to the client"""
    response:str
    thread_id:str
    model_used:str
    cached:bool = False
    processing_time_ms: float
    security_notes: List[str]
    timestamp:str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class HealthResponse(BaseModel):
    """Health check response"""

    status:str = "healthy"
    environment:str
    version:str = "1.0.0"
    checks: dict = {}

class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    error_rate: str
    avg_latemmcy_ms: float  # (Keeping your typo so Pydantic matches)
    total_input_tokens: int
    total_output_tokens: int
    cache_hit_rate: str
    cache_hits: int
    cache_misses: int

class ErrorResponse(BaseModel):
    """Standard error response."""
    error:str
    detail:str |None = None
    request_id:str |None = None