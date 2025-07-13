"""
Pydantic models for API requests and responses
"""
import re
import datetime
from typing import Optional, Dict, Any, List

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request model for document query"""
    request_id: str = Field(
        ...,
        description="Unique request identifier for tracing",
        min_length=1,
        max_length=100
    )
    question: str = Field(
        ..., 
        description="User question to be answered", 
        min_length=1, 
        max_length=10_000
    )
    
    @field_validator("question")
    def validate_question(cls, v):
        """Validate and sanitize question input."""
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()
        
    @field_validator("request_id")
    def validate_request_id(cls, v):
        """Validate request ID format"""
        if not re.match(r"^[a-zA-Z0-9\-_]+$", v):
            raise ValueError("Invalid request_id format")
        return v


class SourceDocument(BaseModel):
    """Source document information"""
    document_id: str = Field(
        ...,
        description="Unique identifier for the document chunk"
    )
    filename: str = Field(
        ...,
        description="Original source filename or URI"
    )
    page_number: Optional[int] = Field(
        None,
        ge=1,
        description="Page number within document, if paginated"
    )
    chunk_index: str = Field(
        ...,
        description="Logical chunk index ot path within the document"
    )
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Relevance score (0-1) assigned by retriever")
    content: str = Field(
        ..., 
        min_length=1,
        max_length=1_000_000,
        description="Text payload of this chunk"
        )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata",
        )


class QueryResponse(BaseModel):
    """Response model for document query"""
    request_id: str = Field(
        ...,
        description="Unique request identifier for tracing",
        min_length=1,
        max_length=100
    )
    answer: str = Field(
        ..., 
        description="Answer from the system", 
        min_length=1, 
        max_length=10_000
    )
    sources: Optional[List[Document]] = Field(
        None,
        description="List of source documents used to answer the question"
    )

    @field_validator("answer")
    def validate_question(cls, v):
        """Validate answer content."""
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()
    
    @field_validator("sources")
    def validate_sources(cls, v):
        """Validate answer content."""
        if len(v) > 100:
            raise ValueError("Too many source documents")
        return v
    

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(
        ...,
        description="Overall service health status: healthy, degraded, or unhealthy"
    )
    version: str = Field(
        ...,
        description="Service semantic version"
    )
    components: Dict[str, str] = Field(
        ...,
        description="Mapping of individual component names to their health status"
    )
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now(datetime.timezone.utc),
        description="UTC time when health was checked"
    )
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total seconds the service has been up"
    )

    @field_validator("status")
    def validate_status(cls, v):
        """Validate health status."""
        if v not in {'healthy', 'unhealthy', 'degraded'}:
            raise ValueError("Invalid health status")
        return v

    @field_validator("timestamp")
    def validate_timestamp(cls, v) -> datetime.datetime:
        """Reject timestamps set in the future.."""
        if v > datetime.datetime.now(datetime.timezone.utc):
            raise ValueError("timestamp cannot be in the future")
        return v
    

class ErrorResponse(BaseModel):
    """Standardized error response model."""
    
    error_code: str = Field(..., description="Error code for client handling")
    message: str = Field(..., description="User-friendly error message")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="UTC time when the health check was performed"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "error_code": "VALIDATION_ERROR",
                "message": "Invalid input parameters",
                "request_id": "req-123456",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response with detailed component status."""
    
    status: str = Field(..., description="Overall service status")
    version: str = Field(..., description="Service version")
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Individual component health status"
    )
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="UTC time when the health check was performed"
    )
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "database": {"status": "healthy", "latency_ms": 15},
                    "llm": {"status": "healthy", "latency_ms": 850},
                    "vector_store": {"status": "healthy", "size": 12500}
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
