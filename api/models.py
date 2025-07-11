"""
Pydantic models for API requests and responses
"""
from datetime import datetime
from typing import Optional, Dict, Any, List

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for document query"""
    request_id: str 
    question: str = Field(..., description="User question", min_length=1, max_length=1000)


class SourceDocument(BaseModel):
    """Source document information"""
    document_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_index: str
    score: float
    content: str


class QueryResponse(BaseModel):
    """Response model for document query"""
    request_id: str
    answer: str
    sources: List[Document]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    components: Dict[str, str]
    timestamp: datetime
    