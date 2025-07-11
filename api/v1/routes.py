
import logging
from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_rag_service
from api.models import QueryRequest, QueryResponse

from rag.orchestrator import RAGOrchestrator
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["RAG"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest, 
    service:RAGOrchestrator=Depends(get_rag_service)
):
    """
    The primary endpoint for queries to RAG.
    """
    try:
        # NOTE: There is no validation or authentication yet.
        # TODO: Add API key authentication and input validation.
        result = service.invoke(query=request.question)
        return QueryResponse(
            request_id=request.request_id, 
            answer=result.answer_text,
            sources=result.source_chunks
            )
    except Exception as e:
        logger.error(f"Query failed for request {request.request_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {e}"
        )
    