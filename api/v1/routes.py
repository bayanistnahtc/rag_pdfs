
import logging
import time
from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_rag_service
from api.models import ErrorResponse, HealthResponse, QueryRequest, QueryResponse

from rag.orchestrator import RAGOrchestrator
from rag.models import Answer


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["RAG"])


@router.post(
        "/query", 
        response_model=QueryResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Invalid request"},
            401: {"model": ErrorResponse, "description": "Authentication required"},
            429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
            500: {"model": ErrorResponse, "description": "Internal server error"},
            503: {"model": ErrorResponse, "description": "Service unavailable"},
        },
        summary="Query equipment documentation",
        description="Submit a question about medical equipment and receive AI-generated answers with source references."

    )
async def query_documents(
    request: QueryRequest, 
    service:RAGOrchestrator=Depends(get_rag_service)
):
    """
    Process a document query and return AI-generated answer with sources.
    
    Args:
        request: Query request with question and metadata
        rag_service: RAG orchestrator service instance
    
    Returns:
        QueryResponse: Generated answer with source documents
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()

    logger.info(
        f"Processing query request",
        extra={
            "request_id": request.request_id,
            "question_length": len(request.question),
        }
    )
    try:
        # Process the query
        # NOTE: There is no validation or authentication yet.
        # TODO: Add API key authentication and input validation.
        result: Answer = service.invoke(query=request.question)

        processing_time = int((time.time() - start_time) * 1000)
        response = QueryResponse(
            request_id=request.request_id, 
            answer=result.answer_text,
            sources=result.source_chunks
        )
        logger.info(
            f"Query processed successfully",
            extra={
                "request_id": request.request_id,
                "processing_time_ms": processing_time,
                "sources_count": len(result.source_chunks),
                "answer_length": len(result.answer_text)
            }
        )
        return response
    except ValueError as e:
        logger.warning(
            f"Validation error in query processing",
            extra={
                "request_id": request.request_id,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error_code="VALIDATION_ERROR",
                message="Invalid input parameters",
                request_id=request.request_id
            ).model_dump()
        )
        
    except TimeoutError as e:
        logger.error(
            f"Query processing timeout",
            extra={
                "request_id": request.request_id,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
        )
        raise HTTPException(
            status_code=504,
            detail=ErrorResponse(
                error_code="PROCESSING_TIMEOUT",
                message="Query processing timed out",
                request_id=request.request_id
            ).model_dump()
        )
        
    except Exception as e:
        logger.error(
            f"Unexpected error in query processing",
            extra={
                "request_id": request.request_id,
                "error_type": type(e).__name__,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message="An unexpected error occurred",
                request_id=request.request_id
            ).model_dump()
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Check the health status of the RAG service and its components."
)
async def health_check(
    rag_service: RAGOrchestrator = Depends(get_rag_service)
) -> HealthResponse:
    """
    Perform health check of the RAG service.
    
    Args:
        rag_service: RAG orchestrator service instance
        
    Returns:
        HealthResponse: Service health status
    """
    try:
        # Check component health
        components = await rag_service.health_check()
        
        overall_status = "healthy" if all(
            comp.get("status") == "healthy" for comp in components.values()
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            components={"error": {"status": "unhealthy", "message": "Health check failed"}}
        )
