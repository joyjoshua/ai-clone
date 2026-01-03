"""
Error Handling Middleware
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from app.schemas.errors import ErrorResponse, ErrorDetail
from app.utils.logger import logger
import traceback


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error.get("loc", []))
        errors.append({
            "field": field,
            "message": error.get("msg"),
            "type": error.get("type")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request validation failed",
                details={"errors": errors}
            )
        ).model_dump()
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred",
                details={"type": type(exc).__name__}
            )
        ).model_dump()
    )


async def http_exception_handler(request: Request, exc):
    """Handle HTTP exceptions"""
    from fastapi import HTTPException as FastAPIHTTPException
    
    if isinstance(exc, FastAPIHTTPException):
        # Handle FastAPI HTTPException
        if isinstance(exc.detail, dict):
            code = exc.detail.get("code", f"HTTP_{exc.status_code}")
            message = exc.detail.get("message", str(exc.detail))
            details = exc.detail.get("details")
        else:
            code = f"HTTP_{exc.status_code}"
            message = str(exc.detail)
            details = None
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=code,
                    message=message,
                    details=details
                )
            ).model_dump()
        )
    
    # Fallback for other HTTP exceptions
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content=ErrorResponse(
            error=ErrorDetail(
                code="HTTP_ERROR",
                message=str(exc.detail) if hasattr(exc, "detail") else "HTTP error occurred"
            )
        ).model_dump()
    )

