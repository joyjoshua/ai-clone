"""
Error Response Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class ErrorDetail(BaseModel):
    """Error detail schema"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response schema"""
    error: ErrorDetail

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input provided",
                    "details": {
                        "field": "user_id",
                        "reason": "Field is required"
                    }
                }
            }
        }

