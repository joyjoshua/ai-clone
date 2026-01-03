"""
Document API Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload"""
    document_id: str = Field(..., description="Created document ID")
    title: str = Field(..., description="Document title")
    file_type: Literal["pdf", "txt", "md"] = Field(..., description="File type")
    status: Literal["processing", "completed", "failed"] = Field(..., description="Processing status")
    chunks_created: int = Field(0, description="Number of chunks created")
    message: Optional[str] = Field(None, description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc123",
                "title": "My Document",
                "file_type": "pdf",
                "status": "completed",
                "chunks_created": 15,
                "message": "Document processed successfully"
            }
        }


class DocumentResponse(BaseModel):
    """Document response schema"""
    id: str = Field(..., alias="_id")
    user_id: str
    title: str
    file_type: str
    metadata: dict
    created_at: datetime
    chunk_ids: List[str]
    status: str

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "doc123",
                "user_id": "user123",
                "title": "My Document",
                "file_type": "pdf",
                "metadata": {"file_size": 1024},
                "created_at": "2024-01-01T12:00:00Z",
                "chunk_ids": ["chunk1", "chunk2"],
                "status": "completed"
            }
        }


class DocumentListResponse(BaseModel):
    """Paginated document list response"""
    documents: List[DocumentResponse]
    total: int
    page: int
    limit: int
    has_more: bool

