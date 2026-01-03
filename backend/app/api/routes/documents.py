"""
Document API Endpoints
"""
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from typing import Optional, List
from app.schemas.document import DocumentUploadResponse, DocumentResponse, DocumentListResponse
from app.models.document import Document
from app.services.crud import DocumentCRUD, UserCRUD
from app.utils.file_validation import validate_file, sanitize_filename
from app.utils.logger import logger
from app.config import settings

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    user_id: str = Form(..., description="User ID"),
    title: str = Form(..., description="Document title", min_length=1),
    file: Optional[UploadFile] = File(None, description="File to upload"),
    text: Optional[str] = Form(None, description="Text content (if no file)")
):
    """
    Upload a document (file or text)
    
    Either file or text must be provided.
    """
    try:
        # Ensure user exists
        await UserCRUD.get_or_create_user(user_id)
        
        # Validate input
        if not file and not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file' or 'text' must be provided"
            )
        
        if file and text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide either 'file' or 'text', not both"
            )
        
        # Process file upload
        if file:
            # Validate file
            file_type, error = validate_file(file)
            if error:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error
                )
            
            # Read file content
            content = await file.read()
            content_text = content.decode('utf-8', errors='ignore')
            
            # Sanitize filename
            filename = sanitize_filename(file.filename or "uploaded_file")
            
            # Create document
            document = Document(
                user_id=user_id,
                title=title,
                content=content_text,
                file_type=file_type,
                metadata={
                    "original_filename": filename,
                    "file_size": len(content),
                    "upload_method": "file"
                },
                status="processing"
            )
        
        # Process text upload
        else:
            if not text or not text.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Text content cannot be empty"
                )
            
            document = Document(
                user_id=user_id,
                title=title,
                content=text.strip(),
                file_type="txt",
                metadata={
                    "upload_method": "text",
                    "content_length": len(text)
                },
                status="processing"
            )
        
        # Save document to MongoDB
        document = await DocumentCRUD.create_document(document)
        
        # TODO: In Step 4, implement document processing:
        # - Chunk the document
        # - Generate embeddings
        # - Store in ChromaDB
        # - Update document status to "completed"
        
        # For now, mark as completed (will be processed in Step 4)
        document.status = "completed"
        await DocumentCRUD.update_document(str(document.id), {"status": "completed"})
        
        return DocumentUploadResponse(
            document_id=str(document.id),
            title=document.title,
            file_type=document.file_type,
            status=document.status,
            chunks_created=0,  # Will be updated in Step 4
            message="Document uploaded successfully. Processing will be implemented in Step 4."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    user_id: str,
    page: int = 1,
    limit: int = 20
):
    """List all documents for a user with pagination"""
    try:
        skip = (page - 1) * limit
        
        documents = await DocumentCRUD.get_user_documents(
            user_id=user_id,
            skip=skip,
            limit=limit
        )
        
        total = await DocumentCRUD.count_user_documents(user_id)
        has_more = (skip + limit) < total
        
        return DocumentListResponse(
            documents=[
                DocumentResponse(
                    _id=str(doc.id),
                    user_id=doc.user_id,
                    title=doc.title,
                    file_type=doc.file_type,
                    metadata=doc.metadata,
                    created_at=doc.created_at,
                    chunk_ids=doc.chunk_ids,
                    status=doc.status
                )
                for doc in documents
            ],
            total=total,
            page=page,
            limit=limit,
            has_more=has_more
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, user_id: str):
    """Get a specific document"""
    try:
        document = await DocumentCRUD.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        if document.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Document does not belong to this user"
            )
        
        return DocumentResponse(
            _id=str(document.id),
            user_id=document.user_id,
            title=document.title,
            file_type=document.file_type,
            metadata=document.metadata,
            created_at=document.created_at,
            chunk_ids=document.chunk_ids,
            status=document.status
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching document: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str, user_id: str):
    """Delete a document"""
    try:
        # Verify document exists and belongs to user
        document = await DocumentCRUD.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        if document.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Document does not belong to this user"
            )
        
        # TODO: In Step 4, also delete chunks from ChromaDB
        # For now, just delete from MongoDB
        deleted = await DocumentCRUD.delete_document(document_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document"
            )
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )

