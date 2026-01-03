"""
Chat API Endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import List
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse, ConversationResponse
from app.services.crud import ConversationCRUD, UserCRUD
from app.utils.logger import logger

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/message", response_model=ChatMessageResponse)
async def send_message(request: ChatMessageRequest):
    """
    Send a message and get AI response
    
    Note: This endpoint will be fully implemented in Step 4 (RAG System Integration).
    For now, it returns a placeholder response.
    """
    try:
        # Ensure user exists
        user = await UserCRUD.get_or_create_user(request.user_id)
        
        # Get or create conversation
        conversation = None
        if request.conversation_id:
            conversation = await ConversationCRUD.get_conversation(request.conversation_id)
            if conversation and conversation.user_id != request.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Conversation does not belong to this user"
                )
        
        if not conversation:
            conversation = await ConversationCRUD.create_conversation(request.user_id)
        
        # Add user message to conversation
        await ConversationCRUD.add_message(
            str(conversation.id),
            "user",
            request.message
        )
        
        # TODO: In Step 4, implement RAG retrieval and LLM response generation
        # For now, return placeholder response
        placeholder_response = (
            f"I received your message: '{request.message}'. "
            "RAG system integration will be implemented in Step 4."
        )
        
        # Add assistant message to conversation
        await ConversationCRUD.add_message(
            str(conversation.id),
            "assistant",
            placeholder_response
        )
        
        return ChatMessageResponse(
            response=placeholder_response,
            conversation_id=str(conversation.id),
            sources=[],
            metadata={
                "model": "placeholder",
                "note": "RAG system will be implemented in Step 4"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(user_id: str, skip: int = 0, limit: int = 20):
    """Get all conversations for a user"""
    try:
        conversations = await ConversationCRUD.get_user_conversations(
            user_id=user_id,
            skip=skip,
            limit=limit
        )
        
        return [
            ConversationResponse(
                _id=str(conv.id),
                user_id=conv.user_id,
                messages=[msg.model_dump() for msg in conv.messages],
                created_at=conv.created_at,
                updated_at=conv.updated_at
            )
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching conversations: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, user_id: str):
    """Get a specific conversation"""
    try:
        conversation = await ConversationCRUD.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        if conversation.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Conversation does not belong to this user"
            )
        
        return ConversationResponse(
            _id=str(conversation.id),
            user_id=conversation.user_id,
            messages=[msg.model_dump() for msg in conversation.messages],
            created_at=conversation.created_at,
            updated_at=conversation.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching conversation: {str(e)}"
        )

