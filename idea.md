# AI Clone Application - Detailed Implementation Plan

## Project Overview
Build an AI Clone application that uses RAG (Retrieval-Augmented Generation) to create a personalized AI assistant that mimics your knowledge, communication style, and responses. The system will learn from your data (conversations, documents, preferences) stored in MongoDB and use ChromaDB (or Qdrant) for vector storage with Langchain for orchestration.

---

## Tech Stack
- **Frontend**: React (TypeScript recommended)
- **Backend**: FastAPI (Python 3.9+)
- **Database**: MongoDB (for structured data)
- **Vector Database**: ChromaDB (primary, recommended) or Qdrant (alternative)
- **RAG Framework**: Langchain
- **LLM**: Groq (fast, low-cost inference via OpenAI-compatible API)
- **PDF Processing**: PyPDF2
- **Memory Management**: ConversationBufferMemory
- **Authentication**: JWT (optional for Phase 1)

---

## Phase 1: Foundation & RAG Integration

### Phase 1.1: Project Setup & Basic Infrastructure

#### 1.1.1 Project Structure
```
ai-clone/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat/
│   │   │   │   ├── ChatWindow.tsx
│   │   │   │   ├── MessageList.tsx
│   │   │   │   ├── MessageInput.tsx
│   │   │   │   └── MessageBubble.tsx
│   │   │   ├── DataUpload/
│   │   │   │   ├── FileUpload.tsx
│   │   │   │   └── TextInput.tsx
│   │   │   └── Layout/
│   │   │       ├── Header.tsx
│   │   │       └── Sidebar.tsx
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   └── websocket.ts (optional)
│   │   ├── hooks/
│   │   │   └── useChat.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   ├── utils/
│   │   │   └── constants.ts
│   │   ├── App.tsx
│   │   └── index.tsx
│   ├── package.json
│   ├── tsconfig.json
│   ├── .env.example
│   ├── .gitignore
│   └── vite.config.ts
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── conversation.py
│   │   │   └── document.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   ├── document.py
│   │   │   └── errors.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chat.py
│   │   │   │   ├── documents.py
│   │   │   │   └── health.py
│   │   │   └── middleware.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── database.py
│   │   │   ├── chromadb_service.py
│   │   │   ├── qdrant_service.py (optional, if using Qdrant)
│   │   │   ├── rag_service.py
│   │   │   ├── llm_service.py
│   │   │   └── memory_service.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── text_processing.py
│   │       ├── file_validation.py
│   │       └── logger.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   └── Dockerfile (optional)
│
├── docker-compose.yml (for MongoDB & Qdrant)
├── .gitignore
└── README.md
```

#### 1.1.2 Frontend Setup (React)
**Dependencies:**
```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.1",
    "axios": "^1.7.7",
    "@tanstack/react-query": "^5.56.2",
    "zustand": "^4.5.5",
    "react-markdown": "^9.0.1",
    "react-syntax-highlighter": "^15.5.0",
    "@types/react-syntax-highlighter": "^15.5.13"
  },
  "devDependencies": {
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "typescript": "^5.6.3",
    "vite": "^5.4.8",
    "@vitejs/plugin-react": "^4.3.1"
  }
}
```

**Key Files to Create:**
1. `frontend/src/services/api.ts` - Axios instance with base URL and interceptors
2. `frontend/src/types/index.ts` - TypeScript interfaces for API responses
3. `frontend/src/components/Chat/ChatWindow.tsx` - Main chat interface
4. `frontend/src/components/Chat/MessageInput.tsx` - Input component
5. `frontend/src/components/Chat/MessageBubble.tsx` - Individual message display
6. `frontend/src/components/DataUpload/FileUpload.tsx` - File upload component with validation
7. `frontend/src/components/DataUpload/TextInput.tsx` - Text input for manual entry
8. `frontend/.env.example` - Environment variable template
9. `frontend/.gitignore` - Node.js gitignore patterns

#### 1.1.3 Backend Setup (FastAPI)
**Dependencies (requirements.txt):**
```
# Web Framework
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12

# Database
pymongo==4.10.1
motor==3.6.0

# Configuration & Validation
python-dotenv==1.0.1
pydantic==2.9.2
pydantic-settings==2.5.2

# Langchain & RAG
langchain==0.3.7
langchain-community==0.3.7
langchain-text-splitters==0.3.3
langchain-openai==0.2.8
langchain-groq==0.1.3

# Vector Database (ChromaDB - Primary)
chromadb==0.5.0
# Vector Database (Qdrant - Alternative)
qdrant-client==1.12.0

# Embeddings
sentence-transformers==3.3.1
langchain-huggingface==0.1.0
openai==1.54.5

# File Processing
PyPDF2==3.0.1
python-magic==0.4.27

# Memory & Evaluation
scikit-learn==1.5.2

# Utilities
httpx==0.27.2
aiofiles==24.1.0
```

**Key Files to Create:**
1. `backend/app/main.py` - FastAPI app initialization with middleware
2. `backend/app/config.py` - Configuration management with Pydantic Settings
3. `backend/app/services/database.py` - MongoDB connection (async with motor)
4. `backend/app/models/` - Pydantic models for MongoDB documents
5. `backend/app/utils/logger.py` - Logging configuration
6. `backend/app/utils/file_validation.py` - File upload validation
7. `backend/app/api/middleware.py` - Error handling and request middleware
8. `backend/app/schemas/errors.py` - Standardized error response schemas
9. `backend/.env.example` - Environment variable template
10. `backend/.gitignore` - Python gitignore patterns

#### 1.1.4 MongoDB Setup
**Database Schema:**
- **users** collection:
  ```python
  {
    "_id": ObjectId,
    "user_id": str,
    "created_at": datetime,
    "preferences": dict
  }
  ```

- **conversations** collection:
  ```python
  {
    "_id": ObjectId,
    "user_id": str,
    "messages": [
      {
        "role": "user" | "assistant",
        "content": str,
        "timestamp": datetime
      }
    ],
    "created_at": datetime,
    "updated_at": datetime
  }
  ```

- **documents** collection:
  ```python
  {
    "_id": ObjectId,
    "user_id": str,
    "title": str,
    "content": str,
    "file_type": str,  # "text", "pdf", "markdown", etc.
    "metadata": dict,
    "created_at": datetime,
    "chunk_ids": list[str]  # Vector database point/chunk IDs
  }
  ```

**Connection Setup:**
- Use `motor` (async MongoDB driver)
- Connection string from environment variables
- Database name: `ai_clone_db`
- Implement connection pooling
- Handle connection errors gracefully
- Add indexes for frequently queried fields (user_id, created_at)

**Indexes to Create:**
```python
# conversations collection
db.conversations.create_index([("user_id", 1), ("created_at", -1)])

# documents collection
db.documents.create_index([("user_id", 1), ("created_at", -1)])
db.documents.create_index([("title", "text")])  # Text search index
```

---

### Phase 1.2: Frontend-Backend Integration

#### 1.2.1 API Endpoints Design

**Health Check:**
- `GET /health` - Service health check

**Chat Endpoints:**
- `POST /api/chat/message` - Send message and get AI response
  ```json
  Request: {
    "message": "string",
    "conversation_id": "string" (optional),
    "user_id": "string"
  }
  Response: {
    "response": "string",
    "conversation_id": "string",
    "sources": [{"text": "string", "score": float}]
  }
  ```

- `GET /api/chat/conversations` - Get all conversations
- `GET /api/chat/conversations/{conversation_id}` - Get specific conversation

**Document Endpoints:**
- `POST /api/documents/upload` - Upload document (text/file)
  ```json
  Request: FormData {
    "file": File (optional, max 10MB),
    "text": string (optional),
    "title": string (required),
    "user_id": string (required)
  }
  Response: {
    "document_id": "string",
    "title": "string",
    "file_type": "pdf" | "txt" | "md",
    "status": "processing" | "completed",
    "chunks_created": int
  }
  ```
  **File Validation:**
  - Allowed types: PDF, TXT, MD only
  - Max file size: 10MB
  - File type validation using magic bytes
  - Content sanitization before processing

- `GET /api/documents` - List all documents (with pagination)
  ```json
  Query Params: {
    "user_id": string (required),
    "page": int (default: 1),
    "limit": int (default: 20)
  }
  ```

- `GET /api/documents/{document_id}` - Get specific document
- `DELETE /api/documents/{document_id}` - Delete document (also removes vectors from ChromaDB/Qdrant)

#### 1.2.2 CORS Configuration
- Configure FastAPI CORS middleware
- Allow frontend origin (e.g., `http://localhost:5173`)
- Allow credentials if using authentication
- Configure allowed methods: GET, POST, DELETE
- Set appropriate headers

#### 1.2.3 Error Handling & Middleware
**Error Response Schema:**
```python
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {}  # Optional additional details
  }
}
```

**Middleware Components:**
- Global exception handler
- Request validation error handler
- Database connection error handler
- Rate limiting (optional for Phase 1)
- Request logging middleware
- Response time tracking

#### 1.2.4 Frontend API Service
- Create axios instance with base URL
- Implement request/response interceptors
- Error handling middleware with user-friendly messages
- Type-safe API calls using TypeScript
- Retry logic for failed requests
- Request timeout configuration (30s default)

---

### Phase 1.3: RAG System Implementation

#### 1.3.1 Vector Database Setup

**Option 1: ChromaDB (Recommended for Phase 1)**
- **Why ChromaDB**: Simpler setup, no Docker required, persistent storage, good for development
- **Configuration:**
  - Collection name: `ai_knowledge_base`
  - Vector size: 384 (for `all-MiniLM-L6-v2`)
  - Distance metric: Cosine
  - Storage: Persistent (local file system) or in-memory

**ChromaDB Service (`backend/app/services/chromadb_service.py`):**
```python
Key Functions:
- initialize_collection() - Create/get collection using PersistentClient
- store_embeddings() - Store document chunks with embeddings
- query_similar() - Search for similar vectors (returns documents + distances)
- delete_chunks() - Remove chunks by IDs
- get_collection_stats() - Get collection metadata

Implementation:
from chromadb import PersistentClient
chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)
```

**Option 2: Qdrant (Alternative)**
- **Why Qdrant**: Better for production, cloud support, advanced features
- **Configuration:**
  - Collection name: `ai_clone_vectors`
  - Vector size: 384 (for `all-MiniLM-L6-v2`) or 1536 (for OpenAI embeddings)
  - Distance metric: Cosine

**Qdrant Service (`backend/app/services/qdrant_service.py`):**
```python
Key Functions:
- initialize_collection() - Create collection if not exists
- upsert_vectors() - Store document chunks as vectors
- search_similar() - Search for similar vectors
- delete_vectors() - Remove vectors by IDs
```

#### 1.3.2 Groq LLM Integration

**Getting Started with Groq:**
1. Sign up at https://groq.com/ and get your free API key
2. Groq offers OpenAI-compatible API, making integration seamless
3. Benefits: Ultra-fast inference, low latency, cost-effective pricing
4. Supports streaming responses for real-time chat experience

**Available Groq Models:**
- `llama-3.1-70b-versatile` - Best for general purpose, high quality responses (recommended)
- `llama-3.1-8b-instant` - Faster, lighter model for quick responses (good for testing)
- `openai/gpt-oss-20b` - Alternative model option
- `mixtral-8x7b-32768` - Good for longer context (32k tokens)
- `gemma2-9b-it` - Faster, lighter model for quick responses

**Integration Methods:**
```python
# Method 1: Using langchain-groq (Recommended)
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Method 2: Using OpenAI-compatible API via langchain-openai
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="llama-3.1-70b-versatile",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
```

#### 1.3.3 Langchain Integration

**RAG Service (`backend/app/services/rag_service.py`):**
```python
Components:
1. Document Loader
   - Use langchain_text_splitters.RecursiveCharacterTextSplitter
   - Chunk size: 500 characters (optimal for Phase 1, based on reference)
   - Chunk overlap: 100 characters (20% overlap for context)
   - Separators: ["\n\n", "\n", " ", ""] (respect document structure)
   - PDF parsing: Use PyPDF2.PdfReader for PDF files
   - Text cleaning: Remove extra whitespace, normalize encoding
   - Handle empty chunks gracefully

2. Embedding Model
   - Option 1: Sentence Transformers (all-MiniLM-L6-v2) - RECOMMENDED
     * Use langchain_huggingface.HuggingFaceEmbeddings
     * Free, local, no API costs
     * Vector size: 384 dimensions
     * Fast inference
     * Model: "sentence-transformers/all-MiniLM-L6-v2"
   - Option 2: OpenAI embeddings (text-embedding-3-small)
     * Requires OPENAI_API_KEY
     * Vector size: 1536 dimensions
     * Better quality but costs money

3. Vector Store
   - Option 1: ChromaDB (Recommended)
     * Use chromadb.PersistentClient for persistent storage
     * Store metadata: document_id, user_id, chunk_index, title, file_type
     * Collection: "ai_knowledge_base"
   - Option 2: Qdrant (Alternative)
     * Use langchain_community.vectorstores.Qdrant
     * Store metadata: document_id, user_id, chunk_index, title, file_type
     * Collection name from environment variable
   - Initialize collection on first use
   - Check for existing documents before adding (avoid duplicates)

4. Retrieval Chain
   - Use Langchain's RetrievalQA chain or ConversationalRetrievalChain
   - Top-k retrieval: 3 chunks (adjustable, start conservative)
   - Score threshold: 0.7 (filter low-quality matches)
   - Return source documents with scores
   - Use MMR (Maximal Marginal Relevance) for diversity (optional)
   - Handle empty retrieval results gracefully

5. LLM Integration
   - Groq via langchain-groq.ChatGroq
   - Recommended model: llama-3.1-70b-versatile (or llama-3.1-8b-instant for testing)
   - Temperature: 0.7 (adjustable per use case)
   - Max tokens: 2048 (adjustable)
   - System prompt: "You are an AI clone of [user]. Respond in their style and 
     use the provided context to answer questions accurately. If the context 
     doesn't contain relevant information, say so."
   - Include conversation history in context (last 3-5 messages)
   - Handle API errors gracefully with fallback responses
```

**LLM Service (`backend/app/services/llm_service.py`):**
```python
Key Functions:
- generate_response() - Generate AI response using Groq
- format_prompt() - Format user query with context
- stream_response() - Stream response for real-time UI (Groq supports streaming)
- initialize_groq_client() - Initialize Groq client with API key

Implementation Options:
Option 1: Using langchain-groq (Recommended)
  from langchain_groq import ChatGroq
  
Option 2: Using OpenAI-compatible API
  from langchain_openai import ChatOpenAI
  # Set base_url="https://api.groq.com/openai/v1"
  
Groq Configuration:
- Base URL: https://api.groq.com/openai/v1
- API Key: From GROQ_API_KEY environment variable
- Models: llama-3.1-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it
- Temperature: 0.7 (adjustable)
- Max tokens: 2048 (adjustable)
```

#### 1.3.4 Memory Management

**Memory Service (`backend/app/services/memory_service.py`):**
```python
Key Functions:
- initialize_memory() - Create ConversationBufferMemory instance
- get_recent_chat_history(n=8) - Retrieve last N interactions
- get_memory_usage() - Get number of stored interactions
- save_context() - Save user input and AI response
- clear_memory() - Clear conversation history (optional)

Implementation:
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Get recent history
def get_recent_chat_history(n=8):
    past_chat_history = memory.load_memory_variables({}).get("chat_history", [])
    return past_chat_history[-n:] if past_chat_history else []

# Save context
memory.save_context(
    {"input": user_query},
    {"output": ai_response}
)
```

**Memory Considerations:**
- Keep last 8-10 interactions for context
- Trim older messages to prevent token limit issues
- Store memory per conversation_id
- Clear memory on conversation reset

#### 1.3.5 Response Evaluation

**Evaluation Service (`backend/app/services/evaluation_service.py`):**
```python
Key Functions:
- evaluate_response() - Calculate semantic similarity between response and context
- calculate_relevance_score() - Score how relevant response is to retrieved context
- get_evaluation_metrics() - Return evaluation metrics

Implementation:
from sentence_transformers import SentenceTransformer, util

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_response(user_query, generated_response, context):
    """Evaluate response quality using semantic similarity."""
    try:
        response_embedding = semantic_model.encode(generated_response, convert_to_tensor=True)
        context_embedding = semantic_model.encode(context, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()
        return similarity_score
    except Exception as e:
        return 0.0  # Return default score on error
```

**Evaluation Metrics:**
- Semantic similarity score (0.0 to 1.0)
- Context relevance score
- Response quality indicators
- Log evaluation scores for monitoring

#### 1.3.6 Document Processing Pipeline

**Flow:**
1. User uploads document/text via frontend
2. Backend receives and stores in MongoDB
3. Document is split into chunks (500 chars, 100 overlap)
4. Check for existing chunks to avoid duplicates
5. Each new chunk is embedded using embedding model
6. Embeddings stored in ChromaDB/Qdrant with metadata
7. Document record updated with chunk IDs
8. Return processing status to frontend

**Text Processing (`backend/app/utils/text_processing.py`):**
- Clean and normalize text (remove extra whitespace, normalize encoding)
- Split into semantic chunks using RecursiveCharacterTextSplitter
  - Chunk size: 500 characters
  - Chunk overlap: 100 characters
- Extract metadata (title, date, file type, etc.)
- Handle special characters and encoding issues
- Preserve document structure where possible
- Filter out empty chunks
- PDF extraction using PyPDF2.PdfReader

**PDF Processing:**
```python
from PyPDF2 import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    return text
```

**File Validation (`backend/app/utils/file_validation.py`):**
- Validate file type using python-magic (magic bytes, not extension)
- Check file size (max 10MB)
- Sanitize filenames
- Validate file content (not empty, readable)
- Return clear error messages for invalid files
- Support PDF, TXT, MD file types

---

### Phase 1.4: Chat Flow Implementation

#### 1.4.1 Request Flow
```
User Input → Frontend → POST /api/chat/message
                              ↓
                    Backend receives message
                              ↓
                    Get recent chat history from Memory Service
                              ↓
                    Save user message to MongoDB
                              ↓
                    RAG Service: Retrieve relevant chunks from ChromaDB/Qdrant
                              ↓
                    Format prompt with context + chat history
                              ↓
                    LLM Service: Generate response using Groq
                              ↓
                    Evaluation Service: Evaluate response quality
                              ↓
                    Save conversation to MongoDB
                              ↓
                    Save context to Memory Service
                              ↓
                    Return response + evaluation metrics to frontend
                              ↓
                    Display in chat UI with source citations
```

#### 1.4.2 Response Format
```json
{
  "response": "AI generated response text",
  "conversation_id": "string",
  "sources": [
    {
      "document_id": "string",
      "title": "string",
      "text": "relevant chunk text",
      "score": 0.85,
      "chunk_index": 2
    }
  ],
  "metadata": {
    "model": "llama-3.1-70b-versatile",
    "tokens_used": 150,
    "response_time_ms": 450,
    "memory_usage": 5,
    "evaluation_score": 0.85
  }
}
```

**Features:**
- Include source citations (which documents were used)
- Show confidence scores for each source
- Include metadata about the response (model, tokens, timing)
- Include memory usage (number of past interactions)
- Include evaluation score (semantic similarity to context)
- Note: Streaming responses deferred to Phase 2 for simplicity

---

## Implementation Checklist

### Step 1: Project Initialization
- [ ] Create project directory structure
- [ ] Initialize React app (Vite + TypeScript)
- [ ] Initialize FastAPI backend
- [ ] Set up virtual environment for Python
- [ ] Create `requirements.txt` and `package.json` with correct versions
- [ ] Create `.env.example` files for both frontend and backend
- [ ] Create `.gitignore` files (Python and Node.js patterns)
- [ ] Create `README.md` with setup instructions
- [ ] Configure `.env` files locally (not committed to git)

### Step 2: Database Setup
- [ ] Install and configure MongoDB (local or cloud)
- [ ] Create database connection service
- [ ] Define Pydantic models for MongoDB documents
- [ ] Test database connection
- [ ] Set up ChromaDB (no Docker needed, persistent storage)
- [ ] Create ChromaDB collection and test connection
- [ ] (Optional) Set up Qdrant (Docker or cloud) as alternative
- [ ] Create vector database collection
- [ ] Test vector database connection

### Step 3: Backend API Development
- [ ] Create FastAPI app with CORS configuration
- [ ] Set up logging configuration (structured logging)
- [ ] Implement health check endpoint (check MongoDB and ChromaDB/Qdrant connections)
- [ ] Create chat endpoints (POST /api/chat/message)
- [ ] Create document endpoints (POST /api/documents/upload with validation)
- [ ] Implement MongoDB CRUD operations (async with motor)
- [ ] Add request validation with Pydantic schemas
- [ ] Add error handling middleware and exception handlers
- [ ] Add file validation utility (type, size, content)
- [ ] Set up API documentation (FastAPI auto-generates Swagger/OpenAPI)

### Step 4: RAG System Integration
- [ ] Install Langchain packages (langchain, langchain-community, langchain-text-splitters, langchain-huggingface)
- [ ] Install ChromaDB and test connection
- [ ] Set up embedding model (Sentence Transformers - HuggingFaceEmbeddings)
- [ ] Create ChromaDB service for vector operations (initialize collection)
- [ ] Implement document chunking logic (500 chars, 100 overlap)
- [ ] Add PDF parsing support (PyPDF2)
- [ ] Create RAG service with retrieval chain
- [ ] Get Groq API key from https://groq.com/
- [ ] Integrate Groq LLM (via langchain-groq - recommended)
- [ ] Configure Groq model (llama-3.1-70b-versatile or llama-3.1-8b-instant)
- [ ] Implement ConversationBufferMemory for chat history
- [ ] Create memory service with get_recent_chat_history()
- [ ] Implement response evaluation using semantic similarity
- [ ] Add fallback patterns for missing dependencies
- [ ] Test RAG pipeline end-to-end with sample documents

### Step 5: Frontend Development
- [ ] Set up React app with routing (react-router-dom)
- [ ] Create API service layer with axios and error handling
- [ ] Build ChatWindow component with message list
- [ ] Build MessageInput component with send button
- [ ] Build MessageBubble component (user/assistant styling)
- [ ] Build FileUpload component with drag-and-drop support
- [ ] Add file type validation on frontend (PDF, TXT, MD)
- [ ] Add file size validation (max 10MB)
- [ ] Implement chat state management (Zustand or React Query)
- [ ] Add loading states and error handling
- [ ] Add error boundaries for React error handling
- [ ] Style components (CSS/Tailwind) - modern, responsive design
- [ ] Add source citations display in chat UI

### Step 6: Integration & Testing
- [ ] Connect frontend to backend API
- [ ] Test document upload flow
- [ ] Test chat message flow
- [ ] Verify RAG retrieval works
- [ ] Test end-to-end user journey
- [ ] Fix any CORS or connection issues
- [ ] Add error boundaries in React

### Step 7: Polish & Optimization
- [ ] Add loading indicators for all async operations
- [ ] Improve error messages (user-friendly, actionable)
- [ ] Optimize chunk sizes based on testing (start with 500/100)
- [ ] Tune retrieval parameters (top-k, score threshold)
- [ ] Add source citations in UI (clickable, shows document title)
- [ ] Display evaluation scores in UI (optional)
- [ ] Performance testing (response times, concurrent requests)
- [ ] Add request timeout handling
- [ ] Optimize MongoDB queries (add indexes if needed)
- [ ] Optimize memory usage (trim old chat history)
- [ ] Add logging for debugging and monitoring
- [ ] Test fallback patterns for production resilience

---

## Environment Variables

### Backend (.env.example)
```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=ai_clone_db

# Vector Database Configuration
# Option 1: ChromaDB (Recommended - no Docker needed)
VECTOR_DB=chromadb
CHROMADB_PATH=./chroma_db
CHROMADB_COLLECTION_NAME=ai_knowledge_base

# Option 2: Qdrant (Alternative)
# VECTOR_DB=qdrant
# QDRANT_URL=http://localhost:6333
# QDRANT_COLLECTION_NAME=ai_clone_vectors

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile
# Alternative models: mixtral-8x7b-32768, gemma2-9b-it

# Embedding Model Configuration
# Option 1: Sentence Transformers (free, recommended)
EMBEDDING_MODEL=sentence-transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# Option 2: OpenAI Embeddings (requires API key)
# EMBEDDING_MODEL=openai
# OPENAI_API_KEY=your_openai_api_key_here
# EMBEDDING_MODEL_NAME=text-embedding-3-small

# Server Configuration
BACKEND_PORT=8000
CORS_ORIGINS=http://localhost:5173

# File Upload Configuration
MAX_FILE_SIZE_MB=10
ALLOWED_FILE_TYPES=pdf,txt,md

# RAG Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K_RETRIEVAL=3
SCORE_THRESHOLD=0.7
MEMORY_HISTORY_LENGTH=8

# Logging
LOG_LEVEL=INFO
```

### Frontend (.env.example)
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=30000
```

---

## Docker Compose (Optional)

**Note**: ChromaDB doesn't require Docker - it uses local file storage. Only use Docker if you want Qdrant or MongoDB in containers.

```yaml
version: '3.8'
services:
  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  # Optional: Only if using Qdrant instead of ChromaDB
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  mongodb_data:
  qdrant_data:
```

---

## Logging Configuration

### Backend Logging (`backend/app/utils/logger.py`)
```python
# Use Python's logging module with structured format
# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
# Log format: timestamp, level, module, message, extra context
# Log to both console and file (optional)
```

**What to Log:**
- API requests (method, path, status, duration)
- Errors with full stack traces
- Document uploads (file type, size, user_id)
- RAG queries (query text, chunks retrieved, response time)
- Database operations (queries, errors)
- External API calls (Groq API usage, errors)

**What NOT to Log:**
- API keys or secrets
- Full user messages (log summaries or hashes)
- Full document content (log metadata only)

## Testing Strategy

### Unit Tests
- Test document chunking logic
- Test embedding generation
- Test vector search
- Test file validation
- Test API endpoints (with mocked dependencies)
- Test error handling

### Integration Tests
- Test MongoDB operations (CRUD)
- Test ChromaDB/Qdrant operations (vector storage/retrieval)
- Test memory management (ConversationBufferMemory)
- Test response evaluation (semantic similarity)
- Test RAG pipeline end-to-end
- Test full chat flow (with test documents)
- Test file upload with various file types (PDF, TXT, MD)
- Test fallback patterns for missing dependencies

### Manual Testing
- Upload various document types (PDF, TXT, MD)
- Test chat with different queries
- Verify context retrieval accuracy
- Check response quality and relevance
- Test error scenarios (invalid files, network failures)
- Test concurrent requests

---

## Next Steps After Phase 1
- Phase 2: Advanced features (fine-tuning, multi-modal, voice input)
- User authentication and multi-user support
- Conversation history and context management
- Advanced RAG techniques (reranking, hybrid search)
- Analytics and monitoring
- Deployment and scaling

---

## Fallback Patterns & Error Resilience

Based on the reference implementation, implement fallback patterns for missing dependencies:

**Import Fallbacks:**
```python
# Example: Text Splitter with fallback
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback: Simple text splitter implementation
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=100):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        def split_text(self, text):
            # Simple chunking logic
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
            return chunks
```

**Service Fallbacks:**
- If ChromaDB fails, log error and return empty results
- If Groq API fails, return error message to user
- If embedding model fails, use mock embeddings for testing
- If memory fails, continue without conversation history

**Benefits:**
- System continues to function with degraded capabilities
- Easier development and testing
- Better error messages for debugging
- Graceful degradation in production

## Key Considerations

1. **Data Privacy**: Ensure user data is properly secured
   - Never commit API keys to git
   - Use environment variables for all secrets
   - Validate and sanitize all user inputs
   - Implement proper file upload security

2. **Cost Management**: Monitor API usage
   - Groq offers cost-effective inference, but monitor usage
   - Use Sentence Transformers for embeddings (free) in Phase 1
   - Set reasonable limits on file sizes and requests

3. **Performance**: 
   - Groq provides ultra-fast inference
   - Optimize chunk sizes for your use case
   - Use async/await patterns throughout backend
   - Consider caching for frequently accessed data

4. **Security**:
   - Validate file types using magic bytes, not extensions
   - Implement file size limits (10MB default)
   - Sanitize user inputs before processing
   - Use parameterized queries for database operations
   - Implement rate limiting (consider for Phase 2)

5. **Error Handling**: 
   - Robust error handling at every layer
   - User-friendly error messages
   - Log errors with context for debugging
   - Handle edge cases (empty files, network failures, etc.)

6. **Logging**: 
   - Implement structured logging
   - Log important events (uploads, queries, errors)
   - Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
   - Don't log sensitive data (API keys, user content)

7. **Scalability**: 
   - Design for horizontal scaling
   - Use async database operations
   - Consider connection pooling
   - Optimize database queries and indexes

8. **Documentation**: 
   - Keep API documentation updated (FastAPI auto-generates)
   - Document environment variables
   - Add code comments for complex logic
   - Maintain README with setup instructions

---

## Additional Files to Create

### Backend `.gitignore`
```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv
*.egg-info/
dist/
build/
.env
*.log
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
```

### Frontend `.gitignore`
```
node_modules/
dist/
.env
.env.local
*.log
.DS_Store
coverage/
.vite/
```

### README.md Structure
- Project overview
- Tech stack
- Prerequisites
- Installation instructions
- Environment setup
- Running the application
- API documentation link
- Troubleshooting

## Resources & Documentation
- FastAPI: https://fastapi.tiangolo.com/
- Langchain: https://python.langchain.com/
- Langchain Community: https://python.langchain.com/docs/integrations/vectorstores/
- Langchain HuggingFace: https://python.langchain.com/docs/integrations/text_embedding/huggingface
- ChromaDB: https://docs.trychroma.com/
- Qdrant: https://qdrant.tech/documentation/
- React: https://react.dev/
- MongoDB: https://www.mongodb.com/docs/
- Motor (Async MongoDB): https://motor.readthedocs.io/
- Groq: https://groq.com/
- Groq API Docs: https://console.groq.com/docs
- Groq Python SDK: https://github.com/groq/groq-python
- Langchain Groq Integration: https://python.langchain.com/docs/integrations/chat/groq
- Sentence Transformers: https://www.sbert.net/
- PyPDF2: https://pypdf2.readthedocs.io/
- Langchain Memory: https://python.langchain.com/docs/modules/memory/

