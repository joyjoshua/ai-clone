# AI Clone Application

Build an AI Clone application that uses RAG (Retrieval-Augmented Generation) to create a personalized AI assistant that mimics your knowledge, communication style, and responses.

## Tech Stack

- **Frontend**: React + TypeScript + Vite
- **Backend**: FastAPI (Python 3.9+)
- **Database**: MongoDB (for structured data)
- **Vector Database**: ChromaDB (primary) or Qdrant (alternative)
- **RAG Framework**: Langchain
- **LLM**: Groq (fast, low-cost inference)
- **PDF Processing**: PyPDF2

## Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- Python 3.9+
- MongoDB (local or cloud)
- Groq API key ([Get one here](https://groq.com/))

## Project Structure

```
ai-clone/
├── frontend/          # React + TypeScript frontend
├── backend/           # FastAPI backend
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai-clone
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env

# Edit .env file with your configuration
# - Add your GROQ_API_KEY
# - Configure MongoDB URL
# - Set other preferences
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# or
yarn install
# or
pnpm install

# Copy environment variables
cp .env.example .env

# Edit .env file if needed (defaults should work)
```

### 4. Database Setup

**MongoDB:**
- Use MongoDB Atlas (cloud) - recommended
  - Sign up at https://www.mongodb.com/cloud/atlas
  - Create a free cluster
  - Get your connection string
  - Update `MONGODB_URL` in `backend/.env`
- Or install MongoDB locally if preferred

**ChromaDB:**
- No setup needed! ChromaDB uses local file storage
- The database will be created automatically in `./chroma_db_data` when first used

## Running the Application

### Start Backend

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn app.main:app --reload --port 8000
```

Backend will be available at: http://localhost:8000
API docs available at: http://localhost:8000/docs

### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend will be available at: http://localhost:5173

## Environment Variables

### Backend (.env)

See `backend/.env.example` for all available options. Key variables:

- `GROQ_API_KEY` - Your Groq API key (required)
- `MONGODB_URL` - MongoDB connection string
- `VECTOR_DB` - Choose "chromadb" or "qdrant"
- `CHROMADB_PATH` - Path for ChromaDB storage

### Frontend (.env)

See `frontend/.env.example` for options. Defaults should work for local development.

## Development

### Backend Development

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd frontend
npm run dev
```

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Features (Phase 1)

- ✅ Document upload (PDF, TXT, MD)
- ✅ RAG-based question answering
- ✅ Conversation history with memory
- ✅ Source citations
- ✅ Response evaluation
- ✅ Chat interface

## Next Steps

See `idea.md` for the complete implementation plan and Phase 2 features.

## Troubleshooting

### MongoDB Connection Issues
- Check connection string in `.env` (should be MongoDB Atlas URL)
- Verify MongoDB Atlas cluster is running and accessible
- Ensure IP whitelist includes your IP address (for MongoDB Atlas)
- Check username/password in connection string are correct

### ChromaDB Issues
- Ensure write permissions in the project directory
- Check `CHROMADB_PATH` in `.env`

### Groq API Issues
- Verify `GROQ_API_KEY` is set correctly
- Check API key is valid at https://console.groq.com/
- Ensure you have API credits/quota

### Frontend Connection Issues
- Verify backend is running on port 8000
- Check `VITE_API_BASE_URL` in frontend `.env`
- Check browser console for CORS errors

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

