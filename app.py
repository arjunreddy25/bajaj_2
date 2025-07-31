import os
import json
import asyncio
import time
import requests
import tempfile
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import lru_cache

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration & Initialization ---
PINECONE_INDEX_NAME = "bajaj-hackathon-index"
API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")  # Set your API key

# Initialize FastAPI app
app = FastAPI(
    title="Bajaj Hackathon RAG API",
    description="Insurance Policy Question Answering System",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    documents: str = None  # Optional document URL
    questions: List[str]

class QuestionResponse(BaseModel):
    answers: List[str]

# --- Authentication Function ---
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API key from the Authorization header."""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return credentials.credentials

# --- RAG System Class ---
class OptimizedRAGSystem:
    def __init__(self):
        """Initialize the RAG system with optimized configurations."""
        # Use faster model with optimized settings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=2048,
            timeout=15
        )
        
        # Initialize embedding model
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize vector store
        self.vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=self.embedding_model
        )
        
        # Cache for retrieved documents
        self.document_cache = {}
        
        # Pre-compiled prompts
        self.setup_prompts()
    
    def setup_prompts(self):
        """Pre-compile prompts for better performance."""
        self.unified_prompt = ChatPromptTemplate.from_template(
            """You are an insurance policy expert. Answer the question directly based on the policy clauses provided.

Question: {query}

Policy Clauses:
{context}

Provide a direct, concise answer in 1-2 sentences. Include specific numbers, percentages, timeframes, and conditions mentioned in the clauses. Mention clause numbers if available.""")
    
    @lru_cache(maxsize=50)
    def get_cached_documents(self, query_hash: str, query: str) -> List[str]:
        """Cache retrieved documents to avoid redundant API calls."""
        if query_hash in self.document_cache:
            return self.document_cache[query_hash]
        
        # Retrieve documents with reduced count for speed
        retrieved_docs = self.vectorstore.similarity_search(query, k=3)
        doc_texts = [doc.page_content for doc in retrieved_docs]
        
        # Cache the results
        self.document_cache[query_hash] = doc_texts
        return doc_texts
    
    def process_single_query(self, query: str) -> str:
        """Process a single query efficiently and return direct answer."""
        try:
            # Create a hash for caching
            query_hash = str(hash(query.lower().strip()))
            
            # Get relevant documents (cached if available)
            relevant_docs = self.get_cached_documents(query_hash, query)
            context = "\n\n".join(relevant_docs)
            
            # Single LLM call for direct answer
            result = self.llm.invoke(
                self.unified_prompt.format_messages(
                    query=query,
                    context=context
                )
            )
            
            # Extract clean answer
            answer = result.content.strip()
            return answer
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    def process_multiple_queries_parallel(self, queries: List[str]) -> List[str]:
        """Process multiple queries in parallel for optimal performance."""
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(len(queries), 8)) as executor:
            # Submit all queries for parallel processing and maintain order
            futures = [executor.submit(self.process_single_query, query) for query in queries]
            
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=20)
                    results.append(result)
                    print(f"Completed query {i+1}/{len(queries)}")
                except Exception as e:
                    print(f"Error in query {i+1}: {str(e)}")
                    results.append(f"Error processing query: {str(e)}")
        
        return results

# --- Document Processing Functions ---
def download_document(url: str) -> str:
    """Download document from URL and save to temporary file."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

def process_document(file_path: str):
    """Process the downloaded document and add to vector store."""
    try:
        # Load PDF document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunked_docs = text_splitter.split_documents(documents)
        
        # Initialize embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Add to vector store
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunked_docs,
            embedding=embedding_model,
            index_name=PINECONE_INDEX_NAME
        )
        
        print(f"Successfully processed document with {len(chunked_docs)} chunks")
        return True
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)

# --- API Endpoints ---
@app.post("/hackrx/run", response_model=QuestionResponse)
async def run_questions(
    request: QuestionRequest
    # api_key: str = Depends(verify_api_key)  # Commented out for testing
):
    """
    Process insurance policy questions and return answers.
    
    This endpoint:
    1. Downloads the policy document from the provided URL
    2. Processes and indexes the document
    3. Answers all questions using the RAG system
    4. Returns answers in the required format
    """
    try:
        print(f"Processing request with {len(request.questions)} questions")
        start_time = time.time()
        
        # Step 1: Download and process document (if provided)
        if request.documents:
            print("Downloading document...")
            temp_file_path = download_document(request.documents)
            
            print("Processing document...")
            process_document(temp_file_path)
        else:
            print("No document provided, using existing indexed documents")
        
        # Step 2: Initialize RAG system
        print("Initializing RAG system...")
        rag_system = OptimizedRAGSystem()
        
        # Step 3: Process all questions
        print("Processing questions...")
        answers = rag_system.process_multiple_queries_parallel(request.questions)
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        # Return in required format
        return QuestionResponse(answers=answers)
        
    except Exception as e:
        print(f"Error in /hackrx/run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Bajaj Hackathon RAG API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Bajaj Hackathon RAG API",
        "version": "1.0.0",
        "endpoints": {
            "POST /hackrx/run": "Process insurance policy questions",
            "GET /health": "Health check",
            "GET /": "API information"
        }
    }

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY environment variable is required")
    
    # Run the FastAPI application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 