from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os
from datetime import datetime
from catalog import CarCatalogRAG

app = FastAPI(
    title="Car Catalog RAG API",
    description="RAG-based question answering system for Nissan Altima owner manual",
    version="1.0.0"
)

# Global RAG instance
rag_system = None

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int = 0

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    
    # Get API keys from environment variables
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not mistral_api_key or not openai_api_key:
        raise ValueError("MISTRAL_API_KEY and OPENAI_API_KEY environment variables must be set")
    
    # Initialize RAG system
    rag_system = CarCatalogRAG(
        mistral_api_key=mistral_api_key,
        openai_api_key=openai_api_key,
        persist_directory="./chroma_db"
    )
    
    # Try to load existing documents
    try:
        rag_system.load_documents("car_documents.json")
        rag_system.setup_vectorstore()
        rag_system.setup_rag_chain()
        print("RAG system initialized successfully with existing documents")
    except FileNotFoundError:
        print("No existing documents found. Please process PDF first using /process-pdf endpoint")
    except Exception as e:
        print(f"Error initializing RAG system: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global rag_system
    
    if rag_system is None:
        return HealthResponse(
            status="error",
            message="RAG system not initialized",
            documents_loaded=0
        )
    
    return HealthResponse(
        status="healthy",
        message="RAG system is running",
        documents_loaded=len(rag_system.all_documents) if rag_system.all_documents else 0
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the car manual
    """
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not rag_system.rag_chain:
        raise HTTPException(status_code=500, detail="RAG chain not setup. Process PDF first.")
    
    try:
        result = rag_system.ask_question(request.question)
        
        # Extract sources from context
        sources = []
        for ctx in result["context"]:
            sources.append({
                "page_number": ctx["metadata"].get("page_number"),
                "document_name": ctx["metadata"].get("document_name"),
                "content_preview": ctx["page_content"][:200] + "..." if len(ctx["page_content"]) > 200 else ctx["page_content"]
            })
        
        return QuestionResponse(
            answer=result["answer"],
            context=result["context"],
            sources=sources,
            metadata={
                "question": request.question,
                "sources_count": len(sources),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-large"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/process-pdf")
async def process_pdf(pdf_path: str = "2022-nissan-altima-owner-manual.pdf"):
    """
    Process a PDF file and create documents for RAG
    """
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")
    
    try:
        # Process PDF
        documents = rag_system.process_pdf_to_documents(pdf_path)
        
        # Save documents
        rag_system.save_documents("car_documents.json")
        
        # Setup vector store and RAG chain
        rag_system.setup_vectorstore()
        rag_system.setup_rag_chain()
        
        return {
            "message": f"Successfully processed PDF and created {len(documents)} documents",
            "documents_created": len(documents),
            "status": "ready"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/documents/info")
async def get_documents_info():
    """
    Get information about loaded documents
    """
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not rag_system.all_documents:
        return {"message": "No documents loaded", "count": 0}
    
    # Get document statistics
    total_pages = len(rag_system.all_documents)
    document_names = list(set(doc.metadata.get("document_name", "Unknown") for doc in rag_system.all_documents))
    
    return {
        "total_documents": total_pages,
        "document_names": document_names,
        "vectorstore_ready": rag_system.vectorstore is not None,
        "rag_chain_ready": rag_system.rag_chain is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
