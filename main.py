from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
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
    document_name: str  # Required: specific document/catalog to query

class QuestionResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}

class ProcessS3Request(BaseModel):
    s3_url: str
    document_name: str = "User Catalog"

class ProcessS3Response(BaseModel):
    message: str
    documents_created: int
    status: str
    s3_url: str

class HealthResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int = 0
    s3_enabled: bool = False

class MaintenanceReminderRequest(BaseModel):
    document_name: str
    current_mileage: int  # Current mileage in kilometers
    user_id: Optional[str] = None  # Optional user identifier

class ReminderData(BaseModel):
    type: str = "maintenance"
    dueDate: str
    message: str
    mileage: Optional[int] = None
    priority: str = "medium"
    category: str = "general"

class MaintenanceReminderResponse(BaseModel):
    success: bool
    reminders: List[ReminderData]
    document_name: str
    current_mileage: int
    generated_at: str
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    
    # Get API keys from environment variables
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not mistral_api_key or not openai_api_key:
        raise ValueError("MISTRAL_API_KEY and OPENAI_API_KEY environment variables must be set")
    
    # Get AWS credentials for S3 (optional)
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    # Initialize RAG system
    rag_system = CarCatalogRAG(
        mistral_api_key=mistral_api_key,
        openai_api_key=openai_api_key,
        persist_directory="./chroma_db",
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_region=aws_region
    )
    
    # Try to load existing documents
    try:
        rag_system.load_documents("car_documents.json")
        # Setup vector stores and RAG chains for all loaded documents
        for document_name in rag_system.documents_by_name.keys():
            try:
                rag_system.setup_vectorstore_for_document(document_name)
                rag_system.setup_rag_chain_for_document(document_name)
            except Exception as e:
                print(f"Error setting up vector store for {document_name}: {e}")
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
        documents_loaded=sum(len(docs) for docs in rag_system.documents_by_name.values()) if rag_system.documents_by_name else 0,
        s3_enabled=rag_system.s3_client is not None
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about a specific document/catalog
    """
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Use the new architecture - document_name is required
        result = rag_system.ask_question(
            question=request.question,
            document_name=request.document_name
        )
        
        # Extract sources from context
        sources = []
        for ctx in result["context"]:
            sources.append({
                "page_number": ctx["metadata"].get("page_number"),
                "document_name": ctx["metadata"].get("document_name"),
                "document_type": ctx["metadata"].get("document_type", "unknown"),
                "source": ctx["metadata"].get("source", "unknown"),
                "content_preview": ctx["page_content"][:200] + "..." if len(ctx["page_content"]) > 200 else ctx["page_content"]
            })
        
        return QuestionResponse(
            answer=result["answer"],
            context=result["context"],
            sources=sources,
            metadata={
                "question": request.question,
                "document_name": request.document_name,
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
    Process a local PDF file and create documents for RAG
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

@app.post("/process-s3", response_model=ProcessS3Response)
async def process_s3_pdf(request: ProcessS3Request):
    """
    Process a PDF from S3 URL and create documents for RAG
    """
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not rag_system.s3_client:
        raise HTTPException(status_code=500, detail="S3 client not initialized. Provide AWS credentials.")
    
    try:
        # Process PDF from S3
        documents = rag_system.process_pdf_to_documents(request.s3_url, request.document_name)
        
        # Setup vector store for this specific document
        rag_system.setup_vectorstore_for_document(request.document_name)
        
        # Setup RAG chain for this specific document
        rag_system.setup_rag_chain_for_document(request.document_name)
        
        # Save documents
        rag_system.save_documents("car_documents.json")
        
        return ProcessS3Response(
            message=f"Successfully processed PDF from S3 and created {len(documents)} documents",
            documents_created=len(documents),
            status="ready",
            s3_url=request.s3_url
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing S3 PDF: {str(e)}")

@app.get("/documents/info")
async def get_documents_info():
    """
    Get information about loaded documents
    """
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not rag_system.documents_by_name:
        return {"message": "No documents loaded", "count": 0}
    
    # Get document statistics
    total_docs = sum(len(docs) for docs in rag_system.documents_by_name.values())
    document_names = list(rag_system.documents_by_name.keys())
    
    return {
        "total_documents": total_docs,
        "document_names": document_names,
        "documents_count": {name: len(docs) for name, docs in rag_system.documents_by_name.items()},
        "vectorstores_ready": len(rag_system.vectorstores),
        "rag_chains_ready": len(rag_system.rag_chains),
        "s3_enabled": rag_system.s3_client is not None
    }

@app.get("/documents/detailed")
async def get_detailed_documents_info():
    """
    Get detailed information about loaded documents including metadata
    """
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not rag_system.documents_by_name:
        return {"message": "No documents loaded", "count": 0}
    
    # Group documents by document_name
    documents_by_name = {}
    for doc_name, documents in rag_system.documents_by_name.items():
        pages = []
        sources = set()
        document_type = "unknown"
        
        for doc in documents:
            pages.append(doc.metadata.get("page_number", 0))
            sources.add(doc.metadata.get("source", "unknown"))
            if document_type == "unknown":
                document_type = doc.metadata.get("document_type", "unknown")
        
        documents_by_name[doc_name] = {
            "count": len(documents),
            "pages": sorted(pages),
            "sources": list(sources),
            "document_type": document_type,
            "vectorstore_ready": doc_name in rag_system.vectorstores,
            "rag_chain_ready": doc_name in rag_system.rag_chains
        }
    
    total_docs = sum(len(docs) for docs in rag_system.documents_by_name.values())
    
    return {
        "total_documents": total_docs,
        "documents_by_name": documents_by_name,
        "s3_enabled": rag_system.s3_client is not None
    }

@app.post("/generate-maintenance-reminders", response_model=MaintenanceReminderResponse)
async def generate_maintenance_reminders(request: MaintenanceReminderRequest):
    """
    Generate maintenance reminders based on current mileage and car manual content
    """
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        reminders = rag_system.generate_maintenance_reminders(
            document_name=request.document_name,
            current_mileage=request.current_mileage,
            user_id=request.user_id
        )
        
        return MaintenanceReminderResponse(
            success=True,
            reminders=[ReminderData(**reminder) for reminder in reminders],
            document_name=request.document_name,
            current_mileage=request.current_mileage,
            generated_at=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        return MaintenanceReminderResponse(
            success=False,
            reminders=[],
            document_name=request.document_name,
            current_mileage=request.current_mileage,
            generated_at=datetime.utcnow().isoformat() + "Z",
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
