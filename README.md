# Car Catalog RAG API

A FastAPI-based RAG (Retrieval-Augmented Generation) system for answering questions about car manuals using Mistral OCR and OpenAI. Each manual gets its own dedicated vector store for optimal performance and accuracy.

## Features

- **Multi-Catalog Support**: Each car manual has its own dedicated vector store
- **S3 Integration**: Process PDFs directly from AWS S3 URLs
- **Large PDF Support**: Automatic chunking for PDFs > 300k tokens
- **RAG System**: Combines vector search with OpenAI GPT for accurate answers
- **Persistent Storage**: ChromaDB for efficient document storage and retrieval
- **Production Ready**: Clean FastAPI endpoints with proper error handling

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and add your API keys:

```bash
cp env_example.txt .env
```

Edit `.env` and add your actual API keys:

```
MISTRAL_API_KEY=your_actual_mistral_api_key
OPENAI_API_KEY=your_actual_openai_api_key

# AWS S3 Configuration (required for S3 PDF processing)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
```

### 3. Start the Server

```bash
python start_server.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Process PDF from S3

**POST** `/process-s3`

Processes a PDF from S3 URL and creates a dedicated vector store for that manual.

**Request:**

```json
{
  "s3_url": "https://your-bucket.s3.region.amazonaws.com/catalogs/nissan-altima-2022.pdf",
  "document_name": "Nissan Altima 2022 Manual"
}
```

**Response:**

```json
{
  "message": "Successfully processed PDF from S3 and created 642 documents",
  "documents_created": 642,
  "status": "ready",
  "s3_url": "https://your-bucket.s3.region.amazonaws.com/catalogs/nissan-altima-2022.pdf"
}
```

### 2. Ask Question

**POST** `/ask`

Ask a question about a specific car manual. **Document name is required.**

**Request:**

```json
{
  "question": "What is the recommended oil viscosity?",
  "document_name": "Nissan Altima 2022 Manual"
}
```

**Response:**

```json
{
  "answer": "The recommended oil viscosity for the Nissan Altima is 0W-20. This is specified in the owner's manual and should be used for optimal engine performance and fuel economy.",
  "context": [
    {
      "page_content": "Engine Oil Recommendations\n\nUse 0W-20 engine oil for optimal performance...",
      "metadata": {
        "document_name": "Nissan Altima 2022 Manual",
        "page_number": 45,
        "document_type": "manual",
        "source": "https://your-bucket.s3.amazonaws.com/catalogs/nissan-altima-2022.pdf"
      }
    }
  ],
  "sources": [
    {
      "page_number": 45,
      "document_name": "Nissan Altima 2022 Manual",
      "document_type": "manual",
      "source": "https://your-bucket.s3.amazonaws.com/catalogs/nissan-altima-2022.pdf",
      "content_preview": "Engine Oil Recommendations\n\nUse 0W-20 engine oil for optimal performance..."
    }
  ],
  "metadata": {
    "question": "What is the recommended oil viscosity?",
    "document_name": "Nissan Altima 2022 Manual",
    "sources_count": 1,
    "timestamp": "2024-01-15T10:30:00Z",
    "model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-large"
  }
}
```

### 3. Get Available Documents

**GET** `/documents/info`

Returns information about all loaded documents.

**Response:**

```json
{
  "total_documents": 1205,
  "document_names": ["Nissan Altima 2022 Manual", "Honda Civic 2023 Manual"],
  "documents_count": {
    "Nissan Altima 2022 Manual": 642,
    "Honda Civic 2023 Manual": 563
  },
  "vectorstores_ready": 2,
  "rag_chains_ready": 2,
  "s3_enabled": true
}
```

### 4. Health Check

**GET** `/health`

Returns the status of the RAG system.

**Response:**

```json
{
  "status": "healthy",
  "message": "RAG system is running",
  "documents_loaded": 1205,
  "s3_enabled": true
}
```

## Usage Examples

### 1. Process Multiple Car Manuals

```bash
# Process Nissan Altima manual
curl -X POST "http://localhost:8000/process-s3" \
     -H "Content-Type: application/json" \
     -d '{
       "s3_url": "https://your-bucket.s3.amazonaws.com/catalogs/nissan-altima-2022.pdf",
       "document_name": "Nissan Altima 2022 Manual"
     }'

# Process Honda Civic manual
curl -X POST "http://localhost:8000/process-s3" \
     -H "Content-Type: application/json" \
     -d '{
       "s3_url": "https://your-bucket.s3.amazonaws.com/catalogs/honda-civic-2023.pdf",
       "document_name": "Honda Civic 2023 Manual"
     }'
```

### 2. Ask Questions About Specific Manuals

```bash
# Ask about Nissan Altima
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the recommended tire pressure?",
       "document_name": "Nissan Altima 2022 Manual"
     }'

# Ask about Honda Civic
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "How often should I change the oil?",
       "document_name": "Honda Civic 2023 Manual"
     }'
```

### 3. Check Available Documents

```bash
curl -X GET "http://localhost:8000/documents/info"
```

## Integration with Spring Boot

Your Spring Boot backend can integrate with this API:

```java
@Service
public class CarCatalogService {

    @Value("${rag.api.url:http://localhost:8000}")
    private String ragApiUrl;

    public String askQuestion(String question, String documentName) {
        QuestionRequest request = new QuestionRequest(question, documentName);
        QuestionResponse response = restTemplate.postForObject(
            ragApiUrl + "/ask",
            request,
            QuestionResponse.class
        );
        return response.getAnswer();
    }
}
```

## Architecture Benefits

- **Separate Vector Stores**: Each manual has its own dedicated vector store
- **No Cross-Contamination**: Questions about one manual won't get answers from another
- **Scalable**: Can handle unlimited number of car manuals
- **Memory Efficient**: Only loads relevant documents for each query
- **Large PDF Support**: Automatically handles PDFs of any size
- **Persistent**: No need to reprocess PDFs on restart

## File Structure

```
motor/
├── catalog.py              # Core RAG system
├── main.py                 # FastAPI application
├── start_server.py         # Server startup script
├── requirements.txt        # Dependencies
├── env_example.txt         # Environment template
├── README.md              # This documentation
├── car_documents.json      # Processed documents (auto-generated)
└── chroma_db/             # Vector stores (auto-generated)
    ├── nissan_altima_2022_manual/
    ├── honda_civic_2023_manual/
    └── chroma.sqlite3
```

## Environment Variables

- `MISTRAL_API_KEY`: Your Mistral API key for OCR processing
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and LLM
- `AWS_ACCESS_KEY_ID`: AWS access key for S3 integration
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for S3 integration
- `AWS_REGION`: AWS region (default: us-east-1)

## Production Deployment

The system is production-ready and will work identically when hosted:

1. **Deploy** the application to your hosting platform
2. **Set environment variables** in your hosting platform
3. **Process PDFs once** via the `/process-s3` endpoint
4. **Start asking questions** - no reprocessing needed!

## Notes

- Each manual gets its own vector store for optimal performance
- Documents are cached in `car_documents.json` to avoid reprocessing
- The API automatically loads existing documents on startup
- Large PDFs are automatically chunked to handle any size
- Only process each PDF once - subsequent startups use cached data
