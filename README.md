# Car Catalog RAG API

A FastAPI-based RAG (Retrieval-Augmented Generation) system for answering questions about the Nissan Altima owner manual using Mistral OCR and OpenAI.

## Features

- **PDF Processing**: Uses Mistral OCR to extract text from PDF documents
- **RAG System**: Combines vector search with OpenAI GPT for accurate answers
- **Persistent Storage**: ChromaDB for efficient document storage and retrieval
- **REST API**: Clean FastAPI endpoints for integration with other systems
- **Context-Aware**: Provides source citations and context for answers

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp env_example.txt .env
```

Edit `.env` and add your actual API keys:

```
MISTRAL_API_KEY=your_actual_mistral_api_key
OPENAI_API_KEY=your_actual_openai_api_key
```

### 3. Start the Server

```bash
python start_server.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

```
GET /health
```

Returns the status of the RAG system and number of loaded documents.

### Process PDF

```
POST /process-pdf?pdf_path=2022-nissan-altima-owner-manual.pdf
```

Processes a PDF file and creates the vector store. Only needed once per PDF.

### Ask Question

```
POST /ask
Content-Type: application/json

{
    "question": "What is the recommended oil viscosity?"
}
```

### Get Documents Info

```
GET /documents/info
```

Returns information about loaded documents.

## Usage Example

### 1. Start the server

```bash
python start_server.py
```

### 2. Process the PDF (first time only)

```bash
curl -X POST "http://localhost:8000/process-pdf?pdf_path=2022-nissan-altima-owner-manual.pdf"
```

### 3. Ask questions

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the recommended oil viscosity for my Nissan Altima?"}'
```

## Integration with Spring Boot

Your Spring Boot backend can integrate with this API by making HTTP requests:

```java
// Example Spring Boot service
@Service
public class CarCatalogService {

    @Value("${rag.api.url:http://localhost:8000}")
    private String ragApiUrl;

    public String askQuestion(String question) {
        // Make HTTP request to RAG API
        QuestionRequest request = new QuestionRequest(question);
        QuestionResponse response = restTemplate.postForObject(
            ragApiUrl + "/ask",
            request,
            QuestionResponse.class
        );
        return response.getAnswer();
    }
}
```

## File Structure

```
├── catalog.py          # Core RAG system class
├── main.py            # FastAPI application
├── start_server.py    # Startup script
├── requirements.txt   # Python dependencies
├── env_example.txt    # Environment variables template
└── README.md         # This file
```

## Environment Variables

- `MISTRAL_API_KEY`: Your Mistral API key for OCR processing
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and LLM
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

## Notes

- The system uses persistent storage (`./chroma_db`) for the vector store
- Documents are cached in `car_documents.json` to avoid reprocessing
- The API automatically loads existing documents on startup
- Only process PDFs once - subsequent startups will use cached data
