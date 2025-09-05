#!/usr/bin/env python3
"""
Startup script for Car Catalog RAG API
"""

import os
import sys
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if API keys are set
    mistral_key = os.getenv("MISTRAL_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not mistral_key or mistral_key == "your_mistral_api_key_here":
        print("❌ Error: MISTRAL_API_KEY not set or using placeholder value")
        print("Please set your actual Mistral API key in the .env file")
        sys.exit(1)
    
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("❌ Error: OPENAI_API_KEY not set or using placeholder value")
        print("Please set your actual OpenAI API key in the .env file")
        sys.exit(1)
    
    print("✅ API keys configured")
    print("🚀 Starting Car Catalog RAG API...")
    
    # Import and run the FastAPI app
    import uvicorn
    from main import app
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
