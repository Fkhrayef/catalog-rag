#!/usr/bin/env python3
"""
Test script for Car Catalog RAG API
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        print(f"📄 Documents loaded: {data['documents_loaded']}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_process_pdf():
    """Test PDF processing"""
    print("\n📄 Testing PDF processing...")
    try:
        response = requests.post(f"{API_BASE_URL}/process-pdf?pdf_path=2022-nissan-altima-owner-manual.pdf")
        response.raise_for_status()
        data = response.json()
        print(f"✅ PDF processed successfully: {data['message']}")
        print(f"📊 Documents created: {data['documents_created']}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ PDF processing failed: {e}")
        return False

def test_ask_question(question: str):
    """Test asking a question"""
    print(f"\n❓ Testing question: '{question}'")
    try:
        payload = {"question": question}
        response = requests.post(f"{API_BASE_URL}/ask", json=payload)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Answer received:")
        print(f"📝 {data['answer']}")
        print(f"📚 Sources: {len(data['sources'])}")
        
        for i, source in enumerate(data['sources'][:2], 1):  # Show first 2 sources
            print(f"   Source {i}: Page {source['page_number']} - {source['content_preview']}")
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Question failed: {e}")
        return False

def test_documents_info():
    """Test documents info endpoint"""
    print("\n📊 Testing documents info...")
    try:
        response = requests.get(f"{API_BASE_URL}/documents/info")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Documents info retrieved:")
        print(f"   Total documents: {data['total_documents']}")
        print(f"   Document names: {data['document_names']}")
        print(f"   Vectorstore ready: {data['vectorstore_ready']}")
        print(f"   RAG chain ready: {data['rag_chain_ready']}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Documents info failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Car Catalog RAG API Tests")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n❌ Server not running. Please start the server first:")
        print("   python start_server.py")
        return
    
    # Test documents info
    test_documents_info()
    
    # Test questions
    test_questions = [
        "What is the recommended oil viscosity for my Nissan Altima?",
        "How often should I change the oil?",
        "What is the NISSAN CUSTOMER CARE PROGRAM?",
        "How do I check the tire pressure?"
    ]
    
    print("\n" + "=" * 50)
    print("🧪 Testing Question Answering")
    print("=" * 50)
    
    for question in test_questions:
        test_ask_question(question)
        time.sleep(1)  # Small delay between requests
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    main()
