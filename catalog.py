# -*- coding: utf-8 -*-
"""
Car Catalog RAG System
Processes Nissan Altima owner manual and provides question answering via RAG
"""

import os
import re
import json
from typing import List, Dict, Any

from mistralai import Mistral
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


class CarCatalogRAG:
    """
    RAG system for car catalog question answering
    """
    
    def __init__(self, mistral_api_key: str, openai_api_key: str, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system
        
        Args:
            mistral_api_key: API key for Mistral OCR
            openai_api_key: API key for OpenAI
            persist_directory: Directory to persist the vector store
        """
        self.mistral_client = Mistral(api_key=mistral_api_key)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Initialize vector store
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        # Initialize documents
        self.all_documents = []
    
    def process_pdf_with_mistral(self, pdf_path: str):
        """
        Uploads a PDF to Mistral OCR and returns the OCR response.
        """
        uploaded_pdf = self.mistral_client.files.upload(
            file={
                "file_name": pdf_path,
                "content": open(pdf_path, "rb"),
            },
            purpose="ocr"
        )

        # Get signed URL for OCR processing
        signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)

        # OCR Processing
        ocr_response = self.mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            }
        )

        return ocr_response
    
    def remove_all_images_from_page(self, page):
        """Remove all images from the page"""
        page.images = []
        page.markdown = re.sub(r'!\[.*?\]\([^)]+\)', '', page.markdown)
    
    def process_pages_to_markdown_list(self, pages):
        """Convert pages into a list of plain markdown"""
        markdown_list = []
        for idx, page in enumerate(pages):
            markdown_list.append(page.markdown or "")
        return markdown_list
    
    def strip_links_and_emails(self, text: str, keep_md_text=True):
        """Remove links and emails from text"""
        if not text:
            return text or ""
        
        # URL patterns
        RE_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
        RE_URL = re.compile(r"(?<!\()(?<!\[)(?:https?://|www\.)[^\s)>\]}\"'""''']+", re.IGNORECASE)
        RE_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
        
        # 1) Markdown links: [text](url)
        if keep_md_text:
            text = RE_MD_LINK.sub(r"\1", text)
        else:
            text = RE_MD_LINK.sub("", text)

        # 2) Bare URLs (http/https/www)
        text = RE_URL.sub("", text)

        # 3) Emails
        text = RE_EMAIL.sub("", text)

        # Clean up extra spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    
    def remove_memo_tokens(self, text: str) -> str:
        """Remove MEMO tokens from text"""
        if not text:
            return text or ""
        
        MEMO_RE = re.compile(
            r'(?<![A-Za-z0-9])'           # left boundary
            r'M[\s.\-•_/]*E[\s.\-•_/]*M[\s.\-•_/]*[O0]'
            r'(?![A-Za-z0-9])',           # right boundary
            re.IGNORECASE
        )
        
        cleaned = MEMO_RE.sub("", text)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    
    def process_pdf_to_documents(self, pdf_path: str, document_name: str = "Nissan Altima Manual"):
        """
        Process PDF and create documents for RAG
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Process PDF with Mistral OCR
        ocr_response = self.process_pdf_with_mistral(pdf_path)
        
        # Remove images from all pages
        for page in ocr_response.pages:
            self.remove_all_images_from_page(page)
        
        # Convert to markdown list
        markdown_pages = self.process_pages_to_markdown_list(ocr_response.pages)
        
        # Drop empty pages
        markdown_pages = [str(md) for md in markdown_pages if str(md).strip() != "."]
        
        # Create documents
        documents = []
        for page_idx, markdown in enumerate(markdown_pages, 1):
            # Clean the text
            cleaned_text = self.strip_links_and_emails(markdown, keep_md_text=True)
            cleaned_text = self.remove_memo_tokens(cleaned_text)
            
            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "document_name": document_name,
                    "document_type": "manual",
                    "page_number": page_idx,
                    "total_pages": len(markdown_pages)
                }
            )
            documents.append(doc)
        
        self.all_documents.extend(documents)
        print(f"Created {len(documents)} documents from PDF")
        return documents
    
    def setup_vectorstore(self):
        """Setup the vector store with documents"""
        if not self.all_documents:
            raise ValueError("No documents available. Process PDFs first.")
        
        print("Setting up vector store...")
        
        # Create persistent ChromaDB
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name="car_catalog",
            embedding_function=self.embeddings,
        )
        
        # Add documents to vector store
        self.vectorstore.add_documents(self.all_documents)
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
        )
        
        print(f"Vector store setup complete with {len(self.all_documents)} documents")
    
    def setup_rag_chain(self):
        """Setup the RAG chain for question answering"""
        if not self.retriever:
            raise ValueError("Vector store not setup. Call setup_vectorstore() first.")
        
        system_prompt = (
            "You are an assistant for Nissan Altima owner manual question-answering tasks.\n"
            "Use the retrieved context to answer questions about the car manual. "
            "If you don't know the answer based on the provided context, say so.\n"
            "Cite relevant page numbers when helpful. Format your answer clearly.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        
        print("RAG chain setup complete")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer using RAG
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with answer and context information
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not setup. Call setup_rag_chain() first.")
        
        result = self.rag_chain.invoke({"input": question})
        
        return {
            "answer": result["answer"],
            "context": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["context"]
            ]
        }
    
    def save_documents(self, filepath: str = "car_documents.json"):
        """Save processed documents to JSON file"""
        docs_to_save = []
        for doc in self.all_documents:
            docs_to_save.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(docs_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(docs_to_save)} documents to {filepath}")
    
    def load_documents(self, filepath: str = "car_documents.json"):
        """Load documents from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        self.all_documents = [
            Document(page_content=d["page_content"], metadata=d["metadata"]) 
            for d in loaded
        ]
        
        print(f"Loaded {len(self.all_documents)} documents from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = CarCatalogRAG(
        mistral_api_key="your_mistral_api_key",
        openai_api_key="your_openai_api_key"
    )
    
    # Process PDF (only needed once)
    # rag.process_pdf_to_documents("2022-nissan-altima-owner-manual.pdf")
    
    # Or load existing documents
    # rag.load_documents("car_documents.json")
    
    # Setup RAG
    # rag.setup_vectorstore()
    # rag.setup_rag_chain()
    
    # Ask questions
    # result = rag.ask_question("What is the recommended oil viscosity?")
    # print(result["answer"])
