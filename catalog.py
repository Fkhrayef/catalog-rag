# -*- coding: utf-8 -*-
"""
Car Catalog RAG System
Processes Nissan Altima owner manual and provides question answering via RAG
"""

import os
import re
import json
import tempfile
from typing import List, Dict, Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
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
    
    def __init__(self, mistral_api_key: str, openai_api_key: str, persist_directory: str = "./chroma_db", 
                 aws_access_key: str = None, aws_secret_key: str = None, aws_region: str = "us-east-1"):
        """
        Initialize the RAG system
        
        Args:
            mistral_api_key: API key for Mistral OCR
            openai_api_key: API key for OpenAI
            persist_directory: Directory to persist the vector store
            aws_access_key: AWS access key for S3
            aws_secret_key: AWS secret key for S3
            aws_region: AWS region for S3
        """
        self.mistral_client = Mistral(api_key=mistral_api_key)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Initialize vector stores (one per document)
        self.persist_directory = persist_directory
        self.vectorstores = {}  # Dictionary to store multiple vector stores
        self.retrievers = {}    # Dictionary to store multiple retrievers
        self.rag_chains = {}    # Dictionary to store multiple RAG chains
        
        # Initialize documents (grouped by document_name)
        self.documents_by_name = {}  # {document_name: [documents]}
        
        # Initialize S3 client if AWS credentials provided
        self.s3_client = None
        if aws_access_key and aws_secret_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
    
    def download_from_s3(self, s3_url: str) -> str:
        """
        Download a PDF from S3 URL to a temporary file
        
        Args:
            s3_url: S3 URL (e.g., https://bucket.s3.region.amazonaws.com/key)
            
        Returns:
            Path to the downloaded temporary file
        """
        if not self.s3_client:
            raise ValueError("S3 client not initialized. Provide AWS credentials.")
        
        # Parse S3 URL
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc.split('.')[0]  # Extract bucket from hostname
        key = parsed_url.path.lstrip('/')  # Remove leading slash
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Download file from S3
            self.s3_client.download_file(bucket_name, key, temp_path)
            print(f"Downloaded PDF from S3: {s3_url}")
            return temp_path
        except ClientError as e:
            os.unlink(temp_path)  # Clean up temp file on error
            raise Exception(f"Failed to download from S3: {e}")
    
    def is_s3_url(self, url: str) -> bool:
        """
        Check if the URL is an S3 URL
        
        Args:
            url: URL to check
            
        Returns:
            True if it's an S3 URL
        """
        return url.startswith('https://') and '.s3.' in url and 'amazonaws.com' in url
    
    def process_pdf_with_mistral(self, pdf_path: str):
        """
        Uploads a PDF to Mistral OCR and returns the OCR response.
        Handles large PDFs by splitting them into smaller chunks if needed.
        """
        try:
            # First attempt: try to process the entire PDF
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
            
        except Exception as e:
            if "max_tokens_per_request" in str(e) or "300000" in str(e):
                print(f"PDF too large for single processing. Attempting to split into smaller chunks...")
                return self._process_large_pdf_in_chunks(pdf_path)
            else:
                raise e
    
    def _process_large_pdf_in_chunks(self, pdf_path: str):
        """
        Process large PDFs by splitting them into smaller chunks
        """
        import PyPDF2
        import tempfile
        import os
        
        try:
            # Read the PDF
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"PDF has {total_pages} pages. Splitting into chunks...")
                
                # Process in chunks of 50 pages (adjust based on your needs)
                chunk_size = 50
                all_pages = []
                
                for start_page in range(0, total_pages, chunk_size):
                    end_page = min(start_page + chunk_size, total_pages)
                    
                    print(f"Processing pages {start_page + 1} to {end_page}...")
                    
                    # Create a temporary PDF with just these pages
                    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    temp_path = temp_pdf.name
                    temp_pdf.close()
                    
                    try:
                        # Create new PDF with selected pages
                        pdf_writer = PyPDF2.PdfWriter()
                        for page_num in range(start_page, end_page):
                            pdf_writer.add_page(pdf_reader.pages[page_num])
                        
                        with open(temp_path, 'wb') as temp_file:
                            pdf_writer.write(temp_file)
                        
                        # Process this chunk with Mistral
                        chunk_response = self._process_single_chunk(temp_path, start_page)
                        
                        # Add page offset to the chunk response
                        for page in chunk_response.pages:
                            page.page_number = page.page_number + start_page
                        
                        all_pages.extend(chunk_response.pages)
                        
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                
                # Create a combined response
                class CombinedOCRResponse:
                    def __init__(self, pages):
                        self.pages = pages
                
                return CombinedOCRResponse(all_pages)
                
        except ImportError:
            raise Exception("PyPDF2 is required for processing large PDFs. Install it with: pip install PyPDF2")
        except Exception as e:
            raise Exception(f"Error processing large PDF: {e}")
    
    def _process_single_chunk(self, pdf_path: str, page_offset: int = 0):
        """
        Process a single PDF chunk with Mistral OCR
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
    
    def process_pdf_to_documents(self, pdf_path_or_url: str, document_name: str = "Nissan Altima Manual"):
        """
        Process PDF and create documents for RAG
        Supports both local file paths and S3 URLs
        
        Args:
            pdf_path_or_url: Local file path or S3 URL
            document_name: Name for the document
            
        Returns:
            List of processed documents
        """
        temp_file_path = None
        
        try:
            # Check if it's an S3 URL
            if self.is_s3_url(pdf_path_or_url):
                print(f"Processing PDF from S3: {pdf_path_or_url}")
                temp_file_path = self.download_from_s3(pdf_path_or_url)
                pdf_path = temp_file_path
            else:
                print(f"Processing local PDF: {pdf_path_or_url}")
                pdf_path = pdf_path_or_url
                
                # Check if local file exists
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
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
                        "total_pages": len(markdown_pages),
                        "source": pdf_path_or_url  # Keep track of original source
                    }
                )
                documents.append(doc)
            
            # Store documents by document_name
            self.documents_by_name[document_name] = documents
            print(f"Created {len(documents)} documents for '{document_name}'")
            return documents
            
        finally:
            # Clean up temporary file if it was downloaded from S3
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
    
    def setup_vectorstore_for_document(self, document_name: str):
        """Setup vector store for a specific document"""
        if document_name not in self.documents_by_name:
            raise ValueError(f"Document '{document_name}' not found. Available documents: {list(self.documents_by_name.keys())}")
        
        documents = self.documents_by_name[document_name]
        if not documents:
            raise ValueError(f"No documents available for '{document_name}'")
        
        print(f"Setting up vector store for '{document_name}' with {len(documents)} documents...")
        
        # Create collection name (sanitized)
        collection_name = document_name.replace(" ", "_").replace("-", "_").lower()
        
        # Create persistent ChromaDB for this document
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        
        # Add documents to vector store
        vectorstore.add_documents(documents)
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
        )
        
        # Store in dictionaries
        self.vectorstores[document_name] = vectorstore
        self.retrievers[document_name] = retriever
        
        print(f"Vector store setup complete for '{document_name}' with {len(documents)} documents")
        return vectorstore
    
    def setup_vectorstore(self):
        """Setup vector stores for all documents"""
        if not self.documents_by_name:
            raise ValueError("No documents available. Process PDFs first.")
        
        print("Setting up vector stores for all documents...")
        
        for document_name in self.documents_by_name.keys():
            self.setup_vectorstore_for_document(document_name)
        
        print(f"Vector stores setup complete for {len(self.documents_by_name)} documents")
    
    def setup_rag_chain_for_document(self, document_name: str):
        """Setup RAG chain for a specific document"""
        if document_name not in self.retrievers:
            raise ValueError(f"Retriever for '{document_name}' not found. Setup vector store first.")
        
        system_prompt = (
            f"You are an assistant for {document_name} question-answering tasks.\n"
            "Use the retrieved context to answer questions about this specific manual. "
            "If you don't know the answer based on the provided context, say so.\n"
            "Cite relevant page numbers when helpful. Format your answer clearly.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(self.retrievers[document_name], question_answer_chain)
        
        self.rag_chains[document_name] = rag_chain
        print(f"RAG chain setup complete for '{document_name}'")
        return rag_chain
    
    def setup_rag_chain(self):
        """Setup RAG chains for all documents"""
        if not self.retrievers:
            raise ValueError("Vector stores not setup. Call setup_vectorstore() first.")
        
        print("Setting up RAG chains for all documents...")
        
        for document_name in self.retrievers.keys():
            self.setup_rag_chain_for_document(document_name)
        
        print(f"RAG chains setup complete for {len(self.retrievers)} documents")
    
    def ask_question(self, question: str, document_name: str = None) -> Dict[str, Any]:
        """
        Ask a question and get an answer using RAG
        
        Args:
            question: The question to ask
            document_name: Specific document to query (required)
            
        Returns:
            Dictionary with answer and context information
        """
        if not document_name:
            available_docs = list(self.documents_by_name.keys())
            return {
                "answer": f"Please specify a document_name. Available documents: {available_docs}",
                "context": [],
                "sources": []
            }
        
        if document_name not in self.rag_chains:
            available_docs = list(self.documents_by_name.keys())
            return {
                "answer": f"Document '{document_name}' not found or not ready. Available documents: {available_docs}",
                "context": [],
                "sources": []
            }
        
        try:
            result = self.rag_chains[document_name].invoke({"input": question})
            
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
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "context": [],
                "sources": []
            }
    
    def save_documents(self, filepath: str = "car_documents.json"):
        """Save processed documents to JSON file"""
        docs_to_save = {}
        for doc_name, documents in self.documents_by_name.items():
            docs_to_save[doc_name] = []
            for doc in documents:
                docs_to_save[doc_name].append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(docs_to_save, f, ensure_ascii=False, indent=2)
        
        total_docs = sum(len(docs) for docs in docs_to_save.values())
        print(f"Saved {total_docs} documents across {len(docs_to_save)} document collections to {filepath}")
    
    def load_documents(self, filepath: str = "car_documents.json"):
        """Load documents from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        self.documents_by_name = {}
        for doc_name, docs_data in loaded.items():
            self.documents_by_name[doc_name] = [
                Document(page_content=d["page_content"], metadata=d["metadata"]) 
                for d in docs_data
            ]
        
        total_docs = sum(len(docs) for docs in self.documents_by_name.values())
        print(f"Loaded {total_docs} documents across {len(self.documents_by_name)} document collections from {filepath}")