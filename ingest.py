"""
============================================================
STEP 1: DATA INGESTION & VECTOR STORE BUILDER
============================================================
File: src/ingest.py

PURPOSE:
  This module reads the PhonePay knowledge base documents,
  splits them into chunks, converts chunks to embeddings,
  and stores them in ChromaDB (local vector database).

WHY THIS IS NEEDED:
  The LLM (Gemini) doesn't have PhonePay-specific knowledge.
  We convert our documents into numeric vectors (embeddings)
  so similar texts can be retrieved quickly when a user asks
  a question. This is the "R" (Retrieval) in RAG.

FLOW:
  Raw Text Files → Text Chunks → Embeddings → ChromaDB
============================================================
"""

import os
import json
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# ── Configure Loguru Logger ──────────────────────────────
logger.add(
    "logs/ingest.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def load_text_documents(file_path: str) -> list:
    """
    Load plain text documents using LangChain's TextLoader.
    
    Args:
        file_path: Path to the .txt knowledge base file
    
    Returns:
        List of LangChain Document objects
    """
    logger.info(f"Loading document: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    logger.success(f"Loaded {len(docs)} document(s) from {file_path}")
    return docs


def load_faq_as_documents(faq_path: str) -> list:
    """
    Load FAQ data and format as Question-Answer pairs.
    Each Q-A pair becomes a separate document chunk for better retrieval.
    
    Args:
        faq_path: Path to FAQ text file
    
    Returns:
        List of LangChain Document objects (one per Q-A pair)
    """
    from langchain.schema import Document
    
    logger.info(f"Loading FAQ data: {faq_path}")
    documents = []
    
    with open(faq_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by Q: to get individual Q-A pairs
    qa_pairs = content.strip().split("\n\nQ: ")
    
    for i, pair in enumerate(qa_pairs):
        if not pair.strip():
            continue
        # Add back the "Q: " prefix if it was removed during split
        if not pair.startswith("Q:"):
            pair = "Q: " + pair
        
        # Create a Document with metadata
        doc = Document(
            page_content=pair.strip(),
            metadata={
                "source": "faq_data.txt",
                "type": "faq",
                "chunk_id": i
            }
        )
        documents.append(doc)
    
    logger.success(f"Loaded {len(documents)} FAQ entries")
    return documents


def load_bills_context(bills_path: str) -> list:
    """
    Load user bills JSON and convert to readable text documents.
    This gives the RAG system context about bill categories and due dates.
    
    Args:
        bills_path: Path to user_bills_data.json
    
    Returns:
        List of LangChain Document objects
    """
    from langchain.schema import Document
    
    logger.info(f"Loading bills data: {bills_path}")
    
    with open(bills_path, "r") as f:
        bills_data = json.load(f)
    
    documents = []
    
    # Convert bill categories to searchable text
    for category, details in bills_data.get("bill_categories", {}).items():
        text = f"Bill Type: {category.upper()}\n"
        text += f"Description: {details.get('description', '')}\n"
        
        if "payment_steps" in details:
            text += "Payment Steps:\n"
            for step in details["payment_steps"]:
                text += f"  - {step}\n"
        
        if "typical_due_date" in details:
            text += f"Typical Due Date: {details['typical_due_date']}\n"
        
        if "providers" in details:
            text += f"Providers: {', '.join(details['providers'])}\n"
        
        if "tax_benefit" in details:
            text += f"Tax Benefit: {details['tax_benefit']}\n"
        
        doc = Document(
            page_content=text.strip(),
            metadata={
                "source": "user_bills_data.json",
                "type": "bill_category",
                "category": category
            }
        )
        documents.append(doc)
    
    logger.success(f"Loaded {len(documents)} bill category documents")
    return documents


def split_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split documents into smaller chunks.
    
    WHY CHUNKING?
    - LLMs have context limits; we can't feed all documents at once.
    - Smaller chunks = more focused, precise retrieval.
    - Overlapping chunks ensures no information is cut off at boundaries.
    
    Args:
        documents: List of Document objects
        chunk_size: Max number of characters per chunk (default: 500)
        chunk_overlap: Characters shared between adjacent chunks (default: 50)
    
    Returns:
        List of smaller Document chunks
    """
    logger.info(f"Splitting {len(documents)} documents into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # Split on paragraphs first
    )
    
    chunks = splitter.split_documents(documents)
    logger.success(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def build_vector_store(chunks: list, persist_dir: str = "./chroma_db") -> Chroma:
    """
    Convert document chunks to embeddings and store in ChromaDB.
    
    WHY EMBEDDINGS?
    - Embeddings are numeric vector representations of text.
    - Similar meanings → similar vectors → stored close in vector space.
    - When user asks a question, we embed it and find closest chunk vectors.
    
    WHY CHROMADB?
    - Lightweight local vector database.
    - No external API needed; runs on your machine.
    - Persists to disk so we don't re-embed on every app start.
    
    Args:
        chunks: List of document chunks
        persist_dir: Directory to save ChromaDB
    
    Returns:
        Chroma vector store object
    """
    logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
    
    # Free, local embedding model — no API key needed
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    logger.info(f"Building ChromaDB vector store at: {persist_dir}")
    
    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name="phonepay_knowledge"
    )
    
    logger.success(f"Vector store built with {len(chunks)} chunks at {persist_dir}")
    return vector_store


def run_ingestion_pipeline():
    """
    Master function that runs the complete ingestion pipeline:
    1. Load all data sources
    2. Split into chunks
    3. Build vector store
    """
    logger.info("=" * 60)
    logger.info("STARTING PHONEPAY RAG INGESTION PIPELINE")
    logger.info("=" * 60)
    
    # Paths from environment or defaults
    kb_path = os.getenv("KNOWLEDGE_BASE_PATH", "./data/phonepay_knowledge_base.txt")
    faq_path = os.getenv("FAQ_DATA_PATH", "./data/faq_data.txt")
    bills_path = os.getenv("USER_BILLS_PATH", "./data/user_bills_data.json")
    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # Step 1: Load all documents
    all_documents = []
    
    if Path(kb_path).exists():
        kb_docs = load_text_documents(kb_path)
        all_documents.extend(kb_docs)
    else:
        logger.warning(f"Knowledge base not found at: {kb_path}")
    
    if Path(faq_path).exists():
        faq_docs = load_faq_as_documents(faq_path)
        all_documents.extend(faq_docs)
    else:
        logger.warning(f"FAQ data not found at: {faq_path}")
    
    if Path(bills_path).exists():
        bills_docs = load_bills_context(bills_path)
        all_documents.extend(bills_docs)
    else:
        logger.warning(f"Bills data not found at: {bills_path}")
    
    logger.info(f"Total documents loaded: {len(all_documents)}")
    
    # Step 2: Split into chunks
    chunk_size = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
    chunks = split_documents(all_documents, chunk_size, chunk_overlap)
    
    # Step 3: Build vector store
    vector_store = build_vector_store(chunks, chroma_path)
    
    logger.info("=" * 60)
    logger.success("INGESTION PIPELINE COMPLETE!")
    logger.info(f"Total chunks indexed: {len(chunks)}")
    logger.info(f"Vector store saved at: {chroma_path}")
    logger.info("=" * 60)
    
    return vector_store


if __name__ == "__main__":
    run_ingestion_pipeline()
