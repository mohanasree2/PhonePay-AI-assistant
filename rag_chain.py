"""
============================================================
STEP 2: RAG RETRIEVER + LLM CHAIN
============================================================
File: src/rag_chain.py

PURPOSE:
  This is the core engine of the RAG system.
  Given a user question:
  1. RETRIEVE top-k relevant document chunks from ChromaDB
  2. AUGMENT the user question with retrieved context
  3. GENERATE an answer using Google Gemini LLM

WHY RAG OVER PURE LLM?
  - LLMs hallucinate; RAG grounds answers in real documents.
  - Our PhonePay knowledge is specific & not in Gemini's training.
  - RAG = Accuracy + Freshness without model retraining.

FLOW:
  User Question → Embed → ChromaDB → Top-K Chunks →
  Prompt (Question + Chunks) → Gemini LLM → Answer
============================================================
"""

import os
from loguru import logger
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# ── Logger ─────────────────────────────────────────────────
logger.add(
    "logs/rag_chain.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# ── System Prompt for PhonePay Assistant ───────────────────
PHONEPAY_SYSTEM_PROMPT = """You are PhonePay AI Assistant — a helpful, friendly, and accurate virtual assistant for the PhonePay digital payments app.

You have access to PhonePay's knowledge base which includes:
- Bill payment guides (electricity, rent, water, gas, mobile)
- UPI and payment information
- Account security and KYC details
- Investment and insurance products
- Troubleshooting and error resolution
- Customer support information
- Bill due date reminders and schedules

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY based on the context provided above.
2. If the context doesn't contain enough information, say "I don't have specific information about that. Please contact PhonePay support at 1800-102-9090."
3. Be concise but complete. Use bullet points for step-by-step instructions.
4. If the question is about bill due dates or reminders, mention that the user can set reminders in the PhonePay app.
5. Always be helpful and suggest next steps.
6. If payment safety is involved, always remind users to never share UPI PIN with anyone.

YOUR ANSWER:"""


def load_vector_store(persist_dir: str = "./chroma_db") -> Chroma:
    """
    Load the pre-built ChromaDB vector store from disk.
    
    This is faster than rebuilding — embeddings are persisted once
    during ingestion and reloaded on every app start.
    
    Args:
        persist_dir: Path to ChromaDB directory
    
    Returns:
        Chroma vector store object
    """
    logger.info(f"Loading vector store from: {persist_dir}")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name="phonepay_knowledge"
    )
    
    logger.success("Vector store loaded successfully")
    return vector_store


def build_rag_chain(vector_store: Chroma) -> RetrievalQA:
    """
    Build the complete RAG chain:
    Retriever (ChromaDB) + Prompt Template + LLM (Gemini)
    
    WHY RETRIEVAL QA CHAIN?
    - RetrievalQA is LangChain's pre-built RAG pipeline.
    - It handles: embedding the question → retrieving → prompting → generating.
    - We customize the prompt to match PhonePay's domain.
    
    Args:
        vector_store: Chroma vector database
    
    Returns:
        LangChain RetrievalQA chain
    """
    # ── Load Gemini LLM ────────────────────────────────────
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Set it in your .env file.")
    
    logger.info("Initializing Google Gemini LLM (gemini-1.5-flash)...")
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
        google_api_key=api_key,
        temperature=float(os.getenv("LLM_TEMPERATURE", 0.3)),
        max_output_tokens=int(os.getenv("LLM_MAX_TOKENS", 1024)),
        convert_system_message_to_human=True
    )
    
    # ── Build Retriever ────────────────────────────────────
    top_k = int(os.getenv("TOP_K_RESULTS", 5))
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    logger.info(f"Retriever configured: top-{top_k} similarity search")
    
    # ── Custom Prompt Template ─────────────────────────────
    prompt = PromptTemplate(
        template=PHONEPAY_SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )
    
    # ── Build Chain ────────────────────────────────────────
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",              # 'stuff' = concat all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,    # Show which chunks were used
        chain_type_kwargs={"prompt": prompt}
    )
    
    logger.success("RAG chain built successfully")
    return chain


def query_rag(chain: RetrievalQA, question: str) -> dict:
    """
    Run a question through the RAG chain and return the answer + sources.
    
    Args:
        chain: The RAG chain
        question: User's question in natural language
    
    Returns:
        dict with 'answer' and 'sources' keys
    """
    logger.info(f"Processing query: {question[:80]}...")
    
    try:
        result = chain.invoke({"query": question})
        
        answer = result.get("result", "I could not find an answer.")
        source_docs = result.get("source_documents", [])
        
        # Extract source metadata
        sources = []
        for doc in source_docs:
            source_info = {
                "source": doc.metadata.get("source", "Knowledge Base"),
                "content_preview": doc.page_content[:150] + "..."
            }
            sources.append(source_info)
        
        logger.success(f"Query answered. Sources used: {len(sources)}")
        
        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "answer": "Sorry, I encountered an error. Please try again or contact PhonePay support at 1800-102-9090.",
            "sources": [],
            "num_sources": 0
        }


# ── Singleton pattern for production use ──────────────────
_rag_chain = None

def get_rag_chain() -> RetrievalQA:
    """
    Returns a cached RAG chain (loads once, reuses across queries).
    This avoids reloading the model on every user message.
    """
    global _rag_chain
    
    if _rag_chain is None:
        chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        
        # Auto-build vector store if not present
        if not os.path.exists(chroma_path):
            logger.warning("ChromaDB not found. Running ingestion pipeline first...")
            from src.ingest import run_ingestion_pipeline
            run_ingestion_pipeline()
        
        vector_store = load_vector_store(chroma_path)
        _rag_chain = build_rag_chain(vector_store)
        logger.info("RAG chain initialized and cached")
    
    return _rag_chain


if __name__ == "__main__":
    # Quick test
    chain = get_rag_chain()
    
    test_questions = [
        "How do I pay my electricity bill on PhonePay?",
        "What is the UPI transaction limit?",
        "How can I set reminders for my rent payment?",
        "What should I do if my transaction fails?"
    ]
    
    for q in test_questions:
        print(f"\nQ: {q}")
        result = query_rag(chain, q)
        print(f"A: {result['answer'][:300]}...")
        print(f"   Sources: {result['num_sources']} chunks used")
        print("-" * 60)
