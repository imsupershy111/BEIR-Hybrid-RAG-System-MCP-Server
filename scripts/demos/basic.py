#!/usr/bin/env python3
"""
Hybrid RAG System - Demo Script for BEIR (SciFact/FiQA)
Main script demonstrating advanced hybrid search (BM25+ & Vector) with RAG.
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Tải các biến môi trường
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI # OpenRouter dùng chuẩn của OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.hybrid_rag.document_loader import DocumentLoaderUtility
from src.hybrid_rag.utils import configure_logging
# THAY ĐỔI: Nhập hàm khởi tạo Hybrid Retriever mới
from src.hybrid_rag.hybrid_retriever import create_beir_hybrid_retriever

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    """Main function to run the hybrid RAG system."""

    # Configure logging to suppress warnings
    configure_logging()

    # Load configuration
    config = load_config()
    print("✅ Configuration loaded from config/config.yaml")

    # --- 1. Load BEIR Documents (.jsonl) ---
    data_dir = config['data']['directory']
    data_path = Path(__file__).parent.parent.parent / data_dir
    loader = DocumentLoaderUtility(str(data_path), config=config)
    documents = loader.load_documents()

    if not documents:
        print("\n⚠️  No BEIR .jsonl documents found in the data directory.")
        print(f"⚠️  Please add corpus.jsonl files to '{data_path}' directory.")
        return

    # --- 2. Initialize Models & Embeddings ---
    print("🧠 Initializing Embeddings & LLM...")
    embeddings = OllamaEmbeddings(model=config['ollama']['embedding_model'])
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        model=config['openrouter']['model'] 
    )       

    # --- 3. Create the Vector Store (Dense Search) ---
    persist_dir = Path(__file__).parent.parent.parent / config['vector_store']['persist_directory']

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("♻️  Loading existing Vector Store từ ổ cứng...")
        vectorstore = Chroma(
            persist_directory=str(persist_dir), 
            embedding_function=embeddings
        )
    else:
        print("🏗️  Vector Store chưa có hoặc trống. Bắt đầu quá trình Embedding (sẽ mất thời gian)...")
        vectorstore = Chroma.from_documents(
            documents, 
            embeddings, 
            persist_directory=str(persist_dir)
        )
    
    print(f"✅ Vector Store Created.")

    # --- 4. Create Advanced BEIR Hybrid Retriever ---
    print("🔧 Constructing BEIR Hybrid Retriever (BM25+ & Vector Fusion)...")
    hybrid_retriever = create_beir_hybrid_retriever(
        documents=documents,
        vectorstore=vectorstore,
        config=config
    )
    print("✅ Hybrid Retriever Created.")

    # --- 6. Define the Prompt (The AI Persona) ---
    # THAY ĐỔI: Tinh chỉnh lại Prompt cho phù hợp với khoa học / tài chính
    prompt = ChatPromptTemplate.from_template("""
    You are an expert scientific and financial analyst. Answer the user's question based ONLY on the provided context.
    If the context does not contain the exact answer, state clearly that the information is not available in the documents. 
    Do not guess or hallucinate facts.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # --- 7. Construct the RAG Chain ---
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(hybrid_retriever, document_chain)
    print("✅ RAG Chain Constructed.")

    # --- 8. Execute Sample Queries ---
    # THAY ĐỔI: Test với các câu hỏi đặc thù của BEIR
    sample_queries = [
        "How does cerebral white matter diffusion change during late gestation?",
        "What imaging technique was used to assess cerebral white matter development in infants?"
    ]

    for query in sample_queries:
        print("\n" + "="*80)
        print(f"🔥 Executing Hybrid Query: '{query}'")
        print("="*80)

        try:
            # Truyền thẳng câu hỏi vào, Hybrid Retriever sẽ tự động tiền xử lý
            response = rag_chain.invoke({"input": query})

            # --- 9. Output the Result ---
            print("\n--- 📑 Retrieved Context (Hybrid Results Top K) ---")
            for i, doc in enumerate(response['context']):
                # Lấy doc_id và điểm số để kiểm tra hệ thống có chấm chuẩn không
                doc_id = doc.metadata.get('doc_id', 'unknown')
                score = doc.metadata.get('hybrid_score', 'N/A')
                content_preview = doc.page_content[:150].replace('\n', ' ')
                
                print(f"[{i+1}] Doc_ID: {doc_id} | Score: {score}")
                print(f"    {content_preview}...")

            print("\n--- 🤖 Final LLM Answer ---")
            print(response['answer'])
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")

    print("\n✅ BEIR Hybrid RAG demonstration complete!")

if __name__ == "__main__":
    main()