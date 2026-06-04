#!/usr/bin/env python3
"""
MCP Server cho BEIR Hybrid RAG (SciFact/FiQA)
Allows Claude Desktop to query the internal scientific/financial database.
"""
import asyncio
import signal
import sys
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hybrid_rag.document_loader import DocumentLoaderUtility
from src.hybrid_rag.utils import configure_logging
# THAY ĐỔI: Import BEIR Retriever mới
from src.hybrid_rag.hybrid_retriever import create_beir_hybrid_retriever

load_dotenv()

# Configure logging to suppress warnings
configure_logging()

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()

# Load configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = SCRIPT_DIR / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Global RAG system components
config = load_config()
embeddings = None
llm = None
vectorstore = None
rag_chain = None
documents = []

# Ingestion status tracking
ingestion_status = {
    "status": "not_started",  
    "progress": 0,  
    "current_file": "",
    "files_processed": 0,
    "total_files": 0,
    "documents_loaded": 0,
    "error_message": None,
    "stage": ""  
}
ingestion_task = None


def initialize_rag_system():
    """Initialize the RAG system with embeddings and LLM."""
    global embeddings, llm

    try:
        ollama_url = config['ollama'].get('base_url', 'http://localhost:11434')
        embedding_model = config['ollama']['embedding_model']

        embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)
        
        # LLM (OpenRouter)
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            model=config['openrouter']['model']
        )
        return True
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return False


def build_rag_chain():
    """Build the RAG chain with BEIR Hybrid Retrieval (BM25+ & Vector)."""
    global vectorstore, rag_chain, documents

    persist_dir = config['vector_store']['persist_directory']
    if not Path(persist_dir).is_absolute():
        persist_dir = str(SCRIPT_DIR / persist_dir)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("♻️ Loading existing vector store...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        if not documents:
            raise ValueError("No documents loaded. Please ingest documents first.")
        print("🏗️ Creating new vector store...")
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)

    print("🔧 Constructing BEIR Hybrid Retriever...")
    hybrid_retriever = create_beir_hybrid_retriever(
        documents=documents,
        vectorstore=vectorstore,
        config=config
    )

    # THAY ĐỔI: Prompt dành cho chuyên gia khoa học/tài chính
    prompt = ChatPromptTemplate.from_template("""
You are an expert scientific and financial analyst. Answer the user's question based ONLY on the provided context.
If the context does not contain the exact answer, state clearly that the information is not available. 
Do not hallucinate facts.

<context>
{context}
</context>

Question: {input}
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(hybrid_retriever, document_chain)


def update_ingestion_progress(current: int, total: int, filename: str):
    """Callback to update ingestion progress."""
    global ingestion_status
    ingestion_status["files_processed"] = current
    ingestion_status["total_files"] = total
    ingestion_status["current_file"] = filename
    file_progress = int((current / total) * 80) if total > 0 else 0
    ingestion_status["progress"] = file_progress


async def ingest_documents_async():
    """Async function to ingest BEIR documents (.jsonl) in the background."""
    global documents, ingestion_status

    try:
        ingestion_status.update({
            "status": "in_progress",
            "progress": 0,
            "current_file": "",
            "files_processed": 0,
            "total_files": 0,
            "documents_loaded": 0,
            "error_message": None,
            "stage": "loading_files"
        })

        data_dir = config['data']['directory']
        if not Path(data_dir).is_absolute():
            data_dir = str(SCRIPT_DIR / data_dir)

        loader = DocumentLoaderUtility(data_dir, config=config)

        documents = await asyncio.to_thread(
            loader.load_documents,
            progress_callback=update_ingestion_progress
        )

        if not documents:
            ingestion_status.update({
                "status": "failed",
                "error_message": "No JSONL documents found in data directory",
                "progress": 0
            })
            return

        ingestion_status.update({
            "progress": 80,
            "stage": "building_index",
            "documents_loaded": len(documents)
        })

        await asyncio.to_thread(build_rag_chain)

        ingestion_status.update({
            "status": "completed",
            "progress": 100,
            "stage": "completed"
        })

    except Exception as e:
        ingestion_status.update({
            "status": "failed",
            "error_message": str(e),
            "progress": 0
        })


async def ingest_documents() -> Dict[str, Any]:
    global ingestion_task, ingestion_status

    if ingestion_status["status"] == "in_progress":
        return {
            "success": False,
            "message": "Ingestion already in progress",
            "progress": ingestion_status["progress"]
        }

    ingestion_task = asyncio.create_task(ingest_documents_async())

    return {
        "success": True,
        "message": "Document ingestion started. Use get_ingestion_status to monitor progress.",
        "status": "in_progress"
    }


async def query_documents(query: str) -> Dict[str, Any]:
    global rag_chain

    if rag_chain is None:
        return {
            "success": False,
            "answer": "RAG system not initialized. Please run ingest_documents tool first.",
            "context": []
        }

    try:
        response = await asyncio.to_thread(rag_chain.invoke, {"input": query})

        # Cập nhật context format để lấy ID và Hybrid Score chuẩn BEIR
        context = [
            {
                "content": doc.page_content,
                "doc_id": doc.metadata.get('doc_id', 'unknown'),
                "score": doc.metadata.get('hybrid_score', 'N/A')
            }
            for doc in response['context']
        ]

        return {
            "success": True,
            "answer": response['answer'],
            "context": context
        }
    except Exception as e:
        return {
            "success": False,
            "answer": f"Error processing query: {str(e)}",
            "context": []
        }


# Create MCP server
server = Server("hybrid-rag-mcp")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools for Claude to use."""
    return [
        Tool(
            name="ingest_documents",
            description="Start loading and indexing JSONL documents from the data directory asynchronously.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="get_ingestion_status",
            description="Get the current status and progress of document ingestion.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="query_documents",
            description="Query the scientific/financial database using advanced hybrid search (BM25+ & Vector).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The exact question or keywords to search for in the database."
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_status",
            description="Get system status, configuration, and number of loaded documents.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from Claude."""
    global ingestion_status

    if name == "ingest_documents":
        result = await ingest_documents()
        return [TextContent(type="text", text=f"Ingestion {'started' if result['success'] else 'failed'}.\nMessage: {result['message']}")]

    elif name == "get_ingestion_status":
        status = ingestion_status.copy()
        if status["status"] == "not_started":
            status_text = "Ingestion Status: Not started. Use ingest_documents to begin."
        elif status["status"] == "in_progress":
            status_text = f"Status: In Progress ({status['progress']}%)\nStage: {status['stage']}\nLoaded: {status['documents_loaded']}"
        elif status["status"] == "completed":
            status_text = f"Status: Completed ✅\nDocuments Loaded: {status['documents_loaded']}\nYou can now use query_documents."
        elif status["status"] == "failed":
            status_text = f"Status: Failed ❌\nError: {status['error_message']}"
        else:
            status_text = f"Unknown status: {status['status']}"

        return [TextContent(type="text", text=status_text)]

    elif name == "query_documents":
        query = arguments.get("query", "")
        if not query:
            return [TextContent(type="text", text="Error: Query parameter is required")]

        result = await query_documents(query)
        if not result['success']:
            return [TextContent(type="text", text=f"Error: {result['answer']}")]

        # Format kết quả cho Claude
        context_text = "\n\n".join([
            f"[DocID: {ctx['doc_id']} | Score: {ctx['score']}]\n{ctx['content']}"
            for ctx in result['context']
        ])

        response_text = f"Answer: {result['answer']}\n\n---\n\nContext retrieved from Hybrid DB:\n{context_text}"
        return [TextContent(type="text", text=response_text)]

    elif name == "get_status":
        status = {
            "rag_initialized": rag_chain is not None,
            "documents_loaded": len(documents),
            "embedding_model": config.get('ollama', {}).get('embedding_model', 'N/A'),
            "llm_model": config.get('openrouter', {}).get('model', 'N/A')
        }
        status_text = "\n".join([f"{k}: {v}" for k, v in status.items()])
        return [TextContent(type="text", text=f"RAG System Status:\n{status_text}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def cleanup():
    global ingestion_task, vectorstore
    try:
        print("\n🛑 Shutting down MCP server...")
    except (ValueError, OSError):
        pass  # stdout đã đóng, bỏ qua
    if ingestion_task and not ingestion_task.done():
        ingestion_task.cancel()
        try:
            await ingestion_task
        except asyncio.CancelledError:
            pass
    if vectorstore:
        vectorstore = None
    print("✅ Cleanup complete")


async def main():
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print("Initializing BEIR Hybrid RAG MCP Server...")
        if initialize_rag_system():
            print("✅ Models initialized successfully")
            
            # Tự động tải dữ liệu nếu đã có sẵn trong Chroma (Tránh việc Claude phải gọi Tool ingest)
            persist_dir = config['vector_store']['persist_directory']
            if not Path(persist_dir).is_absolute():
                persist_dir = str(SCRIPT_DIR / persist_dir)
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                print("♻️ Found existing DB, auto-building index...")
                # Lấy documents từ loader để khởi tạo BM25
                data_dir = str(SCRIPT_DIR / config['data']['directory'])
                global documents
                documents = DocumentLoaderUtility(data_dir, config).load_documents()
                build_rag_chain()
                print("✅ RAG Chain Ready.")

        async with stdio_server() as (read_stream, write_stream):
            server_task = asyncio.create_task(
                server.run(
                    read_stream, write_stream,
                    InitializationOptions(
                        server_name="hybrid-rag-mcp",
                        server_version="2.0.0",
                        capabilities=server.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={}),
                    ),
                )
            )

            shutdown_task = asyncio.create_task(shutdown_event.wait())
            done, pending = await asyncio.wait([server_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED)

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        print(f"\n❌ Server error: {e}")
    finally:
        await cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)