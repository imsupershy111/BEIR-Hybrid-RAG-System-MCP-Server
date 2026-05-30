#!/usr/bin/env python3
"""
Conversational BEIR Hybrid RAG Demo
Interactive CLI with conversation history and context memory for SciFact/FiQA.
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.hybrid_rag.document_loader import DocumentLoaderUtility
from src.hybrid_rag.utils import configure_logging
# THAY ĐỔI: Import BEIR Retriever thay vì Document Type Aware Retriever
from src.hybrid_rag.hybrid_retriever import create_beir_hybrid_retriever
from src.hybrid_rag.query_preprocessor import QueryPreprocessor


class ConversationalRAG:
    """Conversational RAG system with memory for BEIR Datasets."""

    def __init__(self):
        """Initialize the RAG system."""
        print("=" * 70)
        print("🚀 CONVERSATIONAL BEIR HYBRID RAG SYSTEM")
        print("=" * 70)
        print("\n💡 This version maintains conversation history!")
        print("   Follow-up questions will reference previous answers.\n")
        print("Initializing system...\n")

        # Configure logging
        configure_logging()

        # Load configuration
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        print("✅ Configuration loaded")

        # Initialize conversation history
        self.chat_history = ChatMessageHistory()
        self.conversation_count = 0

        # Initialize query preprocessor
        self.query_preprocessor = QueryPreprocessor()
        print("✅ Query preprocessor ready (NLP filtering enabled)")

        # Initialize components
        self._load_documents()
        self._initialize_models()
        self._create_vector_store()
        self._create_retriever()
        self._create_conversational_chain()

        print("\n" + "=" * 70)
        print("✅ SYSTEM READY - You can now have conversations!")
        print("=" * 70)

    def _load_documents(self):
        """Load documents from data directory."""
        data_dir = self.config['data']['directory']
        data_path = Path(__file__).parent.parent.parent / data_dir

        print(f"📂 Loading BEIR documents from: {data_path}")

        loader = DocumentLoaderUtility(str(data_path), config=self.config)
        self.documents = loader.load_documents()

        if not self.documents:
            print(f"\n⚠️  No documents found in '{data_path}'")
            print(f"⚠️  Supported formats: {', '.join(loader.get_supported_formats())}")
            sys.exit(1)

        sources = [doc.metadata.get('source_file', '') for doc in self.documents]
        unique_files = set([s for s in sources if s])

        print(f"✅ Loaded {len(self.documents)} chunks from {len(unique_files)} files")

    def _initialize_models(self):
        """Initialize LLM and Embedding models"""
        self.embedding_model_name = self.config['ollama']['embedding_model']
        self.llm_model_name = self.config['openrouter']['model']

        # Embedding
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model_name,
            base_url=self.config['ollama'].get('base_url', 'http://localhost:11434')
        )
    
        # Chat LLM
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            model=self.llm_model_name
        )

    def _create_vector_store(self):
        """Create or load vector store."""
        persist_dir = Path(__file__).parent.parent.parent / self.config['vector_store']['persist_directory']

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print("♻️ Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=str(persist_dir), 
                embedding_function=self.embeddings
            )
        else:
            print("🏗️ Creating new vector store...")
            self.vectorstore = Chroma.from_documents(
                self.documents, 
                self.embeddings, 
                persist_directory=str(persist_dir)
            )

        print(f"✅ Vector store ready")

    def _create_retriever(self):
        """Create BEIR hybrid retriever."""
        print("🔧 Creating BEIR hybrid retriever (BM25+ & Vector Fusion)...")

        self.retriever = create_beir_hybrid_retriever(
            documents=self.documents,
            vectorstore=self.vectorstore,
            config=self.config
        )

        print("✅ Hybrid retriever ready with PRF enabled")

    def _create_conversational_chain(self):
        """Create conversational QA chain with memory."""
        # THAY ĐỔI: Persona chuyên gia khoa học / tài chính
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert scientific and financial analyst with access to a specific document database.
Answer questions based ONLY on the provided context and conversation history.

When answering follow-up questions:
- Reference previous answers when relevant
- Maintain context across the conversation
- Use precise terminology found in the documents
- If the context does not contain the answer, state clearly that the information is not available.
Do not hallucinate facts."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context: {context}"),
            ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain(self.llm, prompt)
        self.qa_chain = create_retrieval_chain(self.retriever, document_chain)

        print("✅ Conversational chain constructed with memory")

    def query(self, question: str, show_sources: bool = True):
        """Ask a question with conversation context."""
        try:
            # Convert chat history to proper format
            history_messages = []
            for msg in self.chat_history.messages:
                history_messages.append(msg)

            # THAY ĐỔI: Không gọi expand_query nữa, truyền thẳng vào RAG
            response = self.qa_chain.invoke({
                "input": question,
                "chat_history": history_messages
            })

            # Add to conversation history
            self.chat_history.add_user_message(question)
            self.chat_history.add_ai_message(response['answer'])
            self.conversation_count += 1

            if show_sources:
                print("\n📚 Sources (Top Hybrid Results):")
                sources_seen = set()
                # Hiển thị metadata theo chuẩn BEIR
                for i, doc in enumerate(response.get('context', [])[:5], 1):
                    doc_id = doc.metadata.get('doc_id', 'unknown')
                    score = doc.metadata.get('hybrid_score', 'N/A')
                    
                    if doc_id not in sources_seen:
                        sources_seen.add(doc_id)
                        print(f"   [{i}] Doc ID: {doc_id} | Score: {score}")

            return response

        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
            return None

    def clear_history(self):
        self.chat_history.clear()
        self.conversation_count = 0
        print("\n🔄 Conversation history cleared!\n")

    def show_history(self):
        if not self.chat_history.messages:
            print("\n📝 No conversation history yet.\n")
            return

        print("\n📝 CONVERSATION HISTORY:")
        print("-" * 70)

        for i, msg in enumerate(self.chat_history.messages):
            if isinstance(msg, HumanMessage):
                print(f"\n[{i//2 + 1}] 👤 You: {msg.content}")
            elif isinstance(msg, AIMessage):
                content = msg.content
                if len(content) > 300:
                    content = content[:300] + "..."
                print(f"    🤖 Analyst: {content}")

        print("\n" + "-" * 70 + "\n")

    def interactive_mode(self):
        print("\n💬 CONVERSATIONAL MODE")
        print("   • Ask follow-up questions - I'll remember the context!")
        print("   • Type 'exit' or 'quit' to stop")
        print("   • Type 'help' for example questions")
        print("   • Type 'history' to see conversation history")
        print("   • Type 'clear' to start a new conversation")
        print("   • Type 'stats' for system statistics\n")

        while True:
            try:
                if self.conversation_count > 0:
                    print(f"[Turn {self.conversation_count + 1}]")

                question = input("❓ Your question: ").strip()

                if not question:
                    continue

                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                elif question.lower() == 'help':
                    self._show_help()
                    continue
                elif question.lower() == 'stats':
                    self._show_stats()
                    continue
                elif question.lower() == 'history':
                    self.show_history()
                    continue
                elif question.lower() == 'clear':
                    self.clear_history()
                    continue

                print("\n🤔 Analyzing...")
                response = self.query(question, show_sources=True)

                if response:
                    print(f"\n💡 Answer:\n{response['answer']}\n")
                    print("-" * 70)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def _show_help(self):
        """Show example questions relevant to BEIR datasets."""
        print("\n📖 EXAMPLE CONVERSATIONS:")
        print()
        print("SciFact Examples:")
        print("  You: Who developed the theory of relativity?")
        print("  AI: Albert Einstein developed the theory of relativity...")
        print("  You: What was his famous formula?")
        print("  AI: His famous formula is E=mc2, which represents mass-energy equivalence.")
        print()
        print("FiQA Examples:")
        print("  You: What happens to bonds when interest rates rise?")
        print("  AI: When interest rates rise, existing bond prices typically fall...")
        print("  You: How does this affect dividend stocks?")
        print("  AI: High interest rates can make dividend stocks less attractive...")
        print()

    def _show_stats(self):
        """Show system statistics."""
        print("\n📊 SYSTEM STATISTICS:")
        print()

        print(f"Documents:")
        print(f"  • Total chunks indexed: {len(self.documents)}")
        print()
        print(f"Conversation:")
        print(f"  • Messages exchanged: {len(self.chat_history.messages)}")
        print(f"  • Questions asked: {self.conversation_count}")
        print()
        print(f"Models:")
        print(f"  • Embedding Model: {self.embedding_model_name}")
        print(f"  • Chat LLM: {self.llm_model_name}")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Conversational BEIR Hybrid RAG Demo with Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    args = parser.parse_args()

    try:
        rag = ConversationalRAG()
        rag.interactive_mode()
    except Exception as e:
        print(f"\n❌ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()