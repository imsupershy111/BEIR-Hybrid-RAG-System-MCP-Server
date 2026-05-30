"""
Document Loader Utility
Loads documents specifically from BEIR format (.jsonl) for FiQA/SciFact datasets.
"""
import os
import json
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class BEIRJsonlLoader:
    """Loader dành riêng cho file corpus.jsonl của chuẩn BEIR"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        docs = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # Bóc tách dữ liệu theo chuẩn BEIR
                doc_id = str(data.get("_id", ""))
                title = data.get("title", "")
                text = data.get("text", "")
                
                # Nối title và text
                full_text = f"{title}. {text}" if title else text
                
                # Lưu ý cực kỳ quan trọng: Phải giữ lại 'doc_id' trong metadata
                docs.append(Document(
                    page_content=full_text, 
                    metadata={"doc_id": doc_id}
                ))
        return docs

class DocumentLoaderUtility:
    """Utility class to load BEIR documents (.jsonl)."""

    def __init__(self, data_directory: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document loader.
        """
        self.data_directory = Path(data_directory)
        self.config = config or {}

        # Get chunking config
        doc_config = self.config.get('document_processing', {})
        self.text_chunk_size = doc_config.get('text_chunk_size', 1000)
        self.text_chunk_overlap = doc_config.get('text_chunk_overlap', 200)

        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.text_chunk_size,
            chunk_overlap=self.text_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def count_files(self) -> int:
        """Count the number of .jsonl files in the data directory."""
        if not self.data_directory.exists():
            return 0
        return sum(1 for f in self.data_directory.rglob('*.jsonl') if f.is_file())

    def load_documents(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> List[Document]:
        """
        Load all .jsonl documents from the data directory.
        """
        if not self.data_directory.exists():
            print(f"Warning: Data directory '{self.data_directory}' does not exist.")
            return []

        supported_files = [f for f in self.data_directory.rglob('*.jsonl') if f.is_file()]
        total_files = len(supported_files)
        documents = []
        files_loaded = 0

        for idx, file_path in enumerate(supported_files, 1):
            try:
                # Report progress
                if progress_callback:
                    progress_callback(idx, total_files, file_path.name)

                # Nạp file jsonl
                loader = BEIRJsonlLoader(str(file_path))
                loaded_docs = loader.load()

                # Cắt chunk
                chunked_docs = self.text_splitter.split_documents(loaded_docs)

                # Bổ sung metadata về nguồn gốc file
                for doc in chunked_docs:
                    doc.metadata['source_file'] = file_path.name
                    doc.metadata['file_type'] = '.jsonl'
                    doc.metadata['doc_category'] = 'text'

                documents.extend(chunked_docs)
                files_loaded += 1
                print(f"✅ Loaded: {file_path.name} ({idx}/{total_files})")
                
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")

        print(f"\n📚 Total files loaded: {files_loaded}")
        print(f"📄 Total chunks created: {len(documents)}")

        return documents

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.jsonl']