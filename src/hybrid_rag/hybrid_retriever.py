"""
Enhanced Hybrid Retriever for BEIR (SciFact/FiQA)
Integrates BM25+, Dense Vector Search, Alpha Weighting, and Pseudo-Relevance Feedback (PRF).
"""
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Plus
import numpy as np
from collections import Counter

# Import bộ tiền xử lý chuyên dụng vừa tạo
from .query_preprocessor import QueryPreprocessor

class BEIRHybridRetriever(BaseRetriever):
    """
    Hybrid Retriever kết hợp Dense (Chroma) và Sparse (BM25+) 
    sử dụng Alpha Weighting và thuật toán PRF.
    """
    
    # Cấu hình Pydantic cho phép các kiểu dữ liệu phức tạp
    class Config:
        arbitrary_types_allowed = True

    vectorstore: Any
    bm25_model: Any
    documents: List[Document]
    preprocessor: QueryPreprocessor
    
    # Các tham số thuật toán
    alpha: float = 0.3          # Trọng số của Dense (Vector). Sparse (BM25) = 1 - alpha
    top_k: int = 10             # Số lượng tài liệu trả về cuối cùng
    use_prf: bool = True        # Bật/Tắt Pseudo-Relevance Feedback
    prf_doc_count: int = 3      # Số tài liệu top đầu dùng để mở rộng truy vấn
    prf_term_count: int = 5     # Số từ khóa mới sẽ được thêm vào câu hỏi

    def _min_max_norm(self, scores_dict: Dict[str, float]) -> Dict[str, float]:
        """Chuẩn hóa điểm số về khoảng [0, 1] để hợp nhất"""
        if not scores_dict: 
            return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v: 
            return {k: 0.5 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # 1. TIỀN XỬ LÝ RIÊNG BIỆT CHO 2 LUỒNG
        dense_query = self.preprocessor.preprocess_for_dense(query)
        sparse_query_tokens = self.preprocessor.preprocess_for_bm25(query)

        # 2. LUỒNG DENSE (VECTOR SEARCH)
        # Lấy khoảng cách (distance) từ vectorstore. (Khoảng cách càng nhỏ càng giống)
        dense_results = self.vectorstore.similarity_search_with_score(dense_query, k=self.top_k * 2)
        
        # Đảo ngược khoảng cách thành điểm số (Score): 1 / (1 + distance)
        dense_scores_raw = {}
        for doc, distance in dense_results:
            doc_id = doc.metadata.get('doc_id')
            if doc_id:
                dense_scores_raw[doc_id] = 1.0 / (1.0 + distance)

        # 3. LUỒNG SPARSE (BM25+ VÀ PRF)
        if not sparse_query_tokens:
            bm25_scores_raw = np.zeros(len(self.documents))
        else:
            # Chạy BM25+ lần 1
            bm25_scores_raw = self.bm25_model.get_scores(sparse_query_tokens)
            
            # THUẬT TOÁN PRF (Pseudo-Relevance Feedback)
            if self.use_prf and max(bm25_scores_raw) > 0:
                # 3.1 Lấy Top K tài liệu liên quan nhất
                top_indices = np.argsort(bm25_scores_raw)[::-1][:self.prf_doc_count]
                
                # 3.2 Trích xuất toàn bộ token từ các tài liệu này
                prf_tokens = []
                for idx in top_indices:
                    doc_text = self.documents[idx].page_content
                    # Dùng chính bộ lọc BM25 để làm sạch tài liệu
                    tokens = self.preprocessor.preprocess_for_bm25(doc_text)
                    prf_tokens.extend(tokens)
                
                # 3.3 Lọc ra các từ khóa TỪ MỚI (chưa có trong query gốc) xuất hiện nhiều nhất
                q_set = set(sparse_query_tokens)
                new_terms = [t for t in prf_tokens if t not in q_set]
                most_common_terms = [term for term, count in Counter(new_terms).most_common(self.prf_term_count)]
                
                # 3.4 Mở rộng truy vấn và chấm điểm lại lần 2
                expanded_tokens = sparse_query_tokens + most_common_terms
                bm25_scores_raw = self.bm25_model.get_scores(expanded_tokens)

        # Ánh xạ điểm BM25 vào doc_id
        sparse_scores_raw = {}
        for idx, score in enumerate(bm25_scores_raw):
            if score > 0:
                doc_id = self.documents[idx].metadata.get('doc_id')
                if doc_id:
                    sparse_scores_raw[doc_id] = score

        # 4. CHUẨN HÓA VÀ HỢP NHẤT (ALPHA WEIGHTING FUSION)
        norm_dense = self._min_max_norm(dense_scores_raw)
        norm_sparse = self._min_max_norm(sparse_scores_raw)

        combined_scores = {}
        all_doc_ids = set(norm_dense.keys()).union(set(norm_sparse.keys()))
        
        for doc_id in all_doc_ids:
            d_score = norm_dense.get(doc_id, 0.0)
            s_score = norm_sparse.get(doc_id, 0.0)
            # Công thức Alpha:
            combined_scores[doc_id] = (self.alpha * d_score) + ((1.0 - self.alpha) * s_score)

        # 5. SẮP XẾP VÀ TRẢ VỀ KẾT QUẢ
        sorted_doc_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        
        # Tra cứu doc_id để lấy lại object Document gốc
        doc_lookup = {doc.metadata.get('doc_id'): doc for doc in self.documents if 'doc_id' in doc.metadata}
        
        final_docs = []
        for doc_id, final_score in sorted_doc_ids:
            if doc_id in doc_lookup:
                doc = doc_lookup[doc_id]
                # Lưu điểm số vào metadata để theo dõi
                doc.metadata['hybrid_score'] = round(final_score, 4)
                final_docs.append(doc)
                
        return final_docs


def create_beir_hybrid_retriever(
    documents: List[Document],
    vectorstore: Any,
    config: Dict[str, Any]
) -> BaseRetriever:
    """
    Hàm khởi tạo BEIR Hybrid Retriever
    """
    preprocessor = QueryPreprocessor()
    
    print("Đang khởi tạo mô hình BM25+ cho tài liệu...")
    # Tokenize toàn bộ document corpus cho BM25+
    corpus_tokens = [preprocessor.preprocess_for_bm25(doc.page_content) for doc in documents]
    
    # Khởi tạo thuật toán BM25+ (Tham số k1 và b mặc định của thư viện thường rất tối ưu)
    bm25_model = BM25Plus(corpus_tokens)
    
    # Đọc cấu hình (nếu không có thì dùng mặc định)
    retrieval_config = config.get('retrieval', {})
    top_k = retrieval_config.get('final_k', 10)
    alpha = retrieval_config.get('dense_weight', 0.3) # Sparse = 0.7
    use_prf = retrieval_config.get('use_prf', True)

    return BEIRHybridRetriever(
        vectorstore=vectorstore,
        bm25_model=bm25_model,
        documents=documents,
        preprocessor=preprocessor,
        alpha=alpha,
        top_k=top_k,
        use_prf=use_prf
    )