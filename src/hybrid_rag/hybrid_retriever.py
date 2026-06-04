"""
Advanced Hybrid Retriever cho BEIR (Đã sửa lỗi đồng bộ dữ liệu)
Tích hợp kỹ thuật tái cấu trúc văn bản gốc để đảm bảo điểm số logic tuyệt đối.
"""
import re
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Khởi tạo tài nguyên NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class IRSystem:
    def __init__(self, k1=1.0, b=0.45, fb_docs=3, fb_terms=5, alpha=0.5):
        self.k1 = k1
        self.b = b
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms
        self.alpha = alpha
        
        self.Index = {}
        self.idf = {}
        self.doc_lengths = {}
        self.avgdl = 0
        self.N = 0
        self.doc_tokens_map = {}
        
        self.ps = PorterStemmer()
        self.stoplist = set(stopwords.words("english"))
        self.puncts = set(['.', ',', ':', '`', '"', "'", '!', '?', "``", "''", '(', ')', '--', ';'])

    def preprocess(self, text):
        if not text: return []
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        tokens = word_tokenize(text)
        processed = []
        for tok in tokens:
            if tok not in self.stoplist and tok not in self.puncts and len(tok) > 2:
                processed.append(self.ps.stem(tok))
        return processed

    def index_documents(self, documents: List[Document]):
        """Đã bảo vệ tính toàn vẹn của ID: Không trùng lặp tài liệu gốc"""
        total_length = 0
        self.N = 0
        
        for doc in documents:
            self.N += 1
            docid = doc.metadata.get('doc_id')
            content = doc.page_content
            
            terms = self.preprocess(content)
            self.doc_tokens_map[docid] = terms
            
            doc_len = len(terms)
            self.doc_lengths[docid] = doc_len
            total_length += doc_len
            
            counts = {}
            for t in terms:
                counts[t] = counts.get(t, 0) + 1
                
            for t, tf in counts.items():
                if t not in self.Index:
                    self.Index[t] = {}
                self.Index[t][docid] = tf

        if self.N > 0:
            self.avgdl = total_length / self.N

        for term, postings in self.Index.items():
            df = len(postings)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def compute_bm25(self, q_tf_dict):
        scores = {}
        safe_avgdl = self.avgdl if self.avgdl > 0 else 1.0
        
        for t, q_weight in q_tf_dict.items():
            if t in self.Index:
                idf_val = self.idf[t]
                for docid, doc_tf in self.Index[t].items():
                    dl = self.doc_lengths[docid]
                    denom = self.k1 * (1 - self.b + self.b * (dl / safe_avgdl)) + doc_tf
                    score = (idf_val ** 2) * (doc_tf * (self.k1 + 1)) / denom * q_weight
                    scores[docid] = scores.get(docid, 0) + score
        return scores


class AdvancedHybridRetriever(BaseRetriever):
    class Config:
        arbitrary_types_allowed = True

    ir_system: Any
    vectorstore: Any
    documents: List[Document]
    
    dense_weight: float = 0.6
    top_k: int = 10
    use_prf: bool = True

    def _min_max_norm(self, scores_dict: Dict[str, float]) -> Dict[str, float]:
        if not scores_dict: return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v: return {k: 0.5 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # 1. LUỒNG SPARSE
        q_terms = self.ir_system.preprocess(query)
        q_original_tf = {}
        for t in q_terms:
            q_original_tf[t] = q_original_tf.get(t, 0) + 1

        pass_1_scores = self.ir_system.compute_bm25(q_original_tf)
        
        if self.use_prf and pass_1_scores:
            top_docs = sorted(pass_1_scores.items(), key=lambda x: x[1], reverse=True)[:self.ir_system.fb_docs]
            expansion_candidates = {}
            for docid, _ in top_docs:
                if docid not in self.ir_system.doc_tokens_map: continue
                doc_terms = self.ir_system.doc_tokens_map[docid]
                for t in doc_terms:
                    if t not in q_original_tf and t in self.ir_system.idf:
                        tf = self.ir_system.Index[t].get(docid, 0)
                        tf_weight = 1 + math.log10(tf) if tf > 0 else 0
                        expansion_candidates[t] = expansion_candidates.get(t, 0) + (tf_weight * self.ir_system.idf[t])

            top_expansion_terms = sorted(expansion_candidates.items(), key=lambda x: x[1], reverse=True)[:self.ir_system.fb_terms]
            q_expanded_tf = q_original_tf.copy()
            for term, _ in top_expansion_terms:
                q_expanded_tf[term] = self.ir_system.alpha
            sparse_scores_raw = self.ir_system.compute_bm25(q_expanded_tf)
        else:
            sparse_scores_raw = pass_1_scores

        # ĐĂC BIỆT: Nếu tắt Vector (dense_weight == 0), bỏ qua normalize để lấy thứ tự gốc 100%
        if self.dense_weight == 0:
            # Mở rộng kết quả tìm kiếm lên tối thiểu 100 để phục vụ BEIR chấm điểm MAP sâu
            search_k = max(self.top_k, 100)
            sorted_doc_ids = sorted(sparse_scores_raw.items(), key=lambda x: x[1], reverse=True)[:search_k]
            
            doc_lookup = {doc.metadata.get('doc_id'): doc for doc in self.documents if 'doc_id' in doc.metadata}
            final_docs = []
            for doc_id, final_score in sorted_doc_ids:
                if doc_id in doc_lookup:
                    doc = doc_lookup[doc_id]
                    doc.metadata['hybrid_score'] = round(final_score, 4)
                    final_docs.append(doc)
            return final_docs

        # 2. LUỒNG DENSE VÀ FUSION CHUẨN HÓA (KHI DENSE_WEIGHT > 0)
        dense_results = self.vectorstore.similarity_search_with_score(query, k=max(self.top_k * 2, 100))
        dense_scores_raw = {}
        for doc, distance in dense_results:
            doc_id = doc.metadata.get('doc_id')
            if doc_id:
                dense_scores_raw[doc_id] = 1.0 / (1.0 + distance)

        norm_sparse = self._min_max_norm(sparse_scores_raw)
        norm_dense = self._min_max_norm(dense_scores_raw)

        combined_scores = {}
        all_doc_ids = set(norm_sparse.keys()).union(set(norm_dense.keys()))
        for doc_id in all_doc_ids:
            s_score = norm_sparse.get(doc_id, 0.0)
            d_score = norm_dense.get(doc_id, 0.0)
            combined_scores[doc_id] = (self.dense_weight * d_score) + ((1.0 - self.dense_weight) * s_score)

        search_k = max(self.top_k, 100)
        sorted_doc_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:search_k]
        
        doc_lookup = {doc.metadata.get('doc_id'): doc for doc in self.documents if 'doc_id' in doc.metadata}
        final_docs = []
        for doc_id, final_score in sorted_doc_ids:
            if doc_id in doc_lookup:
                doc = doc_lookup[doc_id]
                doc.metadata['hybrid_score'] = round(final_score, 4)
                final_docs.append(doc)
        return final_docs


def create_beir_hybrid_retriever(
    documents: List[Document],
    vectorstore: Any,
    config: Dict[str, Any]
) -> BaseRetriever:
    
    print("🛠️ Đang kích hoạt tiến trình Tái cấu trúc tài liệu gốc (Document Reconstruction)...")
    
    # Gom các chunks của LangChain có cùng doc_id lại thành 1 văn bản lớn duy nhất để tính N và avgdl chuẩn xác
    doc_reconstruct_map = {}
    for chunk in documents:
        did = chunk.metadata.get('doc_id')
        if did:
            if did not in doc_reconstruct_map:
                doc_reconstruct_map[did] = []
            doc_reconstruct_map[did].append(chunk.page_content)
            
    reconstructed_documents = []
    for did, contents in doc_reconstruct_map.items():
        # Nối lại nội dung nguyên bản
        full_content = " ".join(contents)
        reconstructed_documents.append(Document(page_content=full_content, metadata={"doc_id": did}))
        
    print(f" -> Đã gom {len(documents)} chunks về {len(reconstructed_documents)} tài liệu nguyên gốc.")
    
    prf_config = config.get('prf_settings', {})
    ir = IRSystem(
        k1=prf_config.get('k1', 1.0),
        b=prf_config.get('b', 0.45),
        fb_docs=prf_config.get('fb_docs', 3),
        fb_terms=prf_config.get('fb_terms', 5),
        alpha=prf_config.get('alpha', 0.5)
    )
    
    # Nạp tài liệu nguyên bản vào bộ Index của IRSystem để trùng khớp chỉ số siêu tham số
    ir.index_documents(reconstructed_documents)
    
    retrieval_config = config.get('retrieval', {})
    top_k = retrieval_config.get('final_k', 10)
    dense_weight = retrieval_config.get('dense_weight', 0.6)
    use_prf = retrieval_config.get('use_prf', True)

    return AdvancedHybridRetriever(
        ir_system=ir,
        vectorstore=vectorstore,
        documents=documents, # Giữ list gốc cho luồng Vector Store
        dense_weight=dense_weight,
        top_k=top_k,
        use_prf=use_prf
    )