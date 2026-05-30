"""
Query Preprocessor for Scientific and Financial IR

This module handles text normalization, tokenization, stemming, 
and bigram generation for datasets like SciFact and FiQA.
"""
import re
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Đảm bảo đã tải các gói cần thiết của NLTK
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

class QueryPreprocessor:
    """Preprocesses queries and documents for Hybrid Retrieval."""

    def __init__(self):
        """Initialize the NLP tools."""
        self.ps = PorterStemmer()
        self.stoplist = set(stopwords.words("english"))
        
        # Bạn có thể thêm các stopword đặc thù của dạng câu hỏi QA vào đây
        # Ví dụ: loại bỏ các từ mang tính chất hỏi han không mang ngữ nghĩa tài liệu
        question_stopwords = {'what', 'how', 'why', 'who', 'when', 'where', 'show', 'me', 'is', 'the', 'a', 'an'}
        self.stoplist.update(question_stopwords)

    def preprocess_for_bm25(self, text: str) -> List[str]:
        """
        Tiền xử lý văn bản thành list các tokens (Unigrams + Bigrams)
        Dành riêng cho hệ thống Sparse Retriever (BM25).

        Args:
            text: Original query or document string

        Returns:
            List of processed unigrams and bigrams
        """
        if not text:
            return []

        # THAY ĐỔI CỐT LÕI: Giữ lại chữ, số, dấu gạch ngang (-), đô la ($), phần trăm (%)
        # Rất quan trọng cho SciFact (mass-energy) và FiQA ($AAPL, 5%)
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s\-\$%]', ' ', text.lower())
        tokens = word_tokenize(cleaned_text)

        unigrams = []
        for tok in tokens:
            # Giữ lại từ > 1 ký tự và không nằm trong stoplist
            if tok not in self.stoplist and len(tok) > 1:
                unigrams.append(self.ps.stem(tok))

        # Sinh Bigrams để giữ ngữ cảnh (VD: "interest_rate", "quantum_mechanic")
        bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]

        return unigrams + bigrams

    def preprocess_for_dense(self, text: str) -> str:
        """
        Tiền xử lý nhẹ nhàng cho Dense Retriever (Vector/Embedding).
        Vector Models cần câu văn gần với tự nhiên nhất để hiểu ngữ nghĩa.

        Args:
            text: Original query string

        Returns:
            Cleaned string
        """
        if not text:
            return ""
        
        # Chỉ xóa các khoảng trắng thừa hoặc ký tự rác quá đặc biệt, 
        # giữ nguyên ngữ pháp để model tự hiểu
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()


# ==========================================
# Example usage & Testing
# ==========================================
if __name__ == "__main__":
    print("=== Query Preprocessor Test for BEIR ===\n")
    preprocessor = QueryPreprocessor()

    # Test 1: SciFact
    scifact_query = "Who developed the mass-energy equivalence formula E=mc2?"
    print(f"Original SciFact: {scifact_query}")
    print(f" -> BM25 Tokens:  {preprocessor.preprocess_for_bm25(scifact_query)}")
    print(f" -> Dense String: {preprocessor.preprocess_for_dense(scifact_query)}\n")

    # Test 2: FiQA
    fiqa_query = "What is the expected impact of a 5% interest rate hike on $AAPL dividend?"
    print(f"Original FiQA:    {fiqa_query}")
    print(f" -> BM25 Tokens:  {preprocessor.preprocess_for_bm25(fiqa_query)}")
    print(f" -> Dense String: {preprocessor.preprocess_for_dense(fiqa_query)}")