# 🚀 BEIR Hybrid RAG System & MCP Server

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Introduction
Đây là một hệ thống Hybrid RAG (Kết hợp Semantic Vector Search và Keyword BM25+) được tối ưu hóa riêng cho các tập dữ liệu chuẩn BEIR (như SciFact, FiQA). Hệ thống tích hợp sẵn giao thức Model Context Protocol (MCP), cho phép Claude Desktop phân tích dữ liệu nội bộ.
---

## 🛠 1. Yêu cầu hệ thống (Prerequisites)

Trước khi chạy project, bạn cần tải và cài đặt 2 phần mềm lõi sau:
- **Ollama Desktop**: Dùng để chạy mô hình Embedding cục bộ (Local GPU/CPU) - Tải tại: ollama.com
- **Claude Desktop**: Dùng để làm giao diện Chat UI và LLM Agent thông qua giao thức MCP. - Tải tại: claude.ai/download

## 💻 2. Cài đặt và Khởi tạo môi trường
Mở terminal (khuyến nghị dùng PowerShell trên Windows) tại thư mục gốc của dự án và chạy lần lượt các lệnh sau:

```bash
# 1. Tạo môi trường ảo (Virtual Environment)
python -m venv venv

# 2. Kích hoạt môi trường ảo
.\venv\Scripts\Activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 4. Khai báo API Key của OpenRouter (Dùng cho LLM sinh câu trả lời)
# Lưu ý: Thay chuỗi "sk-or-v1-PUTYOURAPIKEYHERE" bằng key thật của bạn
$env:OPENROUTER_API_KEY="sk-or-v1-PUTYOURAPIKEYHERE"
```

## 🧠 3. Tải và Cấu hình Model
Hệ thống sử dụng model nomic-embed-text cực kỳ nhẹ và hiệu quả cho việc chuyển đổi văn bản thành Vector.

```bash
# Tải model embedding về máy thông qua Ollama
ollama pull nomic-embed-text

# Kiểm tra xem model đã được tải thành công chưa
ollama list
```
(Nếu bạn thấy nomic-embed-text trong danh sách trả về nghĩa là đã thành công).

## 🏃 4. Chạy các Demos kiểm thử
Trước khi cắm vào Claude Desktop, bạn có thể test luồng RAG trực tiếp qua terminal. Đảm bảo bạn đã bỏ file corpus.jsonl vào thư mục data/ trước khi chạy.

Cách 1: Chạy luồng cơ bản (Hỏi - Đáp đơn lẻ) - (Lưu ý: Lần chạy đầu tiên sẽ mất thời gian để hệ thống đọc file jsonl và ghi dữ liệu Embedding xuống ổ cứng).
```bash
python scripts/demos/basic.py
```
Cách 1: Cách 2: Chạy luồng hội thoại (Có lưu lịch sử chat)
```bash
python scripts/demos/conversational.py
```
## 🔗 5. Tích hợp hệ thống vào Claude Desktop (MCP Server)
Đây là bước cho phép ứng dụng Claude Desktop của bạn tự động tra cứu cơ sở dữ liệu BEIR này.

Bước 1: Đảm bảo môi trường ảo (venv) đã có đủ thư viện, và code của bạn nằm cố định ở một thư mục.
Bước 2: Mở file cấu hình của Claude Desktop. Nhấn tổ hợp phím Win + R, dán đường dẫn sau và nhấn Enter:

```bash
%APPDATA%\Claude\claude_desktop_config.json
```
Bước 3: Cập nhật nội dung file JSON như sau. LƯU Ý QUAN TRỌNG: Bạn phải thay thế C:\\Path\\To\\Your\\Project bằng đường dẫn tuyệt đối đến thư mục chứa project của bạn trên máy tính.
```json
{
  "mcpServers": {
    "beir_hybrid_search": {
      "command": "C:\\Path\\To\\Your\\Project\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\Path\\To\\Your\\Project\\scripts\\mcp\\server_claude.py"
      ]
    }
  }
}
```
Bước 4: Khởi động lại ứng dụng Claude Desktop.
Nếu cấu hình đúng, bạn sẽ thấy biểu tượng cái búa (Tools) hiển thị trong khung chat của Claude. Bắt đầu chat với Claude về các dữ liệu có trong corpus.jsonl, Claude sẽ tự động gọi hệ thống Python của bạn để tìm kiếm!

## License

This project is provided as-is for educational and demonstration purposes.

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [End to end Hybrid Rag Project](https://github.com/gwyer/hybrid-rag-project)
