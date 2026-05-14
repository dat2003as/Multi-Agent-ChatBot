# Tài liệu Kỹ thuật: AI Chatbot

Tài liệu này cung cấp cái nhìn chi tiết về kiến trúc, luồng dữ liệu và các thành phần kỹ thuật của hệ thống AI Chatbot.

## 1. Tổng quan Hệ thống (System Overview)

AI Chatbot là một hệ thống RAG (Retrieval-Augmented Generation) được thiết kế để hỗ trợ quản lý trang trại thủy sản. Hệ thống sử dụng mô hình ngôn ngữ lớn (LLM) kết hợp với cơ sở dữ liệu tri thức nội bộ để cung cấp câu trả lời chính xác và có ngữ cảnh.

**Thành phần chính:**
- **Backend:** FastAPI (Python)
- **Cơ sở dữ liệu Vector:** PostgreSQL + `pgvector`
- **LLM & Embedding:** Azure OpenAI (GPT-4o, text-embedding-3-small)
- **Bộ nhớ hội thoại:** Redis (lưu trữ lịch sử chat theo session)
- **Xử lý tài liệu:** PyMuPDF (PDF), python-docx (Word)

---

## 2. Luồng xử lý Chat (Chat Processing Flow)

Luồng này bắt đầu khi người dùng gửi tin nhắn từ giao diện người dùng (Frontend).

### Sơ đồ quy trình:
1. **API Endpoint (`POST /chat`):** Nhận `message`, `session_id`, và `farm_id`.
2. **Session Management:** Truy xuất lịch sử hội thoại (history) từ Redis dựa trên `session_id`.
3. **System Prompt Building:** Tạo prompt hệ thống kết hợp với thông tin trang trại (`farm_id`) để định hướng phản hồi của LLM.
4. **Orchestrator Control:**
    - Gửi tin nhắn đến LLM cùng với danh sách các "Tools" khả dụng.
    - LLM quyết định xem có cần sử dụng công cụ tìm kiếm dữ liệu (`search_documents`) hay trả lời trực tiếp.
5. **Tool Execution (RAG):**
    - Nếu LLM gọi `search_documents`, hệ thống sẽ chuyển câu truy vấn (query) thành vector embedding.
    - Thực hiện tìm kiếm vector tương đồng trên bảng `doc_embeddings` trong PostgreSQL.
    - Trả về các đoạn văn bản (chunks) liên quan nhất cho LLM.
6. **Response Synthesis:** LLM tổng hợp thông tin từ kiến thức sẵn có và các đoạn văn bản tìm được để tạo ra câu trả lời cuối cùng.
7. **History Update:** Lưu tin nhắn của người dùng và phản hồi của chatbot vào Redis.
8. **Final Answer:** Trả về kết quả cho người dùng.

---

## 3. Luồng Embedding Dữ liệu (Data Embedding Flow)

Đây là quy trình đưa tri thức từ tài liệu (SOP, FAQ, hướng dẫn...) vào cơ sở dữ liệu vector.

### Các bước thực hiện:
1. **Tiếp nhận tài liệu (`POST /ingest`):** Quản trị viên tải lên các file (.pdf, .docx, .md, .txt, .xlsx).
2. **Xử lý Xóa cũ (Soft-delete):** Hệ thống tự động đánh dấu `is_deleted = TRUE` cho các chunks cũ thuộc cùng một file để tránh trùng lặp thông tin khi cập nhật.
3. **Trích xuất văn bản (Parsing):**
    - Sử dụng các loaders chuyên dụng để đọc nội dung từ file.
    - Làm sạch văn bản (loại bỏ ký tự thừa, chuẩn hóa định dạng).
4. **Chia nhỏ văn bản (Chunking):**
    - Sử dụng `Chunker` để chia văn bản thành các đoạn (chunks) nhỏ dựa trên đoạn văn hoặc tiêu đề markdown.
    - Kích thước mục tiêu: **400 tokens** mỗi đoạn.
    - Kích thước tối đa: **512 tokens**.
    - Độ chồng lấp (overlap): **60 tokens** giữa các đoạn để duy trì ngữ cảnh liên tục.
5. **Tạo Vector Embedding:**
    - Mỗi đoạn văn bản được gửi đến OpenAI API (`text-embedding-3-small`) để chuyển hóa thành một mảng số thực (vector) 1536 chiều.
6. **Lưu trữ vào Database:**
    - Lưu các thông tin sau vào bảng `doc_embeddings`:
        - `farm_id`: Định danh trang trại (nếu có).
        - `source_file`: Tên file gốc.
        - `source_type`: Loại tài liệu (ví dụ: `sop`, `faq`).
        - `chunk_text`: Nội dung văn bản của đoạn.
        - `embedding`: Vector nhúng (lưu dưới kiểu dữ liệu `vector` của pgvector).
        - `metadata`: JSON chứa thông tin bổ sung.

---

## 4. Cơ sở dữ liệu (Database Schema)

### Bảng `doc_embeddings`
Phục vụ lưu trữ tri thức và tìm kiếm vector.

| Cột | Kiểu dữ liệu | Ràng buộc | Mô tả |
| :--- | :--- | :--- | :--- |
| `id` | UUID | PRIMARY KEY | Khóa chính |
| `farm_id` | UUID | NULLABLE | Liên kết với trang trại cụ thể |
| `source_file` | TEXT | NOT NULL | Tên tập tin gốc |
| `source_type` | TEXT | NOT NULL | Phân loại: sop, faq, manual, guide |
| `chunk_index` | INTEGER | NOT NULL | Thứ tự đoạn trong file |
| `chunk_text` | TEXT | NOT NULL | Nội dung văn bản của đoạn |
| `chunk_tokens` | INTEGER | NOT NULL | Số lượng token của đoạn |
| `embedding` | VECTOR(1536) | NOT NULL | Vector nhúng 1536 chiều |
| `metadata` | JSONB | NOT NULL | Metadata dạng JSON |
| `is_deleted` | BOOLEAN | DEFAULT FALSE | Cờ xóa mềm |

---

## 5. Các Công cụ (Tools) của AI

Chatbot được trang bị các kỹ năng (functions) để giải quyết vấn đề:

- **`search_documents(query)`**: Tìm kiếm trong kho tri thức nội bộ. Dùng cho các câu hỏi về quy trình trang trại, hướng dẫn thiết bị, FAQ kỹ thuật.
- **`answer_general(query)`**: Giải đáp các kiến thức nuôi trồng thủy sản chung, tư vấn triệu chứng bệnh thông thường hoặc các cuộc hội thoại xã giao.

---

## 6. Ghi chú Bảo mật & Hiệu năng

- **CORS:** Chỉ cho phép các domain được cấu hình truy cập API.
- **Async Processing:** Toàn bộ luồng I/O (Database, API Call) đều xử lý bất đồng bộ để tối ưu hiệu suất.
- **Vector Index:** Sử dụng index `ivfflat` hoặc `hnsw` trên cột `embedding` để tăng tốc độ truy vấn khi dữ liệu lớn.
