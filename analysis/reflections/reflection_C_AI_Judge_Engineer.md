# Báo cáo Cá nhân - Thành viên 3: AI Judge Engineer

- **Họ và tên:** Trịnh Đức An
- **MSSV:** 2A202600323
- **Vai trò:** AI Judge Engineer (Multi-Judge & Consensus)

---

## 🛠️ Đóng góp kỹ thuật (Engineering Contribution)
Trong Lab 14, tôi đã xây dựng hệ thống **Consensus Engine** (Đồng thuận) đa mô hình, đảm bảo việc đánh giá không bị phụ thuộc vào một LLM duy nhất.
- **Tích hợp Đa mô hình (Multi-Model Integration)**: Triển khai gọi song song `gpt-4o-mini` (OpenAI) và `gemini-1.5-flash` (Google) bằng `asyncio.gather` để tối ưu hóa Latency.
- **Thiết kế Trục đánh giá (Dual-Axis Rubric)**: Khác với cách chấm điểm thông thường, tôi tách biệt đánh giá thành 2 trục: 
    - **Accuracy (70%)**: So khớp với Ground Truth.
    - **Grounding (30%)**: Kiểm tra xem Agent có bị "hallucination" (bịa đặt) thông tin ngoài Context hay không.
- **Token & Cost Tracking**: Tích hợp logic tính toán `cost_usd` thực tế dựa trên pricing của từng Provider, hỗ trợ báo cáo Performance.

---

## 🧠 Chiều sâu chuyên môn (Technical Depth)
Tôi tập trung giải quyết các bài toán về tính khách quan và nhất quán của Judge:
- **Agreement Rate (Độ đồng thuận)**: Sử dụng ngưỡng `AGREEMENT_STRICT_DIFF = 1` để xác định mức độ tin cậy. Nếu hai Judge lệch nhau quá 1 điểm, hệ thống sẽ đánh dấu sự xung đột, giúp con người dễ dàng can thiệp (Human-in-the-loop).
- **Swap Test (Position Bias)**: Để tránh lỗi "Position Bias" (Judge luôn chọn Response 1 hoặc Response 2 do thói quen), tôi triển khai method `check_position_bias`. Method này đảo vị trí câu trả lời và so sánh độ thống nhất. Nếu Judge đổi ý khi đảo vị trí, hệ thống sẽ cảnh báo Bias.
- **Rubric Logic**: Xây dựng Prompt Engineering chi tiết với các ví dụ cụ thể cho điểm 1-5, giúp LLM Judge giảm thiểu Variant (độ nhiễu) giữa các lần gọi.

---

## 🚀 Giải quyết vấn đề (Problem Solving)
Quá trình triển khai gặp nhiều khó khăn về API và dữ liệu:
- **Degraded Mode (Fail-safe)**: Khi Gemini bị lỗi Quota (429) hoặc OpenAI bị Timeout, tôi triển khai cơ chế `degraded_mode`. Hệ thống sẽ tự động hạ cấp xuống dùng 1 Judge duy nhất thay vì crash toàn bộ Pipeline, đảm bảo tính liên tục (Reliability).
- **Robust JSON Parsing**: LLM đôi khi trả về JSON kèm markdown code blocks (```json ... ```). Tôi đã viết hàm `_clean_json_payload` sử dụng Regex để làm sạch dữ liệu trước khi parse, giúp hệ thống không bị lỗi `JSONDecodeError`.
- **Handling Empty Context**: Xử lý trường hợp Retrieval lỗi (không có chunks), Judge sẽ nhận diện và trừ điểm Grounding một cách hợp lý thay vì nhầm lẫn.

---

## ✍️ Tự đánh giá (Self-Reflection)
- **Bài học rút ra**: Một AI Engineer giỏi không chỉ biết gọi API mà phải biết **đánh giá sự sai lệch của chính AI**. Việc dùng Multi-Judge giúp loại bỏ "điểm mù" của mô hình này bằng ưu điểm của mô hình kia.
- **Trade-off**: Việc dùng 2 mô hình giúp tăng độ tin cậy thêm ~30% nhưng làm tăng chi phí và độ phức tạp mã nguồn. Tuy nhiên, trong các hệ thống RAG doanh nghiệp, độ chính xác (Accuracy) luôn quan trọng hơn một phần nhỏ chi phí.
- **Cam kết**: Toàn bộ logic đã được verify qua 4 kịch bản benchmark (Perfect, Partial, Hallucination, Mis-retrieval).
