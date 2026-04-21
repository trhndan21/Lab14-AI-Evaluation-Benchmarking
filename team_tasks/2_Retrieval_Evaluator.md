# Thành viên 2 - Retrieval Engineer (FAISS + V1/V2)

## Mục tiêu
Xây engine retrieval 2 phiên bản để benchmark:
- **V1:** 50% kết quả random chunk.
- **V2:** 100% retrieval chuẩn theo FAISS.

## Thông số cố định (không đổi)
- `EMBEDDING_MODEL = text-embedding-3-small`
- `FAISS_INDEX_TYPE = IndexFlatIP`
- `TOP_K = 3`
- `RANDOM_RATE_V1 = 0.5`
- `RANDOM_SEED = 20260421`
- `TEST_RANDOM_TRIALS = 100`

## File phụ trách
- `agent/retriever.py` (**file mới, tách riêng retrieval**)
- `agent/main_agent.py` (giữ vai trò wrapper gọi retriever + generation)
- `engine/retrieval_eval.py`
- `requirements.txt` (bổ sung `faiss-cpu` nếu thiếu)

## Nhiệm vụ chính
1. Build FAISS index từ `data/chunks.jsonl`:
   - embedding cho từng chunk,
   - map `faiss_idx -> chunk_id`.
2. Implement 2 hàm retrieval:
   - `retrieve_v1(query, top_k)`: 50% random chunk thay cho kết quả đúng.
   - `retrieve_v2(query, top_k)`: trả đúng top-k từ similarity search.
3. Chuẩn hóa output retrieval:
   - `retrieved_chunk_ids`,
   - `retrieved_chunks`,
   - `scores` (nếu có).
4. Cập nhật evaluator để chấm theo `expected_chunk_id`:
   - Hit@K,
   - MRR,
   - (optional) NDCG.

## Quy tắc cho V1 random
- Mỗi query có xác suất 0.5 đi nhánh random.
- Random phải lấy từ toàn bộ tập chunk (trừ trùng nếu cần).
- Cố định seed để benchmark tái lập (reproducible).

## Checklist bàn giao
- [ ] Có thể chạy retrieval độc lập và in top-k chunk_id.
- [ ] V1 và V2 cùng input nhưng kết quả khác biệt rõ.
- [ ] Metric function nhận trực tiếp `expected_chunk_id` + `retrieved_chunk_ids`.

## Interface gửi cho team
- Hàm truy vấn trả về schema ổn định để Runner gọi không cần sửa tay.

---

## Cần sửa gì (cực cụ thể)
### File 1: `agent/retriever.py` (mới)
- Tách toàn bộ retrieval logic khỏi `main_agent.py`.
- Implement các hàm:
  - `build_or_load_index()`
  - `retrieve_v1(question: str, top_k: int) -> dict`
  - `retrieve_v2(question: str, top_k: int) -> dict`
  - `retrieve(question: str, version: str, top_k: int = 3) -> dict`
- Output retrieval chuẩn:
  - `retrieved_chunk_ids`
  - `retrieved_chunks`
  - `scores`
  - `retrieval_mode` (`v1_random_mix` hoặc `v2_faiss`)

### File 2: `agent/main_agent.py`
- Giữ phần orchestration:
  - gọi `Retriever` từ `agent/retriever.py`,
  - tạo `answer` từ context đã retrieve.
- Không giữ retrieval logic trực tiếp trong file này.

### File 3: `engine/retrieval_eval.py`
- Chuẩn hóa evaluator theo expected 1 chunk:
  - `calculate_hit_rate(expected_chunk_id, retrieved_chunk_ids, top_k)`
  - `calculate_mrr(expected_chunk_id, retrieved_chunk_ids)`
  - `evaluate_case(...)`
  - `evaluate_batch(...)`
- Đảm bảo tương thích với schema `golden_set.jsonl` mới.

### File 4: `requirements.txt`
- Bổ sung dependency nếu thiếu:
  - `faiss-cpu`
  - model embedding đang dùng (nếu cần).

---

## Flow chạy code từng bước + output mong đợi
### Bước 1: Build FAISS index từ chunks
- Chạy:
  - `python agent/retriever.py --build-index`
- Script phải:
  1. Đọc `data/chunks.jsonl`.
  2. Embed từng `chunk_text`.
  3. Build FAISS index.
  4. Ghi:
     - `data/faiss.index`
     - `data/chunk_meta.json` (map index -> chunk_id/doc_name/text).
- Output console mong đợi:
  - `Loaded X chunks`
  - `Embeddings shape: (X, D)`
  - `Saved data/faiss.index`
  - `Saved data/chunk_meta.json`

### Bước 2: Test retrieval V2 (không random)
- Chạy:
  - `python agent/retriever.py --test-retrieve --version v2 --question "..." --top_k 3`
- Output console mong đợi:
  - in top_k chunk_id + score.
  - mode hiển thị `v2_faiss`.
- Output logic mong đợi:
  - query lặp lại nhiều lần cho cùng input cho kết quả ổn định (hoặc gần ổn định).

### Bước 3: Test retrieval V1 (50% random)
- Chạy:
  - `python agent/retriever.py --test-retrieve --version v1 --question "..." --top_k 3 --seed 20260421`
- Output console mong đợi:
  - log nhánh:
    - `branch=faiss` hoặc
    - `branch=random`.
  - tỷ lệ random xấp xỉ 50% khi chạy nhiều vòng.
- Validation nhanh:
  - chạy đúng `100` query mẫu -> `branch=random` phải nằm trong khoảng `45-55`.

### Bước 4: Test metric function độc lập
- Chạy:
  - `python engine/retrieval_eval.py`
- Script test phải in:
  - case hit top1,
  - case hit top3,
  - case miss hoàn toàn.
- Output mong đợi:
  - công thức Hit@K và MRR đúng theo tay tính.

---

## Schema output bắt buộc bàn giao cho Runner
- `retrieve(...)` trả về:
  - `retrieved_chunk_ids: list[str]`
  - `retrieved_chunks: list[str]`
  - `scores: list[float]` (nếu có)
  - `retrieval_mode: str`
- `query(...)` (ở `main_agent.py`) trả về:
  - `answer: str`
  - `retrieved_chunk_ids: list[str]`
  - `retrieved_chunks: list[str]`
  - `retrieval_mode: str`
- Evaluator trả về:
  - `hit_rate: float`
  - `mrr: float`
  - `ndcg: float` (nếu dùng)

---

## Tiêu chí hoàn thành (DoD cá nhân)
- Build index chạy được, sinh đủ 2 file index + meta.
- V1 có random 50% tái lập được bằng seed.
- V2 retrieval sạch, không random.
- Metric test pass với expected output đã tính tay.
