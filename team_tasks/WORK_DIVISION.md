# WORK DIVISION - Bản chốt cuối cùng

## 1) Mục tiêu bắt buộc
- Benchmark Retrieval Agent với 2 phiên bản:
  - `v1`: 50% random chunk.
  - `v2`: 0% random, chỉ FAISS similarity.
- Dùng dữ liệu từ `data/docs`.
- Tạo đúng `50` câu benchmark map tới `50` `expected_chunk_id` khác nhau.

## 2) Thông số cố định toàn team (không được đổi)
- `CHUNK_SIZE_TOKENS = 256`
- `CHUNK_OVERLAP_TOKENS = 32`
- `EMBEDDING_MODEL = text-embedding-3-small`
- `FAISS_INDEX_TYPE = IndexFlatIP`
- `TOP_K = 3`
- `RANDOM_RATE_V1 = 0.5`
- `RANDOM_SEED = 20260421`
- `TOTAL_QUESTIONS = 50`
- `CONCURRENCY = 8`
- `JUDGE_WEIGHT_ACCURACY = 0.7`
- `JUDGE_WEIGHT_GROUNDING = 0.3`
- `GATE_MIN_DELTA_HIT_RATE = +0.10`
- `GATE_MIN_DELTA_MRR = +0.10`
- `GATE_MIN_DELTA_FINAL_SCORE = 0.00`

## 3) Output bắt buộc
- `data/chunks.jsonl`
- `data/faiss.index`
- `data/chunk_meta.json`
- `data/golden_set.jsonl`
- `reports/benchmark_results.json`
- `reports/summary.json`
- `analysis/failure_analysis.md`

## 4) Chia việc 5 người (ngắn gọn, dễ nhìn)
- **Người 1 - Data Engineer**
  - Input: `data/docs/*.txt`
  - Làm: chunking + sinh `golden_set`
  - Output: `data/chunks.jsonl`, `data/golden_set.jsonl`
- **Người 2 - Retrieval Engineer**
  - Input: `chunks.jsonl`, `golden_set.jsonl`
  - Làm: build FAISS, tách retrieval sang `agent/retriever.py`, implement `retrieve_v1/retrieve_v2`, metric retrieval
  - Output: agent query schema + retrieval eval pass
- **Người 3 - LLM Judge Engineer**
  - Input: question, expected_answer, retrieved_chunks, agent_answer
  - Làm: multi-judge + agreement + fallback
  - Output: score schema ổn định cho runner
- **Người 4 - Async Runner**
  - Input: `golden_set.jsonl`, agent, evaluator, judge
  - Làm: chạy batch async cho `v1` và `v2`
  - Output: `reports/benchmark_results.json`
- **Người 5 - DevOps/Analyst**
  - Input: `benchmark_results.json`
  - Làm: tổng hợp delta + release gate + 5 whys
  - Output: `reports/summary.json`, `analysis/failure_analysis.md`

## 5) Flow chạy end-to-end (lệnh chuẩn)
1. `python data/synthetic_gen.py --mode chunk`
2. `python data/synthetic_gen.py --mode golden --n 50`
3. `python data/validate_dataset.py`
4. `python agent/retriever.py --build-index`
5. `python main.py --mode benchmark --both`
6. `python main.py --mode summarize`
7. `python check_lab.py`

## 6) Điều kiện PASS
- `golden_set.jsonl` có đúng `50` dòng và `50 expected_chunk_id` unique.
- `summary.json` có đủ: `hit_rate_v1/v2`, `mrr_v1/v2`, `final_score_v1/v2`, `delta_*`, `release_decision`.
- `release_decision = APPROVE` chỉ khi:
  - `delta_hit_rate >= 0.10`
  - `delta_mrr >= 0.10`
  - `delta_final_score >= 0.00`
