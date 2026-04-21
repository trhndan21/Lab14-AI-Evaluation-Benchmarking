import argparse
import json
import os
import random

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHUNKS_PATH = "data/chunks.jsonl"
FAISS_INDEX_PATH = "data/faiss.index"
CHUNK_META_PATH = "data/chunk_meta.json"

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 3
RANDOM_RATE_V1 = 0.5
RANDOM_SEED = 20260421


class Retriever:
    def __init__(self):
        self.client = OpenAI()
        self.index = None
        self.chunk_meta = []
        self._rng = random.Random(RANDOM_SEED)
        self.build_or_load_index()

    def build_or_load_index(self):
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNK_META_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNK_META_PATH, "r", encoding="utf-8") as f:
                self.chunk_meta = json.load(f)
            print(f"Loaded index with {len(self.chunk_meta)} chunks from {FAISS_INDEX_PATH}")
            return
        self._build_index()

    def _build_index(self):
        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError(f"{CHUNKS_PATH} not found. Run synthetic_gen.py first.")

        chunks = []
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        print(f"Loaded {len(chunks)} chunks")

        texts = [c["chunk_text"] for c in chunks]
        embeddings = self._embed_batch(texts)
        print(f"Embeddings shape: {embeddings.shape}")

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.chunk_meta = [
            {"chunk_id": c["chunk_id"], "doc_name": c["doc_name"], "chunk_text": c["chunk_text"]}
            for c in chunks
        ]

        faiss.write_index(self.index, FAISS_INDEX_PATH)
        print(f"Saved {FAISS_INDEX_PATH}")

        with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunk_meta, f, ensure_ascii=False, indent=2)
        print(f"Saved {CHUNK_META_PATH}")

    def _embed_batch(self, texts: list) -> np.ndarray:
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            all_embeddings.extend([item.embedding for item in response.data])
        return np.array(all_embeddings, dtype="float32")

    def _embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        vec = np.array([response.data[0].embedding], dtype="float32")
        faiss.normalize_L2(vec)
        return vec

    def retrieve_v2(self, question: str, top_k: int = TOP_K) -> dict:
        query_vec = self._embed_query(question)
        scores, indices = self.index.search(query_vec, top_k)

        retrieved_chunk_ids = []
        retrieved_chunks = []
        retrieved_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            meta = self.chunk_meta[idx]
            retrieved_chunk_ids.append(meta["chunk_id"])
            retrieved_chunks.append(meta["chunk_text"])
            retrieved_scores.append(float(score))

        return {
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_chunks": retrieved_chunks,
            "scores": retrieved_scores,
            "retrieval_mode": "v2_faiss",
        }

    def retrieve_v1(self, question: str, top_k: int = TOP_K) -> dict:
        if self._rng.random() < RANDOM_RATE_V1:
            print("  branch=random")
            sampled = self._rng.sample(self.chunk_meta, min(top_k, len(self.chunk_meta)))
            return {
                "retrieved_chunk_ids": [m["chunk_id"] for m in sampled],
                "retrieved_chunks": [m["chunk_text"] for m in sampled],
                "scores": [0.0] * len(sampled),
                "retrieval_mode": "v1_random_mix",
            }
        else:
            print("  branch=faiss")
            result = self.retrieve_v2(question, top_k)
            result["retrieval_mode"] = "v1_random_mix"
            return result

    def retrieve(self, question: str, version: str = "v2", top_k: int = TOP_K) -> dict:
        if version == "v1":
            return self.retrieve_v1(question, top_k)
        return self.retrieve_v2(question, top_k)


def _cmd_build_index():
    r = Retriever.__new__(Retriever)
    r.client = OpenAI()
    r.chunk_meta = []
    r.index = None
    r._rng = random.Random(RANDOM_SEED)
    r._build_index()


def _cmd_test_retrieve(version: str, question: str, top_k: int, trials: int):
    retriever = Retriever()

    if trials > 1:
        if version != "v1":
            print("Trials mode currently supports --version v1 only.")
            return
        random_count = 0
        for _ in range(trials):
            result = retriever.retrieve(question, version=version, top_k=top_k)
            # In v1, nhánh random đang trả scores toàn 0.0
            if result["scores"] and all(score == 0.0 for score in result["scores"]):
                random_count += 1
        print(f"\nTrials={trials}, branch=random count={random_count} ({random_count/trials*100:.1f}%)")
        print("Expected range: 45-55%")
        return

    result = retriever.retrieve(question, version=version, top_k=top_k)
    print(f"\nmode: {result['retrieval_mode']}")
    for i, (cid, score) in enumerate(zip(result["retrieved_chunk_ids"], result["scores"])):
        print(f"  [{i+1}] chunk_id={cid}  score={score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retriever CLI")
    parser.add_argument("--build-index", action="store_true", help="Build FAISS index from chunks.jsonl")
    parser.add_argument("--test-retrieve", action="store_true", help="Test retrieval")
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--question", default="Quy trình cấp quyền truy cập Level 3 là gì?")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--trials", type=int, default=1,
                        help="Chạy N lần để kiểm tra tỷ lệ random (dùng với --version v1)")
    args = parser.parse_args()

    if args.build_index:
        _cmd_build_index()
    elif args.test_retrieve:
        _cmd_test_retrieve(args.version, args.question, args.top_k, args.trials)
    else:
        parser.print_help()
