import math
from typing import Any, Dict, List


class RetrievalEvaluator:
    """
    Hệ thống đánh giá hiệu suất của Vector Database/Retrieval Stage.
    Cung cấp các chỉ số chuẩn: Hit Rate, MRR, và NDCG.
    """

    def __init__(self):
        pass

    def calculate_hit_rate(
        self, expected_chunk_id: str, retrieved_chunk_ids: List[str], top_k: int = 3
    ) -> float:
        """
        Tính toán Hit Rate @K.
        Trả về 1.0 nếu ít nhất một ID kỳ vọng nằm trong Top K tài liệu được lấy ra.
        """
        top_retrieved = retrieved_chunk_ids[:top_k]
        return 1.0 if expected_chunk_id in top_retrieved else 0.0

    def calculate_mrr(self, expected_chunk_id: str, retrieved_chunk_ids: List[str]) -> float:
        """
        Tính toán Mean Reciprocal Rank (MRR).
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids (1-indexed).
        MRR = 1 / vị trí.
        """
        for i, chunk_id in enumerate(retrieved_chunk_ids):
            if chunk_id == expected_chunk_id:
                return 1.0 / (i + 1)
        return 0.0

    def calculate_ndcg(
        self, expected_chunk_id: str, retrieved_chunk_ids: List[str], top_k: int = 3
    ) -> float:
        """
        Tính toán Normalized Discounted Cumulative Gain (NDCG) @K.
        Giả định độ liên quan nhị phân (1 cho hit, 0 cho miss).
        """
        actual_relevance = [
            1.0 if chunk_id == expected_chunk_id else 0.0
            for chunk_id in retrieved_chunk_ids[:top_k]
        ]
        
        # DCG = sum(rel_i / log2(i + 1 + 1))
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(actual_relevance))
        
        # IDCG (Lý tưởng): Sắp xếp tất cả các hit lên đầu
        hits = sum(actual_relevance)
        if hits == 0:
            return 0.0
            
        idcg = sum(1.0 / math.log2(i + 2) for i in range(int(hits)))
        
        return dcg / idcg

    def evaluate_case(self, expected_chunk_id: str, retrieved_chunk_ids: List[str], top_k: int = 3) -> Dict[str, float]:
        return {
            "hit_rate": self.calculate_hit_rate(expected_chunk_id, retrieved_chunk_ids, top_k=top_k),
            "mrr": self.calculate_mrr(expected_chunk_id, retrieved_chunk_ids),
            "ndcg": self.calculate_ndcg(expected_chunk_id, retrieved_chunk_ids, top_k=top_k),
        }

    async def evaluate_batch(self, dataset: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, float]:
        """
        Đánh giá toàn bộ bộ dữ liệu và trả về các chỉ số trung bình.
        
        Args:
            dataset: Danh sách các dict, mỗi dict chứa
            'expected_chunk_id' và 'retrieved_chunk_ids' (hoặc các key legacy tương đương).
            top_k: Số lượng tài liệu hàng đầu để tính toán Hit Rate và NDCG.
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "avg_ndcg": 0.0}

        hit_rates = []
        mrr_scores = []
        ndcg_scores = []

        for item in dataset:
            expected_chunk_id = (
                item.get("expected_chunk_id")
                or item.get("expected_id")
                or (item.get("expected_ids") or [None])[0]
                or (item.get("ground_truth_ids") or [None])[0]
                or (item.get("expected_retrieval_ids") or [None])[0]
            )
            retrieved_chunk_ids = (
                item.get("retrieved_chunk_ids")
                or item.get("retrieved_ids")
                or []
            )

            if not expected_chunk_id:
                continue

            case_scores = self.evaluate_case(expected_chunk_id, retrieved_chunk_ids, top_k=top_k)
            hit_rates.append(case_scores["hit_rate"])
            mrr_scores.append(case_scores["mrr"])
            ndcg_scores.append(case_scores["ndcg"])

        if not hit_rates:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "avg_ndcg": 0.0}

        return {
            "avg_hit_rate": sum(hit_rates) / len(hit_rates),
            "avg_mrr": sum(mrr_scores) / len(mrr_scores),
            "avg_ndcg": sum(ndcg_scores) / len(ndcg_scores),
        }


if __name__ == "__main__":
    import asyncio

    evaluator = RetrievalEvaluator()

    # --- Test Cases ---
    test_expected = "doc1"
    test_retrieved = ["doc2", "doc1", "doc3"] # Hit ở vị trí 2

    print("--- Testing Single Case ---")
    print(f"Hit@3: {evaluator.calculate_hit_rate(test_expected, test_retrieved, top_k=3)}")
    print(f"MRR:   {evaluator.calculate_mrr(test_expected, test_retrieved)}")
    print(f"NDCG@3: {evaluator.calculate_ndcg(test_expected, test_retrieved, top_k=3):.4f}")

    # --- Testing Batch ---
    dataset = [
        {"expected_chunk_id": "doc1", "retrieved_chunk_ids": ["doc1", "doc2"]}, # Rank 1
        {"expected_chunk_id": "doc2", "retrieved_chunk_ids": ["doc1", "doc2"]}, # Rank 2
        {"expected_chunk_id": "doc3", "retrieved_chunk_ids": ["doc1", "doc2"]}, # Miss
    ]
    
    async def run_test():
        results = await evaluator.evaluate_batch(dataset)
        print("\n--- Testing Batch ---")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
            
    asyncio.run(run_test())
