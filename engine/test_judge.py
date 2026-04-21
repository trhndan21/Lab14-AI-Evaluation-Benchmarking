import argparse
import asyncio
import os
import sys
from pathlib import Path

# Allow running both:
# - python engine/test_judge.py
# - python -m engine.test_judge
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.llm_judge import LLMJudge


def _print_result(name: str, result: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"Final Score: {result['final_score']}")
    print(f"Accuracy Avg: {result.get('accuracy_score_avg')}")
    print(f"Grounding Avg: {result.get('grounding_score_avg')}")
    print(f"Agreement: {result['agreement_rate']}")
    print(f"Degraded: {result.get('degraded_mode')}")

    usage = result.get("usage", {})
    print(f"Total Tokens: {usage.get('total_tokens', 0)}")
    print(f"Total Cost: ${usage.get('total_cost_usd', 0.0):.8f}")

    gpt = result.get("individual_judgments", {}).get("gpt", {})
    gemini = result.get("individual_judgments", {}).get("gemini", {})
    print(f"GPT reasoning: {gpt.get('reasoning')}")
    print(f"Gemini reasoning: {gemini.get('reasoning')}")


async def run_smoke(judge: LLMJudge) -> None:
    question = "Làm thế nào để đo lường hiệu quả của một hệ thống RAG?"
    expected = "Sử dụng Hit Rate, MRR, và RAGAS (Faithfulness, Relevancy)."
    chunks = [
        "Retrieval stage: Hit Rate và MRR.",
        "Generation stage: Faithfulness và Relevancy trong RAGAS.",
    ]
    answer = "Có thể đo bằng Hit Rate và MRR để đánh giá retrieval."

    result = await judge.evaluate_multi_judge(question, answer, expected, chunks)
    _print_result("Smoke Case", result)


async def run_full(judge: LLMJudge) -> None:
    question = "Làm thế nào để đo lường hiệu quả của một hệ thống RAG?"
    expected = "Sử dụng Hit Rate, MRR, và RAGAS (Faithfulness, Relevancy)."
    good_chunks = [
        "Retrieval stage: Hit Rate và MRR.",
        "Generation stage: Faithfulness và Relevancy trong RAGAS.",
    ]

    cases = [
        (
            "CASE 1: Correct + Grounded",
            "Đo RAG bằng Hit Rate, MRR và các tiêu chí Faithfulness, Relevancy của RAGAS.",
            good_chunks,
        ),
        (
            "CASE 2: Partial",
            "Dùng Hit Rate để đo chất lượng retrieval.",
            good_chunks,
        ),
        (
            "CASE 3: Hallucination",
            "Đo bằng dung lượng ổ cứng và tốc độ mạng internet.",
            good_chunks,
        ),
        (
            "CASE 4: Wrong Retrieval Context",
            "Để đo RAG ta dùng Hit Rate và MRR.",
            ["Thời tiết hôm nay nắng đẹp.", "Món phở bò ngon."],
        ),
    ]

    for name, answer, chunks in cases:
        result = await judge.evaluate_multi_judge(question, answer, expected, chunks)
        _print_result(name, result)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test multi-judge engine")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: Missing OPENAI_API_KEY. Judge will run in degraded/failure mode.")
    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: Missing GEMINI_API_KEY. Judge will run in degraded/failure mode.")

    judge = LLMJudge()

    if args.mode == "smoke":
        await run_smoke(judge)
    else:
        await run_full(judge)


if __name__ == "__main__":
    asyncio.run(main())
