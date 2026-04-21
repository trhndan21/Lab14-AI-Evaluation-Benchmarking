import asyncio
import os
import argparse
from engine.llm_judge import LLMJudge

async def run_test_case(judge, name, question, answer, expected, chunks):
    print(f"\n--- {name} ---")
    result = await judge.evaluate_multi_judge(question, answer, expected, chunks)
    print(f"Final Score: {result['final_score']}")
    print(f"Accuracy Avg: {result.get('accuracy_score_avg')}")
    print(f"Grounding Avg: {result.get('grounding_score_avg')}")
    print(f"Agreement: {result['agreement_rate']}")
    print(f"Total Cost: ${result.get('total_cost', 0):.6f}")
    print(f"Degraded: {result.get('degraded_mode')}")
    
    # Lấy reasoning từ kết quả mới (cấu trúc individual_judgments đã đổi)
    gpt_reasoning = result['individual_judgments']['gpt']['scores'].get('reasoning')
    gemini_reasoning = result['individual_judgments']['gemini']['scores'].get('reasoning')
    print(f"Reasoning (GPT): {gpt_reasoning}")
    print(f"Reasoning (Gemini): {gemini_reasoning}")
    return result

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    judge = LLMJudge()
    
    # Dữ liệu mẫu
    question = "Làm thế nào để đo lường hiệu quả của một hệ thống RAG?"
    expected = "Sử dụng các chỉ số như Hit Rate, MRR, và các khung đánh giá như RAGAS (Faithfulness, Relevancy)."
    chunks = [
        "Hệ thống RAG thường được đánh giá qua Retrieval stage (Hit Rate, MRR) và Generation stage.",
        "RAGAS là một framework phổ biến cung cấp các chỉ số như Faithfulness và Answer Relevancy."
    ]

    if args.mode == "smoke":
        print("🚀 Đang chạy Smoke Test...")
        answer = "Chúng ta dùng Hit Rate và MRR để đo lường."
        await run_test_case(judge, "Smoke Case", question, answer, expected, chunks)

    elif args.mode == "full":
        print("🚀 Đang chạy Full Benchmark (4 Cases)...")
        
        # Case 1: Perfect
        await run_test_case(judge, "CASE 1: Perfect Answer", 
            question, 
            "Để đo lường RAG, ta dùng Hit Rate, MRR cho retrieval và RAGAS cho generation stage.",
            expected, chunks)

        # Case 2: Partial
        await run_test_case(judge, "CASE 2: Partial Answer", 
            question, 
            "Dùng Hit Rate là đủ để biết RAG tốt hay không.",
            expected, chunks)

        # Case 3: Hallucination
        await run_test_case(judge, "CASE 3: Hallucination", 
            question, 
            "Đo lường RAG bằng cách kiểm tra dung lượng ổ cứng và tốc độ mạng internet.",
            expected, chunks)

        # Case 4: Mis-Retrieval (Agent cố đoán dù context không liên quan)
        bad_chunks = ["Thời tiết hôm nay rất đẹp.", "Món phở bò rất ngon."]
        await run_test_case(judge, "CASE 4: Mis-Retrieval (Guessing)", 
            question, 
            "Để đo RAG ta dùng Hit Rate và MRR.", # Đúng kiến thức chung nhưng không có trong context
            expected, bad_chunks)

        print("\n--- TEST: Swap Test (Position Bias) ---")
        bias_result = await judge.check_position_bias(
            question, 
            "Dùng Hit Rate.", 
            "Dùng RAGAS framework.", 
            expected, chunks
        )
        print(f"Bias detected: {bias_result['bias_detected']}")
        print(f"Choices: {bias_result['choices']}")

if __name__ == "__main__":
    asyncio.run(main())
