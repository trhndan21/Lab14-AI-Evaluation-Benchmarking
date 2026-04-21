import asyncio
import json
import os
import time
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- Thông số cố định theo Task 3 (Tối ưu cho environment hiện tại) ---
GPT_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-flash-latest"
SCORE_RANGE = (1, 5)
WEIGHT_ACCURACY = 0.7
WEIGHT_GROUNDING = 0.3
AGREEMENT_STRICT_DIFF = 1
REQUEST_TIMEOUT_SECONDS = 60

# Ước tính chi phí (USD trên 1M tokens)
COST_CONFIG = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4}
}

class LLMJudge:
    def __init__(self, gpt_model: str = GPT_MODEL, gemini_model: str = GEMINI_MODEL):
        self.gpt_model = gpt_model
        self.gemini_model = gemini_model if gemini_model.startswith("models/") else f"models/{gemini_model}"
        
        # Cấu hình API Clients
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.genai_model = genai.GenerativeModel(self.gemini_model)
        
        # Định nghĩa rubrics theo yêu cầu Task 3
        self.rubrics = {
            "accuracy": {
                "1": "Hoàn toàn sai hoặc không liên quan đến Expected Answer.",
                "2": "Có một vài ý đúng nhưng phần lớn là sai thông tin quan trọng.",
                "3": "Đúng khoảng 50-70%, thiếu một vài chi tiết hoặc diễn đạt chưa chuẩn.",
                "4": "Rất chính xác, chỉ thiếu sót những chi tiết cực nhỏ.",
                "5": "Hoàn hảo, chính xác tuyệt đối so với Expected Answer."
            },
            "grounding": {
                "1": "Hallucination nghiêm trọng, thông tin bịa đặt hoàn toàn không có trong context.",
                "3": "Có sử dụng context nhưng có lẫn thông tin bên ngoài không được xác thực.",
                "5": "Hoàn toàn dựa trên tài liệu tham khảo (retrieved chunks), không thêm thắt."
            }
        }

    def _build_judge_prompt(self, question: str, answer: str, expected_answer: str, retrieved_chunks: List[str]) -> str:
        context_str = "\n".join([f"- {c}" for c in retrieved_chunks])
        return f"""
Bạn là một chuyên gia đánh giá chất lượng AI Agent (AI Judge). 
Nhiệm vụ của bạn là chấm điểm câu trả lời của Agent dựa trên 'Expected Answer' và 'Retrieved Chunks' (Tài liệu tham khảo).

[DỮ LIỆU ĐÁNH GIÁ]
- Câu hỏi: {question}
- Expected Answer (Đáp án kỳ vọng): {expected_answer}
- Retrieved Chunks (Tài liệu Agent đã đọc):
{context_str}
- Agent's Answer (Câu trả lời cần chấm): {answer}

[TIÊU CHÍ CHẤM ĐIỂM]
1. Accuracy (Độ chính xác so với Expected Answer):
{json.dumps(self.rubrics['accuracy'], ensure_ascii=False, indent=2)}

2. Grounding (Độ trung thực so với Retrieved Chunks):
{json.dumps(self.rubrics['grounding'], ensure_ascii=False, indent=2)}

[YÊU CẦU ĐẦU RA]
Trả về định dạng JSON duy nhất:
{{
  "accuracy_score": <int 1-5>,
  "grounding_score": <int 1-5>,
  "reasoning": "<Giải thích ngắn gọn lý do>"
}}
"""

    def _calculate_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        model_key = "gpt-4o-mini" if "gpt" in model_name.lower() else "gemini-2.0-flash"
        config = COST_CONFIG.get(model_key, {"input": 0.15, "output": 0.60})
        return (prompt_tokens * config["input"] + completion_tokens * config["output"]) / 1_000_000

    async def _call_gpt_judge(self, prompt: str) -> Dict[str, Any]:
        try:
            start_t = time.perf_counter()
            response = await self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "You are a precise AI Quality Evaluator. Always output JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            data = json.loads(response.choices[0].message.content)
            usage = response.usage
            return {
                "scores": data,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "cost": self._calculate_cost(self.gpt_model, usage.prompt_tokens, usage.completion_tokens),
                    "latency": time.perf_counter() - start_t
                },
                "error": None
            }
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            return {"scores": {"accuracy_score": 1, "grounding_score": 1, "reasoning": f"Error: {err_msg}"}, "usage": None, "error": err_msg}

    async def _call_gemini_judge(self, prompt: str) -> Dict[str, Any]:
        try:
            start_t = time.perf_counter()
            response = await self.genai_model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0
                )
            )
            try:
                usage = response.usage_metadata
                prompt_tokens = usage.prompt_token_count
                completion_tokens = usage.candidates_token_count
            except:
                prompt_tokens, completion_tokens = 0, 0

            data = json.loads(response.text)
            return {
                "scores": data,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": self._calculate_cost("gemini", prompt_tokens, completion_tokens),
                    "latency": time.perf_counter() - start_t
                },
                "error": None
            }
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            return {"scores": {"accuracy_score": 1, "grounding_score": 1, "reasoning": f"Error: {err_msg}"}, "usage": None, "error": err_msg}

    async def evaluate_multi_judge(self, question: str, agent_answer: str, expected_answer: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
        prompt = self._build_judge_prompt(question, agent_answer, expected_answer, retrieved_chunks)
        
        res_gpt, res_gemini = await asyncio.gather(
            self._call_gpt_judge(prompt),
            self._call_gemini_judge(prompt)
        )
        
        valid_results = []
        if res_gpt.get("error") is None: valid_results.append(res_gpt["scores"])
        if res_gemini.get("error") is None: valid_results.append(res_gemini["scores"])
        
        degraded = len(valid_results) < 2
        
        if not valid_results:
            return {
                "final_score": 1.0,
                "agreement_rate": 0.0,
                "accuracy_score_avg": 1.0,
                "grounding_score_avg": 1.0,
                "degraded_mode": True,
                "reasoning": f"Both judges failed. GPT: {res_gpt.get('error')} | Gemini: {res_gemini.get('error')}",
                "total_cost": 0,
                "individual_judgments": {"gpt": res_gpt, "gemini": res_gemini}
            }

        acc_scores = [r.get("accuracy_score", 1) for r in valid_results]
        grd_scores = [r.get("grounding_score", 1) for r in valid_results]
        
        avg_acc = sum(acc_scores) / len(acc_scores)
        avg_grd = sum(grd_scores) / len(grd_scores)
        
        final_score = (avg_acc * WEIGHT_ACCURACY) + (avg_grd * WEIGHT_GROUNDING)
        
        agreement = 1.0
        if len(valid_results) == 2:
            diff_acc = abs(res_gpt["scores"].get("accuracy_score", 1) - res_gemini["scores"].get("accuracy_score", 1))
            diff_grd = abs(res_gpt["scores"].get("grounding_score", 1) - res_gemini["scores"].get("grounding_score", 1))
            if diff_acc <= AGREEMENT_STRICT_DIFF and diff_grd <= AGREEMENT_STRICT_DIFF:
                agreement = 1.0
            elif diff_acc > AGREEMENT_STRICT_DIFF and diff_grd > AGREEMENT_STRICT_DIFF:
                agreement = 0.0
            else:
                agreement = 0.5
        
        total_cost = (res_gpt["usage"]["cost"] if res_gpt["usage"] else 0) + (res_gemini["usage"]["cost"] if res_gemini["usage"] else 0)
        
        return {
            "final_score": round(final_score, 2),
            "agreement_rate": agreement,
            "accuracy_score_avg": avg_acc,
            "grounding_score_avg": avg_grd,
            "degraded_mode": degraded,
            "total_cost": total_cost,
            "individual_judgments": {
                "gpt": res_gpt,
                "gemini": res_gemini
            }
        }

    async def check_position_bias(self, question: str, response_a: str, response_b: str, expected_answer: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
        def build_compare_prompt(ans_1, ans_2):
            return f"Compare Response 1: {ans_1} and Response 2: {ans_2} for Question: {question}. Which is better? Return JSON: {{\"better_response\": 1 or 2, \"reasoning\": \"...\"}}"
        
        p1 = build_compare_prompt(response_a, response_b)
        p2 = build_compare_prompt(response_b, response_a)
        
        res1, res2 = await asyncio.gather(
            self._call_gpt_judge(p1),
            self._call_gpt_judge(p2)
        )
        
        choice1 = res1["scores"].get("better_response")
        choice2 = res2["scores"].get("better_response")
        bias_detected = (choice1 == choice2)
        
        return {
            "bias_detected": bias_detected,
            "choices": [choice1, choice2],
            "reasoning": [res1["scores"].get("reasoning"), res2["scores"].get("reasoning")]
        }
