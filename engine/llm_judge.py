import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Task 3 fixed params
GPT_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-1.5-flash"
WEIGHT_ACCURACY = 0.7
WEIGHT_GROUNDING = 0.3
AGREEMENT_STRICT_DIFF = 1
REQUEST_TIMEOUT_SECONDS = 30
SCORE_MIN = 1
SCORE_MAX = 5

# Approximate pricing per 1M tokens
COST_CONFIG = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-1.5-flash": {"input": 0.10, "output": 0.40},
}


class LLMJudge:
    def __init__(self, gpt_model: str = GPT_MODEL, gemini_model: str = GEMINI_MODEL):
        self.gpt_model = gpt_model
        self.gemini_model = gemini_model if gemini_model.startswith("models/") else f"models/{gemini_model}"

        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.genai_model = genai.GenerativeModel(self.gemini_model)

        self.rubrics = {
            "accuracy": {
                "1": "Hoàn toàn sai hoặc không liên quan đến expected answer.",
                "2": "Có ý đúng nhỏ nhưng sai/thiếu phần lớn ý quan trọng.",
                "3": "Đúng khoảng 50-70%, còn thiếu ý trọng yếu.",
                "4": "Gần đúng đầy đủ, chỉ thiếu chi tiết nhỏ.",
                "5": "Chính xác và đầy đủ theo expected answer.",
            },
            "grounding": {
                "1": "Bịa đặt/hallucination, không bám retrieved chunks.",
                "2": "Có dùng context nhưng vẫn suy diễn nhiều.",
                "3": "Bám context một phần, còn vài chỗ không chắc.",
                "4": "Bám tốt context, sai sót rất nhỏ.",
                "5": "Hoàn toàn dựa trên retrieved chunks, không thêm thắt.",
            },
        }

    def _build_judge_prompt(
        self,
        question: str,
        agent_answer: str,
        expected_answer: str,
        retrieved_chunks: List[str],
    ) -> str:
        context = "\n".join(f"- {c}" for c in (retrieved_chunks or []))
        if not context:
            context = "- (Không có retrieved chunks)"

        return f"""
Bạn là AI Judge trung lập. Chỉ chấm theo dữ liệu cung cấp, không suy diễn ngoài phạm vi.

[DỮ LIỆU]
- Question: {question}
- Expected Answer: {expected_answer}
- Retrieved Chunks:
{context}
- Agent Answer: {agent_answer}

[THANG ĐIỂM]
1) accuracy_score (1-5): so với Expected Answer
{json.dumps(self.rubrics['accuracy'], ensure_ascii=False, indent=2)}

2) grounding_score (1-5): mức bám Retrieved Chunks
{json.dumps(self.rubrics['grounding'], ensure_ascii=False, indent=2)}

[OUTPUT JSON BẮT BUỘC]
{{
  "accuracy_score": <int 1-5>,
  "grounding_score": <int 1-5>,
  "reasoning": "<1-3 câu ngắn, nêu rõ lý do>"
}}
""".strip()

    def _clean_json_payload(self, text: str) -> str:
        payload = (text or "").strip()
        if payload.startswith("```"):
            payload = re.sub(r"^```(?:json)?", "", payload).strip()
            if payload.endswith("```"):
                payload = payload[:-3].strip()
        return payload

    def _safe_int_score(self, value: Any, default: int = SCORE_MIN) -> int:
        try:
            v = int(value)
        except Exception:
            v = default
        return max(SCORE_MIN, min(SCORE_MAX, v))

    def _calc_cost_usd(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        key = "gpt-4o-mini" if "gpt" in model_name.lower() else "gemini-1.5-flash"
        cfg = COST_CONFIG.get(key, {"input": 0.0, "output": 0.0})
        return (prompt_tokens * cfg["input"] + completion_tokens * cfg["output"]) / 1_000_000

    def _normalize_provider_result(
        self,
        model: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        parsed = data or {}
        accuracy = self._safe_int_score(parsed.get("accuracy_score", SCORE_MIN))
        grounding = self._safe_int_score(parsed.get("grounding_score", SCORE_MIN))
        reasoning = parsed.get("reasoning") or (f"Error: {error}" if error else "")

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": self._calc_cost_usd(model, prompt_tokens, completion_tokens),
            "latency_seconds": round(latency_seconds, 4),
        }

        return {
            "model": model,
            "accuracy_score": accuracy,
            "grounding_score": grounding,
            "reasoning": reasoning,
            "error": error,
            "usage": usage,
        }

    async def _call_gpt_judge(self, prompt: str) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[
                        {"role": "system", "content": "You are a strict evaluator. Return JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            payload = self._clean_json_payload(response.choices[0].message.content or "")
            data = json.loads(payload)
            usage = response.usage
            return self._normalize_provider_result(
                model=self.gpt_model,
                data=data,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                latency_seconds=time.perf_counter() - start,
            )
        except Exception as e:
            return self._normalize_provider_result(
                model=self.gpt_model,
                error=f"{type(e).__name__}: {e}",
                latency_seconds=time.perf_counter() - start,
            )

    async def _call_gemini_judge(self, prompt: str) -> Dict[str, Any]:
        start = time.perf_counter()
        candidates = [self.gemini_model, "models/gemini-2.0-flash", "models/gemini-flash-latest"]
        last_err = None

        for model_name in candidates:
            try:
                model = self.genai_model if model_name == self.gemini_model else genai.GenerativeModel(model_name)
                response = await asyncio.wait_for(
                    model.generate_content_async(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=0,
                        ),
                    ),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                payload = self._clean_json_payload(getattr(response, "text", ""))
                data = json.loads(payload)

                prompt_tokens = 0
                completion_tokens = 0
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

                return self._normalize_provider_result(
                    model=model_name,
                    data=data,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_seconds=time.perf_counter() - start,
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"

        return self._normalize_provider_result(
            model=self.gemini_model,
            error=last_err or "Unknown Gemini error",
            latency_seconds=time.perf_counter() - start,
        )

    async def evaluate_multi_judge(
        self,
        question: str,
        agent_answer: str,
        expected_answer: str,
        retrieved_chunks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_judge_prompt(
            question=question,
            agent_answer=agent_answer,
            expected_answer=expected_answer,
            retrieved_chunks=retrieved_chunks or [],
        )

        gpt_result, gemini_result = await asyncio.gather(
            self._call_gpt_judge(prompt),
            self._call_gemini_judge(prompt),
        )

        valid = [r for r in [gpt_result, gemini_result] if r.get("error") is None]
        degraded_mode = len(valid) < 2

        if not valid:
            return {
                "final_score": 1.0,
                "agreement_rate": 0.0,
                "accuracy_score_avg": 1.0,
                "grounding_score_avg": 1.0,
                "degraded_mode": True,
                "individual_judgments": {"gpt": gpt_result, "gemini": gemini_result},
                "usage": {"total_tokens": 0, "total_cost_usd": 0.0},
                "reasoning": "Both judges failed.",
            }

        acc_avg = sum(r["accuracy_score"] for r in valid) / len(valid)
        grd_avg = sum(r["grounding_score"] for r in valid) / len(valid)
        final_score = (acc_avg * WEIGHT_ACCURACY) + (grd_avg * WEIGHT_GROUNDING)

        if len(valid) == 2:
            diff_acc = abs(gpt_result["accuracy_score"] - gemini_result["accuracy_score"])
            diff_grd = abs(gpt_result["grounding_score"] - gemini_result["grounding_score"])
            if diff_acc <= AGREEMENT_STRICT_DIFF and diff_grd <= AGREEMENT_STRICT_DIFF:
                agreement_rate = 1.0
            elif diff_acc > AGREEMENT_STRICT_DIFF and diff_grd > AGREEMENT_STRICT_DIFF:
                agreement_rate = 0.0
            else:
                agreement_rate = 0.5
        else:
            agreement_rate = 0.5

        total_tokens = 0
        total_cost = 0.0
        for r in [gpt_result, gemini_result]:
            usage = r.get("usage") or {}
            total_tokens += int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
            total_cost += float(usage.get("cost_usd", 0.0))

        reasoning = valid[0].get("reasoning", "")

        return {
            "final_score": round(final_score, 2),
            "agreement_rate": agreement_rate,
            "accuracy_score_avg": round(acc_avg, 2),
            "grounding_score_avg": round(grd_avg, 2),
            "degraded_mode": degraded_mode,
            "individual_judgments": {
                "gpt": gpt_result,
                "gemini": gemini_result,
            },
            "usage": {
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 8),
            },
            "reasoning": reasoning,
        }

    async def check_position_bias(
        self,
        question: str,
        response_a: str,
        response_b: str,
        expected_answer: str,
        retrieved_chunks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        context = "\n".join(f"- {c}" for c in (retrieved_chunks or []))

        def _prompt(resp1: str, resp2: str) -> str:
            return f"""
Bạn là giám khảo trung lập. Chọn câu trả lời tốt hơn.
Question: {question}
Expected Answer: {expected_answer}
Retrieved Chunks:
{context}
Response 1: {resp1}
Response 2: {resp2}
Trả về JSON: {{"better_response": 1 hoặc 2, "reasoning": "..."}}
""".strip()

        async def _call_compare(prompt: str) -> Dict[str, Any]:
            try:
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.gpt_model,
                        messages=[
                            {"role": "system", "content": "You are a strict evaluator. Return JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0,
                    ),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                payload = self._clean_json_payload(response.choices[0].message.content or "")
                data = json.loads(payload)
                better = int(data.get("better_response", 0))
                if better not in (1, 2):
                    better = 0
                return {"better_response": better, "reasoning": data.get("reasoning", "")}
            except Exception as e:
                return {"better_response": 0, "reasoning": f"{type(e).__name__}: {e}"}

        p1 = _prompt(response_a, response_b)
        p2 = _prompt(response_b, response_a)
        res1, res2 = await asyncio.gather(_call_compare(p1), _call_compare(p2))

        choice1 = res1.get("better_response", 0)
        choice2 = res2.get("better_response", 0)
        bias_detected = choice1 == choice2 and choice1 in (1, 2)

        return {
            "bias_detected": bias_detected,
            "choices": [choice1, choice2],
            "reasoning": [res1.get("reasoning"), res2.get("reasoning")],
        }
