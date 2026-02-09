import asyncio
import re
from typing import Dict, Any, List, Optional
from collections import Counter
from openai import AsyncOpenAI
from prompts import QUERY_TEMPLATE, GRADER_TEMPLATE

CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))
DEFAULT_GRADE_IF_UNPARSEABLE = "C"

def extract_grade_letter(grading_response_text: str) -> str:
    text = (grading_response_text or "").strip().upper()
    match = re.search(r"(A|B|C)", text)
    if match:
        return match.group(0)
    
    if "CORRECT" in text: return "A"
    if "INCORRECT" in text: return "B"
    if "NOT_ATTEMPTED" in text: return "C"
    return DEFAULT_GRADE_IF_UNPARSEABLE

def calculate_score(grade_letter: str) -> float:
    return 1.0 if grade_letter == "A" else 0.0

def calculate_summary(run_results: list) -> Dict[str, Any]:
    counts = Counter()
    
    for res in run_results:
        if not res: continue
        
        grade = res.get("grade_letter")
        if grade in ("A", "B", "C"):
            counts[grade] += 1
        elif "final_score" in res:
            counts["A" if float(res["final_score"]) == 1.0 else "B"] += 1

    correct = counts["A"]
    incorrect = counts["B"]
    not_attempted = counts["C"]
    total = correct + incorrect + not_attempted

    attempted = correct + incorrect
    mean_correct = (correct / total) if total else 0.0
    attempt_rate = (attempted / total) if total else 0.0
    accuracy_given_attempted = (correct / attempted) if attempted else 0.0

    denominator = accuracy_given_attempted + mean_correct
    f1 = (2 * accuracy_given_attempted * mean_correct / denominator) if denominator else 0.0

    return {
        "f1": f1,
        "mean_correct": mean_correct,
        "accuracy_given_attempted": accuracy_given_attempted,
        "attempt_rate": attempt_rate,
        "counts": {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "not_attempted": not_attempted,
        },
    }

async def call_api_with_retry(client: AsyncOpenAI, messages: List[Dict], model: str, **kwargs) -> str:
    """Calls OpenAI API with retries using streaming."""
    retries = 3
    for attempt in range(retries):
        try:
            # Enable streaming
            kwargs['stream'] = True
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            collected_content = []
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    collected_content.append(content)
            
            return "".join(collected_content)

        except Exception as e:
            if attempt == retries - 1:
                print(f"API call failed after {retries} attempts: {e}")
                raise e
            # Simple backoff
            await asyncio.sleep(1 * (attempt + 1))
    return ""

async def evaluate_task(
    sem: asyncio.Semaphore,
    item: Dict[str, Any],
    evaluator_client: AsyncOpenAI,
    judge_client: AsyncOpenAI,
    evaluator_model: str,
    judge_model: str,
    evaluator_kwargs: Dict,
    judge_kwargs: Dict
) -> Optional[Dict[str, Any]]:
    """Evaluates a single task."""
    async with sem:
        try:
            query = item.get("problem", "")
            gold_answer = item.get("answer", "")
            item_id = item.get("original_index")
            if item_id is None or str(item_id).strip() == "":
                item_id = item.get("id")
    
            # 1. Get Prediction
            formatted_prompt = QUERY_TEMPLATE.format(question=query)
            try:
                prediction = await call_api_with_retry(
                    evaluator_client,
                    [{"role": "user", "content": formatted_prompt}],
                    model=evaluator_model, # Usually ignored by openrouter if base_url is specific, but good practice
                    **evaluator_kwargs
                )
                if not prediction:
                    return None
            except Exception:
                return None # Fail silently/skip as requested

            grader_prompt = GRADER_TEMPLATE.format(
                question=query,
                target=gold_answer,
                predicted_answer=prediction,
            )
            try:
                grading_response_text = await call_api_with_retry(
                    judge_client,
                    [{"role": "user", "content": grader_prompt}],
                    model=judge_model,
                    **judge_kwargs
                )
            except Exception:
                return None

            grade_letter = extract_grade_letter(grading_response_text)
            grade_str = CHOICE_LETTER_TO_STRING.get(grade_letter, "NOT_ATTEMPTED")
            final_score = calculate_score(grade_letter)

            return {
                "id": str(item_id),
                "query": query,
                "llm_answer": prediction,
                "gold_answer": gold_answer,
                "grade_letter": grade_letter,
                "grade_str": grade_str,
                "is_correct": grade_letter == "A",
                "is_incorrect": grade_letter == "B",
                "is_not_attempted": grade_letter == "C",
                "final_score": final_score,
                "topic": item.get("topic"),
                "answer_type": item.get("answer_type"),
                "multi_step": item.get("multi_step"),
                "requires_reasoning": item.get("requires_reasoning"),
                "urls": item.get("urls"),
            }

        except Exception as e:
            print(f"Task {item.get('id')} failed: {e}")
            return None
