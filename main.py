import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from common_config import (
    build_default_result_path,
    build_simpleqa_output_payload,
    get_client_params,
    load_json,
    load_simpleqa_dataset,
    load_simpleqa_output,
    load_yaml_config,
    save_json,
    strip_sensitive_config,
    usage_to_token_count,
)
from prompts import QUERY_TEMPLATE, GRADER_TEMPLATE

# Configuration Constants
DATASET_PATH = "dataset/simpleqa_verified.csv"
EVALUATORS_CONFIG_PATH = "evaluators.yaml"
MODELS_CONFIG_PATH = "models.yaml"

CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))
DEFAULT_GRADE_IF_UNPARSEABLE = "C"

def extract_grade_letter(grading_response_text: str) -> str:
    text = (grading_response_text or "").strip().upper()
    match = re.search(r"(A|B|C)", text)
    if match:
        return match.group(0)

    if "CORRECT" in text:
        return "A"
    if "INCORRECT" in text:
        return "B"
    if "NOT_ATTEMPTED" in text:
        return "C"
    return DEFAULT_GRADE_IF_UNPARSEABLE

def calculate_score(grade_letter: str) -> float:
    return 1.0 if grade_letter == "A" else 0.0

def calculate_summary(run_results: list) -> Dict[str, Any]:
    counts = Counter()

    for res in run_results:
        if not res:
            continue

        judge = res.get("judge")
        if not isinstance(judge, dict):
            continue

        grade = judge.get("grade_letter")
        if grade in ("A", "B", "C"):
            counts[grade] += 1
        elif "final_score" in judge:
            counts["A" if float(judge["final_score"]) == 1.0 else "B"] += 1

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

async def call_api_with_retry(
    client: AsyncOpenAI, messages: List[Dict], model: str, **kwargs
) -> Tuple[str, int]:
    """Calls OpenAI API with retries using streaming. Returns (text, token_count)."""
    retries = 3
    for attempt in range(retries):
        try:
            create_kwargs = {**kwargs, "stream": True}
            if "stream_options" not in create_kwargs:
                create_kwargs["stream_options"] = {"include_usage": True}
            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **create_kwargs,
                )
            except Exception:
                create_kwargs.pop("stream_options", None)
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **create_kwargs,
                )

            collected_content: list[str] = []
            usage_obj = None
            async for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        collected_content.append(content)
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage_obj = chunk_usage

            return "".join(collected_content), usage_to_token_count(usage_obj)

        except Exception as e:
            if attempt == retries - 1:
                print(f"API call failed after {retries} attempts: {e}")
                raise e
            await asyncio.sleep(1 * (attempt + 1))
    return "", 0

async def generate_task(
    sem: asyncio.Semaphore,
    item: Dict[str, Any],
    generator_client: AsyncOpenAI,
    generator_model: str,
    generator_kwargs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Generates answer for a single task."""
    async with sem:
        try:
            query = item.get("query", "")
            gold_answer = item.get("gold_answer", "")
            item_id = item.get("id")

            formatted_prompt = QUERY_TEMPLATE.format(question=query)
            try:
                prediction, tokens = await call_api_with_retry(
                    generator_client,
                    [{"role": "user", "content": formatted_prompt}],
                    model=generator_model,
                    **generator_kwargs,
                )
                if not prediction:
                    print(
                        "Generation returned empty content; skip writing "
                        f"(id={item_id})"
                    )
                    return None
            except Exception:
                print(
                    "Generation API failed after retries; skip writing "
                    f"(id={item_id}, problem={query})"
                )
                return None

            return {
                "id": str(item_id),
                "query": query,
                "llm_answer": prediction,
                "tokens": int(tokens),
                "gold_answer": gold_answer,
                "topic": item.get("topic"),
                "answer_type": item.get("answer_type"),
                "multi_step": item.get("multi_step"),
                "requires_reasoning": item.get("requires_reasoning"),
                "urls": item.get("urls"),
            }

        except Exception as e:
            print(f"Task {item.get('id')} generation failed: {e}")
            return None

async def evaluate_generated_task(
    sem: asyncio.Semaphore,
    item: Dict[str, Any],
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_kwargs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Evaluates a generated answer."""
    async with sem:
        try:
            query = item.get("query", "")
            gold_answer = item.get("gold_answer", "")
            prediction = item.get("llm_answer", "")
            item_id = item.get("id")

            if not prediction:
                return None

            grader_prompt = GRADER_TEMPLATE.format(
                question=query,
                target=gold_answer,
                predicted_answer=prediction,
            )
            try:
                grading_response_text, _ = await call_api_with_retry(
                    judge_client,
                    [{"role": "user", "content": grader_prompt}],
                    model=judge_model,
                    **judge_kwargs,
                )
            except Exception:
                print(
                    "Evaluation API failed after retries; skip writing "
                    f"(id={item_id}, query={query})"
                )
                return None

            if not isinstance(grading_response_text, str) or not grading_response_text.strip():
                print(
                    "Evaluation returned empty content; skip writing "
                    f"(id={item_id}, query={query})"
                )
                return None

            grade_letter = extract_grade_letter(grading_response_text)
            grade_str = CHOICE_LETTER_TO_STRING.get(grade_letter, "NOT_ATTEMPTED")
            final_score = calculate_score(grade_letter)

            result = dict(item)
            result["judge"] = {
                "grade_letter": grade_letter,
                "grade_str": grade_str,
                "is_correct": grade_letter == "A",
                "is_incorrect": grade_letter == "B",
                "is_not_attempted": grade_letter == "C",
                "final_score": final_score,
            }
            return result

        except Exception as e:
            print(f"Task {item.get('id')} evaluation failed: {e}")
            return None

def normalize_results_for_evaluation(results: List[Dict[str, Any]]) -> None:
    for item in results:
        if not isinstance(item, dict):
            continue
        if not item.get("query"):
            item["query"] = item.get("problem", "")
        if not item.get("gold_answer"):
            item["gold_answer"] = item.get("answer", "")
        if item.get("id") is None:
            fallback_id = item.get("original_index")
            if fallback_id is not None:
                item["id"] = str(fallback_id)


def print_mode_completion(mode: str, wrote_any_result: bool, pending_count: int, path: Path) -> None:
    if wrote_any_result:
        print(f"{mode} complete. Saved to {path}")
        return
    if pending_count:
        print(
            f"{mode} complete but no new valid results were produced "
            "(API errors or empty responses); skipped writing JSON."
        )
        return
    print(f"{mode} complete. No pending items; skipped writing JSON.")

def is_already_evaluated(item: Dict[str, Any]) -> bool:
    judge = item.get("judge")
    if isinstance(judge, dict):
        return bool(judge)
    return bool(judge)

def load_results_payload(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"Evaluate file not found: {path}")
        return None

    try:
        content = load_json(path)
    except json.JSONDecodeError:
        print(f"Evaluate file is not valid JSON: {path}")
        return None

    if not isinstance(content, dict):
        print("Evaluate file must be format-1 object: {'results': [...]} ")
        return None
    if not isinstance(content.get("results"), list):
        print("Evaluate file must contain a list field: results")
        return None
    return content

def load_existing_generation_results(output_path: Path) -> List[Dict[str, Any]]:
    content = load_simpleqa_output(output_path)
    results = content.get("results", [])
    return results if isinstance(results, list) else []

async def run_generation_mode(args: argparse.Namespace) -> None:
    output_path = build_default_result_path(
        model_id=args.model_id,
        dataset_path=DATASET_PATH,
        save_to=args.save_to,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = load_yaml_config(MODELS_CONFIG_PATH, args.model_id)
    if not model_cfg:
        print(f"Missing config for {args.model_id}")
        return

    model_params, model_kwargs = get_client_params(model_cfg)
    generator_client = AsyncOpenAI(**model_params)

    try:
        dataset = load_simpleqa_dataset(DATASET_PATH, args.num_tasks)
    except FileNotFoundError:
        print(f"Dataset not found at {DATASET_PATH}")
        return
    except Exception as e:
        print(f"Failed to load dataset from {DATASET_PATH}: {e}")
        return

    existing_results = load_existing_generation_results(output_path)
    generated_ids = {
        str(item.get("id"))
        for item in existing_results
        if item.get("id") is not None and item.get("llm_answer")
    }
    tasks_to_run = [item for item in dataset if str(item["id"]) not in generated_ids]
    print(f"Total: {len(dataset)}, Generated: {len(generated_ids)}, To run: {len(tasks_to_run)}")

    sem = asyncio.Semaphore(args.gen_workers)
    tasks = [
        generate_task(
            sem,
            item,
            generator_client,
            args.model_id,
            model_kwargs,
        )
        for item in tasks_to_run
    ]

    all_results = list(existing_results)
    output_model_config = strip_sensitive_config(model_cfg)
    wrote_any_result = False

    if tasks:
        for future in tqdm.as_completed(tasks, total=len(tasks), desc="Generating"):
            if result := await future:
                all_results.append(result)
                try:
                    output_data = build_simpleqa_output_payload(
                        model_config=output_model_config,
                        results=all_results,
                    )
                    save_json(output_path, output_data)
                    wrote_any_result = True
                except Exception as e:
                    print(f"Error saving: {e}")

    print_mode_completion("Generation", wrote_any_result, len(tasks_to_run), output_path)

async def run_evaluation_mode(args: argparse.Namespace) -> None:
    evaluate_path = Path(args.evaluate_file)
    payload = load_results_payload(evaluate_path)
    if payload is None:
        return

    judge_cfg = load_yaml_config(EVALUATORS_CONFIG_PATH, args.evaluator)
    if not judge_cfg:
        print(f"Missing evaluator config for {args.evaluator}")
        return

    judge_params, judge_kwargs = get_client_params(judge_cfg)
    judge_client = AsyncOpenAI(**judge_params)

    results = payload.get("results", [])
    normalize_results_for_evaluation(results)
    pending_indices = [
        idx for idx, item in enumerate(results) if isinstance(item, dict) and not is_already_evaluated(item)
    ]
    print(f"Total: {len(results)}, Skipped: {len(results) - len(pending_indices)}, To evaluate: {len(pending_indices)}")

    async def evaluate_at_index(idx: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        result = await evaluate_generated_task(sem, results[idx], judge_client, args.evaluator, judge_kwargs)
        return idx, result

    sem = asyncio.Semaphore(args.eval_workers)
    tasks = [asyncio.create_task(evaluate_at_index(idx)) for idx in pending_indices]
    wrote_any_result = False

    if tasks:
        for future in tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating"):
            idx, result = await future
            if result:
                results[idx] = result
                payload = build_simpleqa_output_payload(
                    model_config=payload.get("model_config") if isinstance(payload.get("model_config"), dict) else {},
                    results=results,
                    evaluator_config=strip_sensitive_config(judge_cfg),
                    summary=calculate_summary(results),
                )
                try:
                    save_json(evaluate_path, payload)
                    wrote_any_result = True
                except Exception as e:
                    print(f"Error saving: {e}")

    print_mode_completion("Evaluation", wrote_any_result, len(pending_indices), evaluate_path)

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", help="Model to generate answers", default="kimi-k2.5")
    parser.add_argument("--evaluator", help="Evaluator model to use", default="deepseek-v4-pro")
    parser.add_argument("--save-to", help="Path to save the results")
    parser.add_argument("--evaluate-file", help="Evaluate an existing generation JSON file")
    parser.add_argument("--num-tasks", type=int, help="Number of tasks to run from the start of the dataset")
    parser.add_argument("--gen-workers", type=int, help="Concurrent generation workers", default=50)
    parser.add_argument("--eval-workers", type=int, help="Concurrent evaluation workers", default=50)
    args = parser.parse_args()

    if args.evaluate_file:
        if args.num_tasks:
            print("--num-tasks is ignored in --evaluate-file mode.")
        await run_evaluation_mode(args)
        return

    await run_generation_mode(args)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
