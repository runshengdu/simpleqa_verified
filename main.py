import argparse
import asyncio
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from evaluator import evaluate_task, calculate_summary

# Configuration Constants
DATASET_PATH = "dataset/simpleqa_verified.csv"
EVALUATORS_CONFIG_PATH = "evaluators.yaml"
MODELS_CONFIG_PATH = "models.yaml"

def load_yaml_config(path: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Loads configuration for a specific model from a YAML file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            for model in data.get('models', []):
                if model.get('name') == model_name:
                    config = model.copy()
                    if api_key := config.get('api_key'):
                        config['api_key'] = os.path.expandvars(api_key)
                        # Handle ${VAR} if expandvars didn't catch it
                        if config['api_key'].startswith('${') and config['api_key'].endswith('}'):
                            config['api_key'] = os.environ.get(config['api_key'][2:-1], '')
                    return config
    except Exception as e:
        print(f"Error loading config from {path}: {e}")
    return None

def get_client_params(config: Dict) -> Tuple[Dict, Dict]:
    params = {
        "api_key": config.get("api_key"),
        "base_url": config.get("base_url"),
    }
    exclude_keys = {"name", "api_key", "base_url"}
    chat_kwargs = {k: v for k, v in config.items() if k not in exclude_keys}
    return params, chat_kwargs

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", help="Model to evaluate", default="kimi-k2.5")
    parser.add_argument("--evaluator", help="Evaluator model to use", default="deepseek-reasoner")
    parser.add_argument("--save-to", help="Path to save the results")
    parser.add_argument("--num-tasks", type=int, help="Number of tasks to run from the start of the dataset")
    parser.add_argument("--max-workers", type=int, help="Maximum number of concurrent tasks", default=50)
    args = parser.parse_args()

    # 1. Output Path
    if args.save_to:
        output_path = Path(args.save_to)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = args.model_id.replace("/", "_")
        test_set_slug = Path(DATASET_PATH).stem
        output_path = Path(f"result/{model_slug}/{test_set_slug}/{timestamp}.json")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Configs & Clients
    eval_cfg = load_yaml_config(MODELS_CONFIG_PATH, args.model_id)
    judge_cfg = load_yaml_config(EVALUATORS_CONFIG_PATH, args.evaluator)

    if not eval_cfg or not judge_cfg:
        print(f"Missing config for {args.model_id} or {args.evaluator}")
        return

    eval_params, eval_kwargs = get_client_params(eval_cfg)
    judge_params, judge_kwargs = get_client_params(judge_cfg)
    
    eval_client = AsyncOpenAI(**eval_params)
    judge_client = AsyncOpenAI(**judge_params)

    # 3. Load Dataset
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = list(csv.DictReader(f))
            for idx, item in enumerate(dataset):
                item_id = item.get("original_index") or item.get("id") or str(idx)
                item["id"] = str(item_id).strip()
    except FileNotFoundError:
        print(f"Dataset not found at {DATASET_PATH}")
        return

    if args.num_tasks:
        dataset = dataset[:args.num_tasks]

    # 4. Resume
    completed_ids = set()
    existing_results = []
    if output_path.exists():
        try:
            content = json.loads(output_path.read_text(encoding='utf-8'))
            if isinstance(content, dict): existing_results = content.get("results", [])
            elif isinstance(content, list): existing_results = content
            
            completed_ids = {str(r["id"]) for r in existing_results if "id" in r and r.get("final_score") is not None}
            print(f"Resuming... {len(completed_ids)} tasks already completed.")
        except json.JSONDecodeError:
            print("Output file exists but is not valid JSON. Starting fresh.")

    tasks_to_run = [item for item in dataset if str(item["id"]) not in completed_ids]
    
    print(f"Total: {len(dataset)}, Skipped: {len(completed_ids)}, To run: {len(tasks_to_run)}")

    # 5. Run
    sem = asyncio.Semaphore(args.max_workers)
    tasks = [
        evaluate_task(sem, item, eval_client, judge_client, args.model_id, args.evaluator, eval_kwargs, judge_kwargs)
        for item in tasks_to_run
    ]

    all_results = list(existing_results)
    
    # Prepare configs for output (excluding API keys)
    output_model_config = {k: v for k, v in eval_cfg.items() if k != 'api_key'}
    output_evaluator_config = {k: v for k, v in judge_cfg.items() if k != 'api_key'}

    if tasks:
        for future in tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating"):
            if result := await future:
                all_results.append(result)
                try:
                    output_data = {
                        "model_config": output_model_config,
                        "evaluator_config": output_evaluator_config,
                        "summary": calculate_summary(all_results),
                        "results": all_results
                    }
                    output_path.write_text(
                        json.dumps(output_data, indent=2, ensure_ascii=False),
                        encoding='utf-8'
                    )
                except Exception as e:
                    print(f"Error saving: {e}")

    print(f"Evaluation complete. Saved to {output_path}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
