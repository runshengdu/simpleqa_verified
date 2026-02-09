# SimpleQA Verified Evaluation Framework

This project provides an automated framework for evaluating Large Language Models (LLMs) on the **SimpleQA Verified** benchmark, as described in the paper [SimpleQA Verified: A Reliable Factuality Benchmark to Measure Parametric Knowledge](https://arxiv.org/abs/2509.07968).

It uses a "Model-as-a-Judge" approach to grade model predictions against gold-standard answers.

## Features

- **Automated Evaluation**: Automatically queries models and grades their answers using a judge model.
- **Async & Concurrent**: Uses `asyncio` for high-throughput concurrent API calls.
- **Streaming Support**: Handles streaming API responses for better compatibility and reliability.
- **Resume Capability**: Automatically detects existing results and resumes evaluation where it left off.
- **Flexible Configuration**: Define models and evaluators easily in YAML files.
- **Detailed Reporting**: Produces JSON reports with full configuration, metrics (F1, Accuracy), and per-task results.

## Prerequisites

- Python 3.8+
- Required Python packages:
  ```bash
  pip install openai pyyaml tqdm
  ```

## Configuration

The framework uses two main YAML configuration files:

### 1. `models.yaml`
Defines the models to be evaluated.
```yaml
models:
  - name: model-name-slug
    temperature: 1.0
    base_url: https://api.provider.com/v1
    api_key: "${ENV_VAR_NAME}"
    # Additional parameters can be added
```

### 2. `evaluators.yaml`
Defines the judge models used for grading.
```yaml
models:
  - name: judge-model-name
    temperature: 0.0
    base_url: https://api.provider.com/v1
    api_key: "${ENV_VAR_NAME}"
```

**Note**: Ensure required environment variables (e.g., `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`) are set in your environment before running the script.

## Usage

Run the main script to start the evaluation. You can specify the model to evaluate and the evaluator (judge) model using command-line arguments.

```bash
python main.py [options]
```

### Arguments

- `--model-id`: The name of the model to evaluate (must match a name in `models.yaml`). Default: `kimi-k2.5`.
- `--evaluator`: The name of the judge model (must match a name in `evaluators.yaml`). Default: `deepseek-reasoner`.
- `--save-to`: (Optional) Custom path to save the results JSON file. If not provided, results are saved to `result/<model_slug>/simpleqa_verified/<timestamp>.json`.
- `--num-tasks`: (Optional) Number of tasks to run from the start of the dataset. Useful for testing.
- `--max-workers`: (Optional) Maximum number of concurrent tasks. Default: 50.

### Examples

Run evaluation with default settings:
```bash
python main.py
```

Evaluate `gpt-4o` using `deepseek-chat` as the judge:
```bash
python main.py --model-id openai/gpt-4o --evaluator deepseek-chat
```

Run only the first 10 tasks for testing:
```bash
python main.py --num-tasks 10
```

## How It Works

1. **Dataset Loading**: Reads questions and gold answers from `dataset/simpleqa_verified.csv`.
2. **Prediction**: Queries the target model (specified by `--model-id`) for an answer.
3. **Grading**:
   - The judge model (specified by `--evaluator`) compares the predicted answer with the gold answer.
   - It assigns a grade: `A` (CORRECT), `B` (INCORRECT), or `C` (NOT_ATTEMPTED).
4. **Scoring**:
   - `CORRECT` (A) = 1.0
   - `INCORRECT` (B) / `NOT_ATTEMPTED` (C) = 0.0
5. **Metrics**:
   - **F1 Score**: Harmonic mean of precision (accuracy given attempted) and recall (mean correct).
   - **Mean Correct**: Percentage of correct answers (Recall).
   - **Accuracy Given Attempted**: Accuracy on questions the model attempted to answer.
   - **Attempt Rate**: Percentage of questions the model attempted.
6. **Output**: Results are saved incrementally to a JSON file.

## Output Format

The output JSON file contains the configuration, summary metrics, and detailed results:

```json
{
  "model_config": {
    "name": "model-name",
    "base_url": "...",
    "temperature": 0.5
    // api_key is excluded
  },
  "evaluator_config": {
    "name": "judge-name",
    "base_url": "...",
    // api_key is excluded
  },
  "summary": {
    "f1": 0.45,
    "mean_correct": 0.4,
    "accuracy_given_attempted": 0.6,
    "attempt_rate": 0.66,
    "counts": {
      "total": 100,
      "correct": 40,
      "incorrect": 20,
      "not_attempted": 40
    }
  },
  "results": [
    {
      "id": "0",
      "query": "Question text...",
      "llm_answer": "Model's predicted answer...",
      "gold_answer": "Correct answer...",
      "grade_letter": "A",
      "grade_str": "CORRECT",
      "final_score": 1.0,
      ...
    },
    ...
  ]
}
```

## Reference

If you use this benchmark or code, please cite the original paper:

```bibtex
@article{simpleqa_verified_2025,
  title={SimpleQA Verified: A Reliable Factuality Benchmark to Measure Parametric Knowledge},
  author={Lukas Haas and others},
  journal={arXiv preprint arXiv:2509.07968},
  year={2025},
  url={https://arxiv.org/abs/2509.07968}
}
```
