## Notes:
Works with any OpenAI or Anthropic model. It also works with any model from this site: https://deepinfra.com/
- Anthropic models have been buggy when being target models
    - It runs now but occassionally prints out a list index out of range error. I think when it enters the dynamic modification and there is an error when it tries to do  if dialog_hist[-1]['score'] == 5: in the inattack.py file
- The other models work as targets

## Supported Models

- OpenAI models (e.g., gpt-4o)
- Anthropic models (e.g., claude-3-5-sonnet-20241022)
- DeepInfra models (e.g., meta-llama/Meta-Llama-3.1-405B-Instruct)

Note: Anthropic models have shown some instability when used as target models.

## Dependencies
1) Create environment
2) Install required packages

`pip install -r requirement.txt`

3) Make sure you have a `creds.env` file with the following api keys:
    - Openai
    - Anthropic
    - Deepinfra

## Basic Usage (main.py)

Run a single attack:
```
python main.py
```

Parameters:
- `questions`: Number of test questions to run. Default is 1
- `actors`: Number of actors for each question, Default is 3
- `behavior`: Path to CSV file containing test cases. Default is the jailbreak_bench dataset
- `attack_model_name`: Model used for generating attacks. Defualt is GPT-4o
- `target_model_name`: Model being evaluated. Defualt is GPT-4o
- `early_stop`: Stop if judge determines success. Default is True
- `dynamic_modify`: Allow dynamic modification of queries. Default is True
- `evaluate`: Generate evaluation metrics. Default is False

## Multiple Evaluations (run_evaluation.py)

To run multiple evaluations across different models:

```
python run_evaluation.py
```
Parameters:
- `runs`: Number of evaluation runs per model. Default is 5
- `questions`: Number of questions per run. Default is 100 (length of jailbreak bench dataset)
- `actors`: Number of actors per question. Default is 3
- `behavior`: Path to test cases CSV. Default is jailbreak bench dataset
- `attack_model`: Model used for generating attacks. Default is gpt-4o
- `target_models`: Space-separated list of models to evaluate. e.g. gpt-4o claude-sonnet-3-5-20241022
    - Default is gpt-4o
- `early_stop`: Stop if judge determines success. Default is True
- `dynamic_modify`: Allow dynamic modification of queries. Default is True
- `evaluate`: Generate evaluation metrics. Default is True