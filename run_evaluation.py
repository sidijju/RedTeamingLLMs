import argparse
import subprocess
import json
import os
from datetime import datetime
from evaluation import EvaluationMetrics

def run_single_evaluation(args_dict):
    """Run a single evaluation with the given parameters"""
    cmd = [
        "python3", "main.py",
        "--questions", str(args_dict["questions"]),
        "--actors", str(args_dict["actors"]),
        "--behavior", args_dict["behavior"],
        "--attack_model_name", args_dict["attack_model"],
        "--target_model_name", args_dict["target_model"],
        "--early_stop", str(args_dict["early_stop"]),
        "--dynamic_modify", str(args_dict["dynamic_modify"]),
        "--evaluate", str(args_dict["evaluate"])
    ]
    
    # Add goat flag if enabled
    if args_dict.get("goat", False):
        cmd.append("--goat")

    try:
        # Run subprocess
        subprocess.run(cmd, check=True)
        
        # Get metrics from the most recent file
        eval_dir = './evaluation_result'
        metrics_files = [f for f in os.listdir(eval_dir) if f.startswith('eval_metrics_')]
        if not metrics_files:
            raise Exception("No metrics files found in evaluation_result directory")
            
        latest_file = max(metrics_files, key=lambda f: os.path.getctime(os.path.join(eval_dir, f)))
        metrics_path = os.path.join(eval_dir, latest_file)
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        print(f"  - Error running evaluation: {str(e)}")
        return None

def save_aggregated_results(aggregated_results, target_model, goat):
    """Save aggregated results to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = target_model.split('/')[-1].replace('.', '-')
    goat_str = '_goat' if goat else ''
    agg_path = f'./evaluation_result/aggregated_eval_{model_name}_{timestamp}{goat_str}.json'

    # Add GOAT flag to results
    aggregated_results["goat_enabled"] = goat
    
    with open(agg_path, 'w') as f:
        json.dump(aggregated_results, f, indent=4)
    
    return agg_path

def evaluate_model(args_dict, runs):
    """Run multiple evaluations for a single model"""
    print(f"\nStarting evaluation for target model: {args_dict['target_model']}")
    print(f"Planning {runs} runs...")
    
    results = []
    for i in range(runs):
        print(f"\nStarting run {i+1}/{runs}")
        result = run_single_evaluation(args_dict)
        if result:
            results.append(result)
            print(f"Run {i+1} completed with ASR: {result['test_parameters']['attack_success_rate']:.2%}")
        else:
            print(f"Run {i+1} failed")
    
    if not results:
        print(f"Error: All runs failed for {args_dict['target_model']}. Please check the output above for errors.")
        return None
    
    # Aggregate results
    aggregated = EvaluationMetrics.aggregate_results(results)
    if aggregated:
        agg_path = save_aggregated_results(aggregated, args_dict['target_model'], args_dict['goat'])
        
        # Print summary
        print(f"\nEvaluation Complete for {args_dict['target_model']}!")
        print(f"Successful runs: {len(results)}/{runs}")
        print(f"Mean ASR: {aggregated['metrics']['mean_asr']:.2%}")
        print(f"ASR Standard Deviation: {aggregated['metrics']['std_asr']:.2%}")
        print(f"Min ASR: {aggregated['metrics']['min_asr']:.2%}")
        print(f"Max ASR: {aggregated['metrics']['max_asr']:.2%}")
        print(f"Results saved to: {agg_path}")
        
        return aggregated
    return None

def main():
    parser = argparse.ArgumentParser(description='Run multiple evaluations and aggregate results')
    parser.add_argument("--runs", type=int, default=5, help="Number of evaluation runs per model")
    parser.add_argument("--questions", type=int, default=100, help="Number of questions per run")
    parser.add_argument("--actors", type=int, default=3, help="Number of actors for each question")
    parser.add_argument("--behavior", default="./data/jailbreakbench_harmful.csv", help="Path of harmful behaviors CSV file")
    parser.add_argument("--attack_model", type=str, default="gpt-4o", help="Attack Model name")
    parser.add_argument("--target_models", nargs='+', default=["gpt-4o"], help="List of target model names")
    parser.add_argument("--early_stop", type=bool, default=True, help="early stop if judge determines success")
    parser.add_argument("--dynamic_modify", type=bool, default=True, help="allow dynamic modification of queries")
    parser.add_argument("--evaluate", type=bool, default=True, help="generate evaluation metrics")
    parser.add_argument("--goat", action='store_true', default=False, help="use GOAT attacks")
    
    args = parser.parse_args()
    
    print("Starting evaluation across multiple models...")
    print(f"Target models to evaluate: {args.target_models}")
    
    # Store base arguments in dict
    base_args = {
        "questions": args.questions,
        "actors": args.actors,
        "behavior": args.behavior,
        "attack_model": args.attack_model,
        "early_stop": args.early_stop,
        "dynamic_modify": args.dynamic_modify,
        "evaluate": args.evaluate,
        "goat": args.goat
    }
    
    # Track results for all models
    all_results = {}
    
    # Run evaluations for each target model
    for target_model in args.target_models:
        eval_args = base_args.copy()
        eval_args["target_model"] = target_model
        
        results = evaluate_model(eval_args, args.runs)
        if results:
            all_results[target_model] = results
    
    # Print final comparative summary
    print("\n=== Final Summary Across All Models ===")
    print(f"Models evaluated: {len(all_results)}/{len(args.target_models)}")
    for model, results in all_results.items():
        print(f"\n{model}:")
        print(f"Mean ASR: {results['metrics']['mean_asr']:.2%}")
        print(f"ASR Standard Deviation: {results['metrics']['std_asr']:.2%}")

if __name__ == "__main__":
    main() 
