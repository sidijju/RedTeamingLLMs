import argparse
import subprocess
from datetime import datetime
import json

def run_test_suite(config):
    """
    Run evaluation with specified configuration
    """

    start_time = datetime.now()
    print(f"\n=== Starting Test Suite at {start_time} ===")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    cmd = [
        "python3", "run_evaluation.py",
        "--runs", str(config["runs"]),
        "--questions", str(config["questions"]),
        "--actors", str(config["actors"]),
        "--attack_model", config["attack_model"],
        "--target_models"
    ] + config["target_models"]


    if config["goat"]:
        cmd.append("--goat")

    try:
        subprocess.run(cmd, check=True)
        print("\n=== Test Suite Completed Successfully ===")
        print("Time to complete: ", datetime.now() - start_time)
    except subprocess.CalledProcessError as e:
        print(f"\n=== Test Suite Failed with Error: {e} ===")

def main():
    # testing config
    config = {
        "runs": 2,
        "questions": 10,
        "actors": 3,
        "behavior": "./data/jailbreakbench_harmful.csv",
        "attack_model": "gpt-4o",
        "target_models": [
            "gpt-4o-mini",
            "gpt-4o",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            # "meta-llama/Meta-Llama-3.1-405B-Instruct"
        ],
        "early_stop": True,
        "dynamic_modify": True,
        "evaluate": True,
        "goat": False
    }

    # Run without GOAT attacks
    print("\n=== Running without GOAT attacks ===")
    run_test_suite(config)

    # Run with GOAT attacks
    print("\n=== Running with GOAT attacks ===") 
    config["goat"] = True
    run_test_suite(config)

if __name__ == "__main__":
    main() 