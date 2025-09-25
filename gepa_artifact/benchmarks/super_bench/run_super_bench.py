#!/usr/bin/env python3
"""
Super Benchmark Runner

This script runs the SuperReactAgent on Super benchmark tasks with proper train/val/test splits.
It creates fresh Docker containers for each task and evaluates performance.

Usage:
    python gepa_artifact/benchmarks/super_bench/run_super_bench.py --split test                    # Run on test split (default)
    python gepa_artifact/benchmarks/super_bench/run_super_bench.py --split train                   # Run on train split
    python gepa_artifact/benchmarks/super_bench/run_super_bench.py --split val                     # Run on validation split
    python gepa_artifact/benchmarks/super_bench/run_super_bench.py --split test --single mera      # Run single task from test split
    python gepa_artifact/benchmarks/super_bench/run_super_bench.py --split train --task_ids mera team  # Run specific tasks from train split

Hardcoded splits:
    Train: ['mera', 'team']
    Val: ['dir-gnn', 'scicode']
    Test: ['math-comp', 'ml-engineering']
"""

import argparse
import dspy
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add the project root to Python path for proper imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from gepa_artifact.benchmarks.super_bench.super_data import SuperBenchmark
from gepa_artifact.benchmarks.super_bench.super_program import SuperReactAgent
from gepa_artifact.benchmarks.super_bench.super_utils import super_metric

# Load environment variables
load_dotenv()

def setup_dspy():
    """Setup DSPy with Azure GPT-4o"""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Warning: No API key found. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.")

    try:
        lm = dspy.LM("azure/gpt-4o")
        dspy.configure(lm=lm)
        print(f"DSPy configured with language model: {lm}")
        return lm
    except Exception as e:
        print(f"Error setting up DSPy: {e}")
        print("Trying fallback configuration...")
        try:
            lm = dspy.LM("azure/gpt-4o")
            dspy.configure(lm=lm)
            print(f"DSPy configured with fallback model: {lm}")
            return lm
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise

def run_single_task(agent, example):
    """Run agent on a single task"""
    print(f"\n{'='*50}")
    print(f"Running task: {example.instance_id}")
    print(f"Query: {example.query[:100]}...")
    print(f"{'='*50}")

    try:
        # Call agent with new single input pattern but pass all fields via kwargs
        result = agent(
            query=example.query,
            github_repo=getattr(example, 'github_repo', ''),
            git_commit=getattr(example, 'git_commit', ''),
            instance_id=getattr(example, 'instance_id', '')
        )

        # Evaluate the result
        metrics = super_metric(example, result)

        print(f"\nTask {example.instance_id} completed!")
        print(f"Metrics: {metrics}")

        return metrics

    except Exception as e:
        print(f"Error running task {example.instance_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"submitted": 0, "output_match": 0, "landmarks": 0, "error": str(e)}

def run_benchmark(split="test", task_ids=None, max_iters=100):
    """Run the Super benchmark with specified split and optional task filtering"""

    # Setup DSPy
    lm = setup_dspy()

    # Create benchmark dataset with hardcoded splits
    benchmark = SuperBenchmark(dataset_mode="lite")

    # Get the appropriate split
    if split == "train":
        dataset = benchmark.get_train_set()
        print(f"Running on train set: {[ex.instance_id for ex in dataset]}")
    elif split == "val":
        dataset = benchmark.val_set
        print(f"Running on validation set: {[ex.instance_id for ex in dataset]}")
    elif split == "test":
        dataset = benchmark.get_test_set()
        print(f"Running on test set: {[ex.instance_id for ex in dataset]}")
    else:
        print(f"Invalid split '{split}'. Available splits: train, val, test")
        return

    # Optional task ID filtering
    if task_ids is not None:
        if 'all' not in task_ids:
            dataset = [ex for ex in dataset if ex.instance_id in task_ids]
            print(f"Filtered to specific task IDs: {task_ids}")

    if not dataset:
        print("No examples found for the specified criteria!")
        return

    print(f"Loaded {len(dataset)} examples")

    # Create agent with error handling
    try:
        agent = SuperReactAgent(max_iters=max_iters)
        print(f"SuperReactAgent created successfully with max_iters={max_iters}")

        # Test that agent can get tools
        tools = agent.get_fresh_tools()
        print(f"Agent initialized with {len(tools)} tools")

    except Exception as e:
        print(f"Error creating SuperReactAgent: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run evaluation
    all_metrics = []

    for i, example in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] Processing task: {example.instance_id}")

        metrics = run_single_task(agent, example)
        all_metrics.append(metrics)

    # Calculate overall metrics
    if all_metrics:
        avg_metrics = {
            "submitted": sum(m.get("submitted", 0) for m in all_metrics) / len(all_metrics),
            "output_match": sum(m.get("output_match", 0) for m in all_metrics) / len(all_metrics),
            "landmarks": sum(m.get("landmarks", 0) for m in all_metrics) / len(all_metrics),
        }

        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Split: {split}")
        print(f"Tasks completed: {len(all_metrics)}")
        print(f"Average submission rate: {avg_metrics['submitted']:.2%}")
        print(f"Average output match: {avg_metrics['output_match']:.2%}")
        print(f"Average landmarks hit: {avg_metrics['landmarks']:.2%}")
        print(f"Overall score: {(avg_metrics['output_match'] + avg_metrics['landmarks']) / 2:.2%}")

    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="Run Super Benchmark with React Agent")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to run on (train/val/test)"
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        help="Optional task IDs to filter (space-separated). Use 'all' for all tasks in the split."
    )
    parser.add_argument(
        "--single",
        type=str,
        help="Run a single task ID"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="Maximum iterations for React agent"
    )

    args = parser.parse_args()

    if args.single:
        task_ids = [args.single]
    else:
        task_ids = args.task_ids

    run_benchmark(split=args.split, task_ids=task_ids, max_iters=args.max_iters)

if __name__ == "__main__":
    main()