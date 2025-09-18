#!/usr/bin/env python3
"""
Super Benchmark Runner

This script runs the SuperReactAgent on specified Super benchmark tasks.
It creates fresh Docker containers for each task and evaluates performance.

Usage:
    python run_super_bench.py --task_ids mera team dir-gnn
    python run_super_bench.py --task_ids all  # Run all available tasks
    python run_super_bench.py --single mera   # Run single task
"""

import argparse
import dspy
from dotenv import load_dotenv
import os

from super_data import SuperBenchmark
from super_program import SuperReactAgent
from super_utils import super_metric

# Load environment variables
load_dotenv()

def setup_dspy():
    """Setup DSPy with Azure GPT-4o"""
    lm = dspy.LM("azure/gpt-4o")
    dspy.configure(lm=lm)
    return lm

def run_single_task(agent, example):
    """Run agent on a single task"""
    print(f"\n{'='*50}")
    print(f"Running task: {example.instance_id}")
    print(f"Query: {example.query[:100]}...")
    print(f"{'='*50}")
    
    try:
        result = agent(
            query=example.query,
            github_repo=example.github_repo,
            git_commit=example.git_commit,
            instance_id=example.instance_id
        )
        
        # Evaluate the result
        metrics = super_metric(example, result)
        
        print(f"\nTask {example.instance_id} completed!")
        print(f"Metrics: {metrics}")
        
        return metrics
        
    except Exception as e:
        print(f"Error running task {example.instance_id}: {e}")
        return {"submitted": 0, "output_match": 0, "landmarks": 0, "error": str(e)}

def run_benchmark(task_ids=None, max_iters=100):
    """Run the Super benchmark with specified task IDs"""
    
    # Available task IDs in Super benchmark
    available_tasks = ['mera', 'team', 'dir-gnn', 'scicode', 'math-comp', 'ml-engineering']
    
    if task_ids is None or 'all' in task_ids:
        task_ids = available_tasks
    
    # Filter valid task IDs
    valid_task_ids = [tid for tid in task_ids if tid in available_tasks]
    invalid_task_ids = [tid for tid in task_ids if tid not in available_tasks]
    
    if invalid_task_ids:
        print(f"Warning: Invalid task IDs: {invalid_task_ids}")
        print(f"Available task IDs: {available_tasks}")
    
    if not valid_task_ids:
        print("No valid task IDs provided!")
        return
    
    print(f"Running Super benchmark with task IDs: {valid_task_ids}")
    
    # Setup DSPy
    lm = setup_dspy()
    
    # Create benchmark dataset
    benchmark = SuperBenchmark(dataset_mode="test", instance_ids=valid_task_ids)
    test_set = benchmark.get_test_set()
    
    print(f"Loaded {len(test_set)} test examples")
    
    # Create agent
    agent = SuperReactAgent(max_iters=max_iters)
    
    # Run evaluation
    all_metrics = []
    
    for i, example in enumerate(test_set):
        print(f"\n[{i+1}/{len(test_set)}] Processing task: {example.instance_id}")
        
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
        print(f"Tasks completed: {len(all_metrics)}")
        print(f"Average submission rate: {avg_metrics['submitted']:.2%}")
        print(f"Average output match: {avg_metrics['output_match']:.2%}")
        print(f"Average landmarks hit: {avg_metrics['landmarks']:.2%}")
        print(f"Overall score: {(avg_metrics['output_match'] + avg_metrics['landmarks']) / 2:.2%}")
    
    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="Run Super Benchmark with React Agent")
    parser.add_argument(
        "--task_ids", 
        nargs="+", 
        default=["mera", "team", "dir-gnn"],
        help="Task IDs to run (space-separated). Use 'all' for all tasks."
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
    
    run_benchmark(task_ids=task_ids, max_iters=args.max_iters)

if __name__ == "__main__":
    main()