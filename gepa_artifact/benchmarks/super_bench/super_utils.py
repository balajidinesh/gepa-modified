import os
import json
import re
from typing import Any, List
from aicodetools import ClientManager, CodeToolsClient
import dspy
from pydantic import BaseModel, Field



class FinishResponse(BaseModel):
    success: bool = Field(..., description="Indicates whether the task was completed successfully.")
    structured_output: Any = Field(
        ...,
        description="The main output or result of the task"
    )
    reasoning: str = Field(..., description="Explanation of the reasoning or process followed for the task.")
    structured_output: Any = Field(
        ...,
        description="The main output or result of the task"
    )
    summary: str = Field(..., description="Summary of the steps taken to complete the task.")



class RuntimeManager:
    """Singleton-like runtime manager maintaining one ClientManager."""

    _client_manager = None  # shared instance across all RuntimeManager objects

    def __init__(self):
        # Create the global ClientManager once
        if RuntimeManager._client_manager is None:
            RuntimeManager._client_manager = ClientManager(
                "super-bench:latest", base_log_dir="runs/super/"
            )

    def setup(self, id):
        """Setup client for the given ID."""
        return RuntimeManager._client_manager.get_client(id)

    def cleanup(self, id):
        """Cleanup client for the given ID."""
        return RuntimeManager._client_manager.close_client(id)

def get_runtime_tools(id):
    rmgr = RuntimeManager()
    rt : CodeToolsClient = rmgr.setup(id)
    runtime_tools = rt.tools(selection_list=["read_file", "write_file", "edit_file", "run_command"])
    print(f"Available Tools for {id}: on runtime {rt.docker_container}", len(runtime_tools))
    return runtime_tools, rt

def close_runtime_tools(id):
    rmgr = RuntimeManager()
    rc = rmgr.cleanup(id)
    print(f"Cleaned up Tools {id} : {rc} ", rc )
    return rc

def evaluate(gold: Any, predicted: Any, float_epsilon: float = 1e-2) -> float:
    """Evaluate predicted value against gold standard"""
    if type(gold) == int:
        gold = float(gold)
    if type(predicted) == int:
        predicted = float(predicted)

    if type(gold) != type(predicted):
        return 0.0

    if type(gold) == list:
        if len(gold) == 0:
            raise ValueError("Gold is empty")
        return sum([evaluate(g, p) for p, g in zip(predicted, gold)]) / len(gold)

    if type(gold) == dict:
        if len(gold) == 0:
            raise ValueError("Gold is empty")
        return sum([evaluate(gv, predicted.get(gk, None), float_epsilon=float_epsilon) for gk, gv in gold.items()]) / len(gold)

    if type(gold) == str:
        return float(predicted.strip() == gold.strip())

    if type(gold) == float:
        return float(abs(predicted - gold) < float_epsilon)

    raise NotImplementedError


def evaluate_checkpoints(gold_checkpoints: List[str], agent_history: List[Any]) -> float:
    """
    Evaluate if the agent has gone through some gold checkpoints by looking for certain outputs in the agent's history,
    e.g. "Training completed..."
    """
    checkpoints_hit = []
    agent_history_str: List[str] = []
    if len(agent_history) and type(agent_history[0]) == dict:
        agent_history_str: List[str] = [str(step) for step in agent_history]

    for checkpoint in gold_checkpoints:
        hit = False
        for step in agent_history_str:
            if re.search(checkpoint, step.replace("\n", " ")):
                hit = True
                break
        checkpoints_hit.append(hit)
        print(f"Checkpoint '{checkpoint}': {'Hit' if hit else 'Miss'}")
    
    return sum(checkpoints_hit) / len(checkpoints_hit) if checkpoints_hit else 0.0


def normalize_trace(obj):
    """Convert a trace (dict or string) into a list of readable steps."""
    # --- CASE 1: If it's a structured dict like {'thought_0': ..., 'tool_name_0': ...} ---
    if isinstance(obj, dict) and any(k.startswith('thought_') for k in obj):
        steps = []
        i = 0
        while f"thought_{i}" in obj or f"tool_name_{i}" in obj or f"observation_{i}" in obj:
            step = {
                "thought": obj.get(f"thought_{i}"),
                "tool_call": {
                    "name": obj.get(f"tool_name_{i}"),
                    "args": obj.get(f"tool_args_{i}")
                } if f"tool_name_{i}" in obj or f"tool_args_{i}" in obj else None,
                "observation": obj.get(f"observation_{i}")
            }
            steps.append(step)
            i += 1
        return steps

    # --- CASE 2: Generic dict (not step-based) ---
    elif isinstance(obj, dict):
        return [f"{k}: {v}" for k, v in obj.items()]

    # --- CASE 3: String or other type ---
    else:
        return [str(obj)]


def super_score(example, prediction, trace=None):
    """Simple scoring function that returns average of output_match and landmarks"""
    final_submission = prediction.result if hasattr(prediction, 'result') else prediction
    print(final_submission)
    
    reasoning = prediction.reasoning if hasattr(prediction, 'reasoning') else prediction
    print(reasoning)

    metrics = {
        "submitted": 0,
        "output_match": 0,
        "landmarks": 0
    }
    
    task = example
    submission = None
    
    if final_submission:
        if hasattr(final_submission, 'structured_output') and final_submission.structured_output:
            metrics["submitted"] = 1
            submission = final_submission.structured_output

    gold_answer = json.loads(task["answer"]) if task.get("answer") else None

    print("Gold : ", gold_answer)
    print("Prediction : ", submission)
    if gold_answer is not None:
        metrics["output_match"] = evaluate(gold=gold_answer, predicted=submission)

    # Handle trace and landmarks
    if hasattr(prediction, 'trajectory'):
        trace = prediction.trajectory
        print("has_trajectory")
    
    if trace is not None:
        trajectory = normalize_trace(trace)
        
        gold_landmarks = task.get("landmarks", [])
        if gold_landmarks is not None :
            metrics["landmarks"] = evaluate_checkpoints(gold_landmarks, trajectory)

    # Calculate score as average of output_match and landmarks
    score = (metrics["output_match"] + metrics["landmarks"]) / 2
    pred =  dspy.Prediction(
        score=score,
        score_dict=metrics
    )

    print(f"Task {task.get("instance_id", '')} metrics with feedback : {pred}" )


    return pred


def super_score_with_feedback(example, prediction, trace=None):
    """Simple scoring function with feedback that returns average of output_match and landmarks"""
    final_submission = prediction.result if hasattr(prediction, 'result') else prediction
    print(prediction)
    
    metrics = {
        "submitted": 0,
        "output_match": 0,
        "landmarks": 0
    }
    
    task = example
    submission = None
    
    if final_submission:
        if hasattr(final_submission, 'structured_output') and final_submission.structured_output:
            metrics["submitted"] = 1
            submission = final_submission.structured_output

    gold_answer = json.loads(task["answer"]) if task.get("answer") else None

    print("Gold : ", gold_answer)
    print("Prediction : ", submission)
    if gold_answer is not None:
        metrics["output_match"] = evaluate(gold=gold_answer, predicted=submission)

    # Handle trace and landmarks
    if hasattr(prediction, 'trajectory'):
        trace = prediction.trajectory
        print("has_trajectory")
    
    if trace is not None:
        trajectory = normalize_trace(trace)
        
        gold_landmarks = task.get("landmarks", [])
        if gold_landmarks is not None :
            metrics["landmarks"] = evaluate_checkpoints(gold_landmarks, trajectory)

    # Calculate score as average of output_match and landmarks
    score = (metrics["output_match"] + metrics["landmarks"]) / 2
    
    # Create feedback text as string representation of metrics dict
    feedback_text = str(metrics)
    
    
    pred =  dspy.Prediction(
        score=score,
        score_dict=metrics,
        feedback=feedback_text,
    )


    print(f"Task {task.get("instance_id", '')} metrics with feedback : {pred}" )


    return pred




def super_score_with_gold_feedback(example, prediction, trace=None):
    """Simple scoring function with feedback that returns average of output_match and landmarks"""
    final_submission = prediction.result if hasattr(prediction, 'result') else prediction
    print(prediction)
    
    metrics = {
        "submitted": 0,
        "output_match": 0,
        "landmarks": 0
    }
    
    task = example
    submission = None
    
    if final_submission:
        if hasattr(final_submission, 'structured_output') and final_submission.structured_output:
            metrics["submitted"] = 1
            submission = final_submission.structured_output

    gold_answer = json.loads(task["answer"]) if task.get("answer") else None

    print("Gold : ", gold_answer)
    print("Prediction : ", submission)
    if gold_answer is not None:
        metrics["output_match"] = evaluate(gold=gold_answer, predicted=submission)

    # Handle trace and landmarks
    if hasattr(prediction, 'trajectory'):
        trace = prediction.trajectory
        print("has_trajectory")
    
    if trace is not None:
        trajectory = normalize_trace(trace)
        
        gold_landmarks = task.get("landmarks", [])
        if gold_landmarks is not None :
            metrics["landmarks"] = evaluate_checkpoints(gold_landmarks, trajectory)

    # Calculate score as average of output_match and landmarks
    score = (metrics["output_match"] + metrics["landmarks"]) / 2
    

    solution = task['solution']
    
    feedback_txt = f"the expert solution for the problem : \n\n{solution} {'-'*10} \n\n current metrics : {str(metrics)}"
    
    
    pred =  dspy.Prediction(
        score=score,
        score_dict=metrics,
        feedback=feedback_txt,
    )


    print(f"Task {task.get("instance_id", '')} metrics with feedback : {pred}" )


    return pred