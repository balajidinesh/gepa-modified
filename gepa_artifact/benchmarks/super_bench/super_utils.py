import os
import json
import re
from typing import Any, List
from aicodetools import CodeInstance
import dspy


def create_runtime(config: dict = None):
    """Create and initialize the runtime environment"""
    code_config = {
        'docker': {
            'image': 'super-bench:latest',
        },
        'tool_config': {
            'jupyter_enabled': True,
        }
    }
    
    runtime = CodeInstance('docker', code_config, auto_start=True)
    return runtime


def get_runtime_tools():
    """Get fresh runtime tools with new Docker container"""
    rt = create_runtime()
    runtime_tools = rt.get_tools(include=['read_file', 'write_file', 'edit_file', 'run_command'])
    print("Available Tools:", len(runtime_tools))
    return runtime_tools


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


def super_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Main metric function for Super benchmark evaluation"""
    final_submission = prediction.result if hasattr(prediction, 'result') else prediction

    print("**** final submission ****:", final_submission)

    output_folder = 'runs'
    os.makedirs(output_folder, exist_ok=True)

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

    task_name = task["instance_id"]
    print(f"Task {task_name}\n****agent submission: {submission}**")

    gold_answer = json.loads(task["answer"]) if task.get("answer") else None
    if gold_answer:
        print(f"Task {task_name} gold answer: {gold_answer}")
        metrics["output_match"] = evaluate(gold=gold_answer, predicted=submission)
        print(f"Task {task_name} output match metric: {metrics['output_match']}")

    # Handle trace and landmarks
    if hasattr(prediction, 'trajectory'):
        trace = prediction.trajectory
    
    if trace is not None:
        trajectory = [str(i[1]) for i in trace] if isinstance(trace[0], (list, tuple)) else [str(i) for i in trace]
        
        gold_landmarks = task.get("landmarks", [])
        if gold_landmarks:
            metrics["landmarks"] = evaluate_checkpoints(gold_landmarks, trajectory)

        # Save trace to file
        file_name = f"{example['instance_id']}.txt"
        file_path = os.path.join(output_folder, file_name)
        
        with open(file_path, 'w') as file:
            file.write(f"Type of trace: {type(trace).__name__}\n\n")
            file.write(f"Trace length: {len(trace)}\n\n")
            file.write(f"Content of trace:\n{trace}\n\n")
            file.write(f"Metrics of the task:\n{metrics}\n\n")
        
        print('--' * 20)
        print('TRACE IS STORED')
        print('--' * 20)
    else:
        print("Trace is None. File will not be created.")

    print("metrics", metrics)
    return metrics


def super_score(example, prediction, trace=None):
    """Simple scoring function that returns average of output_match and landmarks"""
    final_submission = prediction.result if hasattr(prediction, 'result') else prediction
    
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
    if gold_answer:
        metrics["output_match"] = evaluate(gold=gold_answer, predicted=submission)

    # Handle trace and landmarks
    if hasattr(prediction, 'trajectory'):
        trace = prediction.trajectory
    
    if trace is not None:
        trajectory = [str(i[1]) for i in trace] if isinstance(trace[0], (list, tuple)) else [str(i) for i in trace]
        
        gold_landmarks = task.get("landmarks", [])
        if gold_landmarks:
            metrics["landmarks"] = evaluate_checkpoints(gold_landmarks, trajectory)

    # Calculate score as average of output_match and landmarks
    score = (metrics["output_match"] + metrics["landmarks"]) / 2
    return score


def super_score_with_feedback(example, prediction, trace=None):
    """Scoring function with feedback that returns dspy.Prediction(score=score, feedback=feedback_text)"""
    final_submission = prediction.result if hasattr(prediction, 'result') else prediction
    
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
    if gold_answer:
        metrics["output_match"] = evaluate(gold=gold_answer, predicted=submission)

    # Handle trace and landmarks
    if hasattr(prediction, 'trajectory'):
        trace = prediction.trajectory
    
    if trace is not None:
        print("yyLL",trace)
        trajectory = [str(i[1]) for i in trace] if isinstance(trace[0], (list, tuple)) else [str(i) for i in trace]
        
        gold_landmarks = task.get("landmarks", [])
        if gold_landmarks:
            metrics["landmarks"] = evaluate_checkpoints(gold_landmarks, trajectory)

    # Calculate score as average of output_match and landmarks
    score = (metrics["output_match"] + metrics["landmarks"]) / 2
    
    # Create feedback text as string representation of metrics dict
    feedback_text = str(metrics)
    
    
    return dspy.Prediction(
        score=score,
        feedback=feedback_text,
    )