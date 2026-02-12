import os
import json
import re
from typing import Any, List
from aicodetools import ClientManager, CodeToolsClient
import dspy
from pydantic import BaseModel, Field

from typing import Optional
import logging

from pydantic import BaseModel

from pathlib import Path

import subprocess
import subprocess
from typing import Union


from .evaluator import evaluate_instance

# class FinishResponse(BaseModel):
#     success: bool = Field(..., description="Indicates whether the task was completed successfully.")
    
#     reasoning: str = Field(..., description="Explanation of the reasoning or process followed for the task.")
#     summary: str = Field(..., description="Summary of the steps taken to complete the task.")



abs_benchmark_path = '/mnt/d/sab/ScienceAgentBench/benchmark'
abs_dataset_path = '/mnt/d/sab/ScienceAgentBench/benchmark/datasets'


class RuntimeManager:
    """Singleton-like runtime manager maintaining one ClientManager."""

    _client_manager = None  # shared instance across all RuntimeManager objects

    def __init__(self):
        # Create the global ClientManager once (lazy import to avoid heavy import cost)
        if RuntimeManager._client_manager is None:
            from aicodetools import ClientManager  # local import to keep top-level light
            RuntimeManager._client_manager = ClientManager(
                "sab:latest", base_log_dir="tool_runs/sab/"
            )

    def setup(self, id):
        """Setup client for the given ID."""
        return RuntimeManager._client_manager.get_client(id, mnts= [f'{abs_dataset_path}:/benchmark/datasets'])

    def cleanup(self, id):
        """Cleanup client for the given ID."""
        return RuntimeManager._client_manager.close_client(id)
    
    

def get_runtime_tools(id,logger: Optional[logging.Logger] = None):
    # TODO TEST If multiple threads call setup or cleanup concurrently, if i may see unexpected behavior.
    logger = logger 
    rmgr = RuntimeManager()
    rt = rmgr.setup(id)
    runtime_tools = rt.tools(selection_list=["read_file", "write_file", "edit_file", "run_command"])
    print(f"Available Tools for {id}: on runtime {rt.docker_container} ({len(runtime_tools)} tools)")
    return runtime_tools, rt

def close_runtime_tools(id,logger: Optional[logging.Logger] = None):
    logger =  logger 
    rmgr = RuntimeManager()
    rc = rmgr.cleanup(id)
    print(f"Cleaned up Tools {id} : {rc}")
    return rc



def _check_and_read(container_name: str, path: str) -> Union[str, bool]:
    """
    Check if a file exists at exact path inside container and return content.
    """

    try:
        exists = subprocess.run(
            ["docker", "exec", container_name, "test", "-f", path],
            capture_output=True,
            timeout=5
        )

        if exists.returncode != 0:
            return False

        result = subprocess.run(
            ["docker", "exec", container_name, "cat", path],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return result.stdout

    except Exception:
        return False

    return False


def read_file_from_container(client, path: str):
    """
    Read file from container with simple, explicit path resolution.
    """

    container_name = client.container_name
    if not container_name:
        return False
    
    if not path.lower().endswith(".py"):
        return 'NOT A PYTHON PROGROM, ENSURE YOU SUBMIT THE ABSOLUTE PATH OF THE PYTHON FILE .py'

    path = path.strip()

    # 1️⃣ Absolute path → check only this
    if path.startswith("/"):
        return _check_and_read(container_name, path)

    # 2️⃣ Home (~) path
    if path.startswith("~"):
        home_path = path.replace("~", "/root", 1)
        content = _check_and_read(container_name, home_path)
        if content:
            return content

    # Normalize relative path
    clean_path = path.lstrip("./")

    # 3️⃣ Plain relative
    content = _check_and_read(container_name, clean_path)
    if content:
        return content

    # 4️⃣ /workspace
    content = _check_and_read(container_name, f"/workspace/{clean_path}")
    if content:
        return content

    # 5️⃣ /app
    content = _check_and_read(container_name, f"/app/{clean_path}")
    if content:
        return content

    # 6️⃣ /root
    content = _check_and_read(container_name, f"/root/{clean_path}")
    if content:
        return content

    return 'NO PROGROM FOUND, ENSURE YOU SUBMIT THE ABSOLUTE PATH OF THE PYTHON FILE'


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


def sab_score(example, prediction, trace=None):
    """Simple scoring function that returns average of output_match and landmarks"""

    final_code = prediction.generated_code if hasattr(prediction, 'generated_code') else prediction
    print("submission :", final_code.replace("\n", "\\n")[:30])

    task = example
    submission = final_code 
    submitted =  True if final_code else False

    r_kwargs = {
        "instance_id": task.instance_id,
        "gold_program_name": task.get("gold_program_name"),
        "task_inst": task.get("task_inst"),
        "output_fname": task.get("output_fname"),
        "eval_script_name": task.get("eval_script_name"),
        "benchmark_path": abs_benchmark_path,
        "run_id": "eval",
        "predicted_code": submission or "",
    }

    print(r_kwargs)


    try:

        # log_kwargs = r_kwargs.copy()
        # predicted_code = log_kwargs.pop("predicted_code", "")
        eval_metrics, run_logs = evaluate_instance(**r_kwargs)

    except Exception as e:

        # Fallback EvalMetrics (safe defaults)
        eval_metrics = {
            "instance_id": task.instance_id,
            "valid_program": 0,
            "success_rate": 0,
            "codebert_score": 0.0,
            "log_info": "",
        }
        valid_program = eval_metrics.get("valid_program", 0)
        success_rate = eval_metrics.get("success_rate", 0)
        codebert_score = eval_metrics.get("codebert_score", 0.0)

        print(
            f"FAILED Task {task.instance_id} - "
            f"valid_program={valid_program}, "
            f"success_rate={success_rate}, "
            f"codebert_score={codebert_score}"
        )
    
    if isinstance(eval_metrics, BaseModel):
        score = eval_metrics.model_dump()
    elif isinstance(eval_metrics, dict):
        score = dict(eval_metrics)
    else:
        score = {}


    score["submitted"] = submitted

    # Logging  metrics
    valid_program = score.get("valid_program", 0)
    success_rate = score.get("success_rate", 0)
    codebert_score = score.get("codebert_score", 0.0)

    score['abs_code_path_submitted'] = prediction.final_path

    print(
        f"{'-'*20}"
        f"Task {task.instance_id} - "
        f"submitted={submitted}, "
        f"valid_program={valid_program}, "
        f"success_rate={success_rate}, "
        f"codebert_score={codebert_score}"
        f"abs_code_path_submitted={prediction.final_path}"
        f"{'-'*20}"
    )

    metrics = score 

    score_val = (metrics["success_rate"] + metrics["codebert_score"]) / 2
    pred =  dspy.Prediction(
        score=score_val,
        score_dict=score
    )

    print(f"Task {task.get("instance_id", '')} metrics with out feedback : {pred}" )

    return pred


def sab_score_with_feedback(example, prediction, trace=None):

    final_code = prediction.generated_code if hasattr(prediction, 'generated_code') else prediction
    print("submission :", final_code.replace("\n", "\\n")[:30])
    
    task = example
    submission = final_code 
    submitted =  True if final_code else False

    r_kwargs = {
        "instance_id": task.instance_id,
        "gold_program_name": task.get("gold_program_name"),
        "task_inst": task.get("task_inst"),
        "output_fname": task.get("output_fname"),
        "eval_script_name": task.get("eval_script_name"),
        "benchmark_path": abs_benchmark_path,
        "run_id": "eval",
        "predicted_code": submission or "",
    }

    print(r_kwargs)


    try:
        # log_kwargs = r_kwargs.copy()
        # predicted_code = log_kwargs.pop("predicted_code", "")
        eval_metrics, run_logs = evaluate_instance(**r_kwargs)

    except Exception as e:

        # Fallback EvalMetrics (safe defaults)
        eval_metrics = {
            "instance_id": task.instance_id,
            "valid_program": 0,
            "success_rate": 0,
            "codebert_score": 0.0,
            "log_info": "",
        }
        valid_program = eval_metrics.get("valid_program", 0)
        success_rate = eval_metrics.get("success_rate", 0)
        codebert_score = eval_metrics.get("codebert_score", 0.0)

        print(
            f"FAILED Task {task.instance_id} - "
            f"valid_program={valid_program}, "
            f"success_rate={success_rate}, "
            f"codebert_score={codebert_score}"
        )
    
    if isinstance(eval_metrics, BaseModel):
        score = eval_metrics.model_dump()
    elif isinstance(eval_metrics, dict):
        score = dict(eval_metrics)
    else:
        score = {}


    score["submitted"] = submitted

    # Logging  metrics
    valid_program = score.get("valid_program", 0)
    success_rate = score.get("success_rate", 0)
    codebert_score = score.get("codebert_score", 0.0)
    score['abs_code_path_submitted'] = prediction.final_path

    print(
        f"{'-'*20}"
        f"Task {task.instance_id} - "
        f"submitted={submitted}, "
        f"valid_program={valid_program}, "
        f"success_rate={success_rate}, "
        f"codebert_score={codebert_score}"
        f"abs_code_path_submitted={prediction.final_path}"
        f"{'-'*20}"
    )

    metrics = score


    score_val = (metrics["success_rate"] + metrics["codebert_score"]) / 2

    # feedback_txt = f"the expert solution for the problem : \n\n{task.gold_code} {'-'*10} \n\n current metrics : {str(metrics)}"
    
    pred =  dspy.Prediction(
        score=score_val,
        score_dict=score,
        feedback=str(score)
    )

    print(f"Task {task.get("instance_id", '')} metrics with feedback : {pred}" )


    return pred


def sab_score_with_gold_feedback(example, prediction, trace=None):

    final_code = prediction.generated_code if hasattr(prediction, 'generated_code') else prediction
    print("submission :", final_code.replace("\n", "\\n")[:30])
    
    task = example
    submission = final_code 
    submitted =  True if final_code else False

    r_kwargs = {
        "instance_id": task.instance_id,
        "gold_program_name": task.get("gold_program_name"),
        "task_inst": task.get("task_inst"),
        "output_fname": task.get("output_fname"),
        "eval_script_name": task.get("eval_script_name"),
        "benchmark_path": abs_benchmark_path,
        "run_id": "eval",
        "predicted_code": submission or "",
    }

    print(r_kwargs)


    try:
        # log_kwargs = r_kwargs.copy()
        # predicted_code = log_kwargs.pop("predicted_code", "")
        eval_metrics, run_logs = evaluate_instance(**r_kwargs)

    except Exception as e:

        # Fallback EvalMetrics (safe defaults)
        eval_metrics = {
            "instance_id": task.instance_id,
            "valid_program": 0,
            "success_rate": 0,
            "codebert_score": 0.0,
            "log_info": "",
        }
        valid_program = eval_metrics.get("valid_program", 0)
        success_rate = eval_metrics.get("success_rate", 0)
        codebert_score = eval_metrics.get("codebert_score", 0.0)

        print(
            f"FAILED Task {task.instance_id} - "
            f"valid_program={valid_program}, "
            f"success_rate={success_rate}, "
            f"codebert_score={codebert_score}"
        )
    
    if isinstance(eval_metrics, BaseModel):
        score = eval_metrics.model_dump()
    elif isinstance(eval_metrics, dict):
        score = dict(eval_metrics)
    else:
        score = {}


    score["submitted"] = submitted

    # Logging  metrics
    valid_program = score.get("valid_program", 0)
    success_rate = score.get("success_rate", 0)
    codebert_score = score.get("codebert_score", 0.0)

    score['abs_code_path_submitted'] = prediction.final_path

    print(
        f"{'-'*20}"
        f"Task {task.instance_id} - "
        f"submitted={submitted}, "
        f"valid_program={valid_program}, "
        f"success_rate={success_rate}, "
        f"codebert_score={codebert_score}"
        f"abs_code_path_submitted={prediction.final_path}"
        f"{'-'*20}"
    )

    metrics = score


    score_val = (metrics["success_rate"] + metrics["codebert_score"]) / 2

    feedback_txt = f"the expert solution for the problem : \n\n{task.gold_code} {'-'*10} \n\n current metrics : {str(metrics)}"
    
    pred =  dspy.Prediction(
        score=score_val,
        score_dict=score,
        feedback=feedback_txt
    )

    print(f"Task {task.get("instance_id", '')} metrics with feedback : {pred}" )


    return pred