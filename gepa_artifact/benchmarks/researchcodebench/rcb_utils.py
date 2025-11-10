import os
import shutil
import tempfile
import subprocess
from typing import Dict, Any, Tuple
import re

from pathlib import Path


REQUIRED_INSTANCE_FIELDS = [
    "problem_root_rel",
    "annotated_file_path",
    "snippet_name",
    "test_file_path",
]


def _ensure_required_fields(instance: Dict[str, Any], extra=[]) -> None:
    for key in (REQUIRED_INSTANCE_FIELDS + extra):
        if key not in instance:
            raise KeyError(f"Missing required field in instance: {key}")




def _prediction_to_lines(prediction_block: str):
    """Convert a text block to a list of lines with newline endings preserved."""
    lines = []
    for ln in prediction_block.splitlines():
        lines.append(ln if ln.endswith("\n") else ln + "\n")
    # Special case: if prediction_block ends with a newline, splitlines() drops it.
    # The above logic ensures each line ends with \n; trailing empty lines are fine.
    return lines


def _patch_file_by_markers(target_path: str, snippet_name: str, prediction_block: str) -> None:
    """
    Replace the code between paper2code markers for `snippet_name` by working
    with lines and preserving marker lines exactly. This avoids character-level
    slicing pitfalls and newline inconsistencies.
    """
    start_pat = re.compile(r'^\s*#\s*<paper2code\s+name="' + re.escape(snippet_name) + r'">\s*$')
    end_pat = re.compile(r'^\s*#\s*</paper2code\s+name="' + re.escape(snippet_name) + r'">\s*$')

    with open(target_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if start_pat.match(line):
            start_idx = i
            for j in range(i + 1, len(lines)):
                if end_pat.match(lines[j]):
                    end_idx = j
                    break
            break

    if start_idx is None or end_idx is None:
        raise ValueError(f"Snippet markers for '{snippet_name}' not found in {target_path}")

    new_lines = _prediction_to_lines(prediction_block)
    lines[start_idx + 1 : end_idx] = new_lines

    with open(target_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    
    return "".join(lines)


def _run_docker_test(mount_dir: str, test_entry_point: str, docker_image: str, timeout_seconds: int) -> Tuple[bool, int, str, str]:
    """Run tests inside the Docker image, mounting mount_dir at /workspace."""
    abs_mount = os.path.abspath(mount_dir)
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{abs_mount}:/workspace",
        "-w",
        "/workspace",
        docker_image,
        "python",
        test_entry_point,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
        success = proc.returncode == 0
        return success, proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return False, -1, e.stdout or "", (e.stderr or "") + "\nTimeoutExpired"
    except FileNotFoundError as e:
        # Docker not found on host
        return False, -1, "", f"Docker not found: {e}"


def rcb_score(example, prediction, trace=None):
    """
    Evaluate a prediction against a problem instance using a Dockerized test env.

    Parameters:
    - instance: JSONL record with required fields
    - prediction_block: code block string inserted as-is
    - docker_image: tag of the test environment image
    - timeout_seconds: host-side timeout for the test run

    Returns: dict with keys {success, exit_code, stdout, stderr, passed}
    """

    prediction_block = prediction.result if hasattr(prediction, 'result') else prediction
    instance = example 
    docker_image = "researchcodebench:latest"

    _ensure_required_fields(instance)

    problem_root_rel = instance.get("problem_root_rel",None)
    annotated_rel_path = instance.get("annotated_file_path",None)
    test_entry_point = instance.get("test_file_path",None)

    
    base_path =  Path('gepa_artifact/benchmarks/researchcodebench/')
    problem_root_rel = Path(problem_root_rel)
    problem_root_rel = (base_path / problem_root_rel)
    # Prepare temp workspace
    temp_root = tempfile.mkdtemp(prefix="rcb_eval_")
    temp_problem_dir = os.path.join(temp_root, os.path.basename(problem_root_rel))
    shutil.copytree(problem_root_rel, temp_problem_dir, dirs_exist_ok=True)

    # Patch annotated file by markers (no reliance on start/end indices)
    target_path = os.path.join(temp_problem_dir, annotated_rel_path)
    patched_file = _patch_file_by_markers(target_path, instance.get("snippet_name"), prediction_block)

    # Run tests in Docker
    success, exit_code, stdout, stderr = _run_docker_test(
        mount_dir=temp_problem_dir,
        test_entry_point=test_entry_point,
        docker_image=docker_image,
        timeout_seconds=60*2,
    )

    # Pass criteria: exit_code==0 and no obvious error keywords in stdout/stderr
    error_keywords = ("Error:", "Exception:", "Traceback", "failed")
    has_error_kw = any(k in (stdout or "") for k in error_keywords) or any(k in (stderr or "") for k in error_keywords)
    passed = bool(success and exit_code == 0 and not has_error_kw)

    return float(passed)



def rcb_score_with_feedback(example, prediction, trace=None):
    """
    Evaluate a prediction against a problem instance using a Dockerized test env.

    Parameters:
    - instance: JSONL record with required fields
    - prediction_block: code block string inserted as-is
    - docker_image: tag of the test environment image
    - timeout_seconds: host-side timeout for the test run

    Returns: dict with keys {success, exit_code, stdout, stderr, passed}
    """
    
    prediction_block = prediction.result if hasattr(prediction, 'result') else prediction
    instance = example 
    docker_image = "researchcodebench:latest"

    _ensure_required_fields(instance)

    problem_root_rel = instance.get("problem_root_rel",None)
    annotated_rel_path = instance.get("annotated_file_path",None)
    test_entry_point = instance.get("test_file_path",None)

    base_path =  Path('gepa_artifact/benchmarks/researchcodebench/')
    problem_root_rel = Path(problem_root_rel)
    problem_root_rel = (base_path / problem_root_rel)
    # Prepare temp workspace
    temp_root = tempfile.mkdtemp(prefix="rcb_eval_")
    temp_problem_dir = os.path.join(temp_root, os.path.basename(problem_root_rel))
    shutil.copytree(problem_root_rel, temp_problem_dir, dirs_exist_ok=True)

    # Patch annotated file by markers (no reliance on start/end indices)
    target_path = os.path.join(temp_problem_dir, annotated_rel_path)
    patched_file = _patch_file_by_markers(target_path, instance.get("snippet_name"), prediction_block)

    # Run tests in Docker
    success, exit_code, stdout, stderr = _run_docker_test(
        mount_dir=temp_problem_dir,
        test_entry_point=test_entry_point,
        docker_image=docker_image,
        timeout_seconds=60*2,
    )

    # Pass criteria: exit_code==0 and no obvious error keywords in stdout/stderr
    error_keywords = ("Error:", "Exception:", "Traceback", "failed")
    has_error_kw = any(k in (stdout or "") for k in error_keywords) or any(k in (stderr or "") for k in error_keywords)
    passed = bool(success and exit_code == 0 and not has_error_kw)

    return float(passed),{
        "task_id" : instance['task_id'],
        "success": success,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "passed": passed,
        }


def rcb_score_with_gold_feedback(example, prediction, trace=None):
    """
    Evaluate a prediction against a problem instance using a Dockerized test env.

    Parameters:
    - instance: JSONL record with required fields
    - prediction_block: code block string inserted as-is
    - docker_image: tag of the test environment image
    - timeout_seconds: host-side timeout for the test run

    Returns: dict with keys {success, exit_code, stdout, stderr, passed}
    """
    
    prediction_block = prediction.result if hasattr(prediction, 'result') else prediction
    instance = example 
    docker_image = "researchcodebench:latest"

    _ensure_required_fields(instance, extra=['gold_snippet'])

    problem_root_rel = instance.get("problem_root_rel",None)
    annotated_rel_path = instance.get("annotated_file_path",None)
    test_entry_point = instance.get("test_file_path",None)

    
    base_path =  Path('gepa_artifact/benchmarks/researchcodebench/')
    problem_root_rel = Path(problem_root_rel)
    problem_root_rel = (base_path / problem_root_rel)
    # Prepare temp workspace
    temp_root = tempfile.mkdtemp(prefix="rcb_eval_")
    temp_problem_dir = os.path.join(temp_root, os.path.basename(problem_root_rel))
    shutil.copytree(problem_root_rel, temp_problem_dir, dirs_exist_ok=True)

    # Patch annotated file by markers (no reliance on start/end indices)
    target_path = os.path.join(temp_problem_dir, annotated_rel_path)
    patched_file = _patch_file_by_markers(target_path, instance.get("snippet_name"), prediction_block)

    # Run tests in Docker
    success, exit_code, stdout, stderr = _run_docker_test(
        mount_dir=temp_problem_dir,
        test_entry_point=test_entry_point,
        docker_image=docker_image,
        timeout_seconds=60*2,
    )

    gold_snippet = instance.get('gold_snippet', None)

    # Pass criteria: exit_code==0 and no obvious error keywords in stdout/stderr
    error_keywords = ("Error:", "Exception:", "Traceback", "failed")
    has_error_kw = any(k in (stdout or "") for k in error_keywords) or any(k in (stderr or "") for k in error_keywords)
    passed = bool(success and exit_code == 0 and not has_error_kw)

    return float(passed),{
        "task_id" : instance['task_id'],
        "success": success,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "passed": passed,
        "gold_snippet" : gold_snippet
        }

