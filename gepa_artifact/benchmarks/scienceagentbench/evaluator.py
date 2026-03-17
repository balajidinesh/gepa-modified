import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class EvalMetrics(BaseModel):
    instance_id: str
    valid_program: int
    success_rate: int
    codebert_score: float
    # Aggregated logs from predicted program exec and eval script.
    # JSON string containing keys: pred_stdout, pred_stderr, eval_stdout, eval_stderr, eval_log_info
    log_info: str



def _append_envs(cmd, var_names):
    """Append environment variables to docker run command if present locally.
    var_names: iterable of env var names to forward into the container.
    """
    for name in var_names:
        val = os.environ.get(name)
        if val is not None and val != "":
            cmd += ["-e", f"{name}={val}"]
    return cmd


def evaluate_instance( # Keys are explicitly passed; no CSV reads.
        # Predicted code is written to a temp dir and mounted; no env passing.
    instance_id: str,
    gold_program_name: str,
    task_inst: str,
    output_fname: str,
    eval_script_name: str,
    benchmark_path: str,
    run_id: str = "codex-run",
    predicted_code: str | None = None,
) :
    """
    Host-side wrapper that runs the single-image container for one instance with explicit inputs and returns metrics.
    Requires that `pred_program_path` contains `pred_{gold_program_name}`.
    """
    # Ensure image is present (build if missing)
    dockerfile = Path("eval/Dockerfile")
    res = subprocess.run(["docker", "image", "inspect", "sab-eval:latest"], capture_output=True)
    if res.returncode != 0:
        raise RuntimeError("Docker image 'sab-eval:latest' not found. Please build it with: docker build -t sab-eval:latest -f eval/Dockerfile . (run from repo root)")

    # Run the container with envs; read metrics from stdout
    tmpdir_pred = tempfile.TemporaryDirectory()
    pred_dir = Path(tmpdir_pred.name)
    pred_dir.mkdir(parents=True, exist_ok=True)
    benchmark = Path(benchmark_path)

    # Write predicted code into temp pred_programs dir
    if predicted_code is None:
        raise ValueError("predicted_code must be provided")
    pred_file = pred_dir / f"pred_{gold_program_name}"
    pred_file.write_text(predicted_code, encoding="utf-8")

    # Write input.json for runner
    # Environment will carry required fields; no input.json

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{benchmark.resolve()}:/benchmark",
        "-v", f"{pred_dir.resolve()}:/program_to_eval",
        "-e", f"GOLD_PROGRAM_NAME={gold_program_name}",
        "-e", f"TASK_INST={task_inst}",
        "-e", f"OUTPUT_FNAME={output_fname}",
        "-e", f"EVAL_SCRIPT_NAME={eval_script_name}",
    ]

    timeout_sec = 600
    # TODO check potential api key/ env vars leaks in logs, cause of above stringified command

    # Forward relevant LLM/LiteLLM provider credentials if present (model-agnostic)
    default_envs = [
        "AZURE_API_KEY", "AZURE_API_VERSION", "AZURE_API_BASE", 
    ]
    cmd = _append_envs(cmd, default_envs)
    cmd += ["sab-eval:latest", "python", "/app/eval_runner.py"]
    try:
        proc = subprocess.run(
        ["timeout", "-k", "10", str(timeout_sec)] + cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec + 20
    )
    except subprocess.TimeoutExpired as e:
        print(f"Evaluation timed out after {timeout_sec}s")
        print("Partial STDOUT:", e.stdout)
        print("Partial STDERR:", e.stderr)

        try:
            tmpdir_pred.cleanup()
        except Exception:
            pass

        return EvalMetrics(
            instance_id=instance_id,
            valid_program=0,
            success_rate=0,
            codebert_score=0.0,
            log_info="timeout",
            run_id=run_id,
            result_path="",
        ), "timeout"
    except subprocess.CalledProcessError as e:
        print("Command failed")
        print("Return code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    # Read result.json written under /program_to_eval
    # print(proc)
    result_json = pred_dir / "result.json"
    with open(result_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # Cleanup temp dir
    try:
        tmpdir_pred.cleanup()
    except Exception:
        pass

    return EvalMetrics(
        instance_id=instance_id,
        valid_program=int(payload.get("valid_program", 0)),
        success_rate=int(payload.get("success_rate", 0)),
        codebert_score=float(payload.get("codebert_score", 0.0)),
        log_info=str(payload.get("log_info", "")),
        run_id=run_id,
        result_path="",
    ), ''
