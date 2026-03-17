
# %pip install python-dotenv
# %uv add dspy
import sys
from pathlib import Path
# project_root = Path.cwd().parent
# sys.path.append(str(project_root))

## check aicodetools library


from gepa_setup import tlm, os , dspy , lm ,bench, sb_metas, time, threading, tiktoken, deque, BaseCallback, GEPAState, base_program

from extra_utils.cost_tracker import init_tracker, get_tracker
import dspy

tracker = init_tracker(log_dir="super_bench/runs/cost-super-var-wog", name="test")

print(base_program.named_predictors())



state = GEPAState.load('super_bench/runs/gepa-state-without-gold')

def idxmax(lst):
    """Return the index of the maximum value in a list."""
    max_val = max(lst)
    return lst.index(max_val)


gepa_state = state
best_prog_idx = idxmax(gepa_state.per_program_tracked_scores)
best_progs = gepa_state.program_candidates
best_prog = best_progs[best_prog_idx]

optimized_program = best_prog
latest_prog = best_progs[-1]


print(len(best_progs) ,best_prog_idx)



import dspy
import os
import json
import dspy


def safe_extract_score_dict(score_obj):
    """
    Robust extractor that handles:
        - DSPy score objects
        - JSON-loaded dicts with nested score_dict
        - flat dicts
        - numeric scores
    """

    if score_obj is None:
        return {}

    # Case 1: DSPy object
    if hasattr(score_obj, "score_dict"):
        return score_obj.score_dict or {}

    # Case 2: JSON structure: {"score": x, "score_dict": {...}}
    if isinstance(score_obj, dict):

        # If nested score_dict exists → use it
        if "score_dict" in score_obj and isinstance(score_obj["score_dict"], dict):
            return score_obj["score_dict"]

        # If already flat
        if any(k in score_obj for k in ["submitted", "output_match", "landmarks"]):
            return score_obj

        return {}

    # Case 3: numeric
    if isinstance(score_obj, (int, float)):
        return {"submitted": float(score_obj)}

    return {}

def load_scores_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    score_objects = []
    for item in data:
        score = item.get("score", {})
        score_objects.append(score)
        print(score)
    correct = 0
    total = len(score_objects)
    for s in score_objects:
        score_dict = safe_extract_score_dict(s)
        if score_dict.get("submitted", 0) == 1:
            correct += 1
    overall_accuracy = round(100 * correct / total, 2) if total > 0 else 0.0
    return overall_accuracy, score_objects

def add_to_jsonl(jsonl_path, record):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def get_completed_iters(jsonl_path):
    """
    Returns a set of program_index values already processed.
    Fast O(n) single read.
    """
    completed = set()

    if not os.path.exists(jsonl_path):
        return completed

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                completed.add(record.get("var_idx"))
            except:
                continue

    return completed

def metric_aggregator(overall_accuracy, score_objects):

    ntotal = len(score_objects)

    if ntotal == 0:
        return {
            "overall_accuracy": 0.0,
            "mean_submitted": 0.0,
            "mean_output_match": 0.0,
            "mean_landmarks": 0.0,
            "ntotal": 0,
        }

    total_submitted = 0.0
    total_output_match = 0.0
    total_landmarks = 0.0

    for score_obj in score_objects:
        score_dict = safe_extract_score_dict(score_obj)

        total_submitted += score_dict.get("submitted", 0.0)
        total_output_match += score_dict.get("output_match", 0.0)
        total_landmarks += score_dict.get("landmarks", 0.0)

    return {
        "overall_accuracy": overall_accuracy,
        "mean_submitted": round(total_submitted / ntotal, 4),
        "mean_output_match": round(total_output_match / ntotal, 4),
        "mean_landmarks": round(total_landmarks / ntotal, 4),
        "ntotal": ntotal,
    }
def metric_aggregator_from_score_objects(overall_accuracy, score_objects):
    """
    Robust metric aggregator
    """

    ntotal = len(score_objects)

    if ntotal == 0:
        return {
            "overall_accuracy": 0.0,
            "mean_submitted": 0.0,
            "mean_output_match": 0.0,
            "mean_landmarks": 0.0,
            "ntotal": 0,
        }

    total_submitted = 0.0
    total_output_match = 0.0
    total_landmarks = 0.0

    for score_obj in score_objects:
        score_dict = safe_extract_score_dict(score_obj)

        total_submitted += score_dict.get("submitted", 0.0)
        total_output_match += score_dict.get("output_match", 0.0)
        total_landmarks += score_dict.get("landmarks", 0.0)

    return {
        "overall_accuracy": overall_accuracy,
        "mean_submitted": round(total_submitted / ntotal, 4),
        "mean_output_match": round(total_output_match / ntotal, 4),
        "mean_landmarks": round(total_landmarks / ntotal, 4),
        "ntotal": ntotal,
    }


base_dir = "super_bench/runs/variance-wog"
os.makedirs(base_dir, exist_ok=True)

output_jsonl_path = os.path.join(base_dir, f"var-results-best-prog-{best_prog_idx}.jsonl")


completed_iters = get_completed_iters(output_jsonl_path)
skipped_iters = []

for i in range(4):

    print(f"\n========== Variance Run {i} ==========\n")

    # -------------------------------------------------
    # STEP 1: If already processed → skip
    # -------------------------------------------------
    if i in completed_iters:
        print(f"variance {i} already exists in JSONL. Skipping.")
        continue

    program_dir = os.path.join(base_dir, f"variance_{i}")
    os.makedirs(program_dir, exist_ok=True)

    react_json_path = os.path.join(program_dir, "optimized_react.json")

    overall_accuracy = None
    score_objects = None

    # -------------------------------------------------
    # STEP 2: If optimized_react.json exists → load
    # -------------------------------------------------
    if os.path.exists(react_json_path):
        print("Found optimized_react.json. Loading...")

        try:
            overall_accuracy, score_objects = load_scores_from_json(react_json_path)

        except Exception as e:
            print(f"Failed loading optimized_react.json: {e}")

    # -------------------------------------------------
    # STEP 3: If still no scores → evaluate
    # -------------------------------------------------
    if score_objects is None:
        print("Running evaluation...")

        try:
            evaluator = dspy.Evaluate(
                devset=bench.test_set,
                metric=sb_metas[0].metric,
                num_threads=8,
                display_table=True,
                display_progress=True,
                max_errors=100 * len(bench.test_set),
                provide_traceback=True,
                failure_score=0,
                return_all_scores=True,
                return_outputs=True,
                save_as_json=react_json_path
            )

            results = evaluator(best_prog)

            overall_accuracy, detailed_results, score_objects = results

        except Exception as e:
            print(f"Evaluation failed: {e}")

            # Retry loading JSON if it was partially saved
            if os.path.exists(react_json_path):
                print("Retrying with saved optimized_react.json...")
                try:
                    overall_accuracy, score_objects = load_scores_from_json(react_json_path)
                except Exception as e2:
                    print(f"Retry failed: {e2}")
                    skipped_iters.append(i)
                    continue
            else:
                skipped_iters.append(i)
                continue

    # -------------------------------------------------
    # STEP 4: Aggregate and Save
    # -------------------------------------------------
    try:
        complete_result = metric_aggregator(overall_accuracy, score_objects)

        prog_instructions = {
            name: pred.signature.instructions
            for name, pred in best_prog.named_predictors()
        }

        record = {
            "program_index": best_prog_idx,
            "var_idx" : i, 
            "prog_instructions": prog_instructions,
            "complete_result": complete_result,
        }

        add_to_jsonl(output_jsonl_path, record)

        print(f"Saved results for program {i}")

    except Exception as e:
        print(f"Aggregation failed: {e}")

        # Final retry from JSON
        if os.path.exists(react_json_path):
            try:
                overall_accuracy, score_objects = load_scores_from_json(react_json_path)
                complete_result = metric_aggregator(overall_accuracy, score_objects)

                record = {
                    "program_index": best_prog_idx,
                    "var_idx" : i, 
                    "prog_instructions": prog_instructions,
                    "complete_result": complete_result,
                }

                add_to_jsonl(output_jsonl_path, record)
                print(f"Recovered and saved program var run {i}")

            except Exception:
                skipped_iters.append(i)
        else:
            skipped_iters.append(i)

# -------------------------------------------------
# FINAL REPORT
# -------------------------------------------------

print("\n==============================")
print("Processing complete.")
print(f"Skipped iterations: {skipped_iters}")
print("==============================")

