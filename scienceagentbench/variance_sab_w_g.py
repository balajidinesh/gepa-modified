import sys
from pathlib import Path


from gepa_setup import tlm, os , dspy , lm ,bench, sab_metas, time, threading, tiktoken, deque, BaseCallback, GEPAState, base_program

from extra_utils.cost_tracker import init_tracker, get_tracker
import dspy
import os
import json


print(base_program.named_predictors())



state = GEPAState.load('scienceagentbench/runs/sab-train-with-gold')

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


def safe_extract_score_dict(score_obj):
    if score_obj is None:
        return {}

    if hasattr(score_obj, "score_dict"):
        return score_obj.score_dict or {}

    if isinstance(score_obj, dict):
        if "score_dict" in score_obj and isinstance(score_obj["score_dict"], dict):
            return score_obj["score_dict"]

        if any(k in score_obj for k in ["valid_program", "success_rate", "codebert_score"]):
            return score_obj

        return {}

    if isinstance(score_obj, (int, float)):
        return {"success_rate": float(score_obj)}

    return {}


def load_scores_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    score_objects = []
    for item in data:
        score_objects.append(item.get("score", {}))

    total = len(score_objects)
    correct = 0
    for s in score_objects:
        sd = safe_extract_score_dict(s)
        if sd.get("success_rate", 0) == 1:
            correct += 1

    overall_accuracy = round(100 * correct / total, 2) if total > 0 else 0.0
    return overall_accuracy, score_objects


def add_to_jsonl(jsonl_path, record):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_completed_iters(jsonl_path):
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                completed.add(record.get("var_idx"))
            except Exception:
                continue
    return completed


def metric_aggregator(overall_accuracy, score_objects):
    ntotal = len(score_objects)
    if ntotal == 0:
        return {
            "overall_accuracy": 0.0,
            "mean_valid_program": 0.0,
            "mean_success_rate": 0.0,
            "mean_codebert_score": 0.0,
            "mean_combined": 0.0,
            "ntotal": 0,
        }

    total_valid = 0.0
    total_sr = 0.0
    total_cb = 0.0
    total_combined = 0.0

    for s in score_objects:
        sd = safe_extract_score_dict(s)
        vp = float(sd.get("valid_program", 0) or 0)
        sr = float(sd.get("success_rate", 0) or 0)
        cb = float(sd.get("codebert_score", 0.0) or 0.0)
        total_valid += vp
        total_sr += sr
        total_cb += cb
        total_combined += (sr + cb) / 2.0

    return {
        "overall_accuracy": overall_accuracy,
        "mean_valid_program": round(total_valid / ntotal, 4),
        "mean_success_rate": round(total_sr / ntotal, 4),
        "mean_codebert_score": round(total_cb / ntotal, 4),
        "mean_combined": round(total_combined / ntotal, 4),
        "ntotal": ntotal,
    }


base_dir = "scienceagentbench/runs/variance"
os.makedirs(base_dir, exist_ok=True)

output_jsonl_path = os.path.join(base_dir, f"var-results-best-prog-{best_prog_idx}.jsonl")


completed_iters = get_completed_iters(output_jsonl_path)
skipped_iters = []

for i in range(5):

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
                metric=sab_metas[0].metric,
                num_threads=8,
                display_table=True,
                display_progress=True,
                max_errors=10 * len(bench.test_set),
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
            "complete_result": complete_result,
            "prog_instructions": prog_instructions,
        }

        add_to_jsonl(output_jsonl_path, record)

        print(f"Saved results for var run {i}")

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
                    "complete_result": complete_result,
                    "prog_instructions": prog_instructions,
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
