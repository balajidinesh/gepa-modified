from lm_setup import tlm, os , dspy , lm , time, threading, tiktoken, deque, BaseCallback

from gepa_artifact.benchmarks.scienceagentbench import benchmark as sab_metas


bench = sab_metas[0].benchmark()
# len(bench.train_set), len(bench.val_set), len(bench.test_set)
import pprint

# pprint.pprint(bench.train_set[0])

## Load the program and display the program
# The program is a 3-module system, each of which handles the urgency, sentiment and categories classification respectively
program = sab_metas[0].program[0]

import dspy
from gepa_artifact.gepa.gepa import GEPA,GEPAState
from gepa_artifact.utils.capture_stream_logger import Logger

import time



runs_dir = os.path.join(os.getcwd(), "runs", 'sab-test-trained-with-gold')
os.makedirs(runs_dir, exist_ok=True)

gepa_logger = Logger(os.path.join(runs_dir, "run_log.txt"))


if sab_metas[0].feedback_fn_maps is None or sab_metas[0].feedback_fn_maps[0] is None:
    def feedback_func(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
        pred = sab_metas[0].metric_with_feedback(module_inputs, module_outputs, None)
        return {
            "feedback_score": pred.score,
            "feedback_text": pred.feedback,
        }

    feedback_fn_map = {k:feedback_func for k, v in program.named_predictors()}
else:
    feedback_fn_map = sab_metas[0].feedback_fn_maps[0]

optimizer = GEPA(
    named_predictor_to_feedback_fn_map=feedback_fn_map,
    knowledgebase_qe=None,
    metric=sab_metas[0].metric,
    run_linearized_gepa=False,
    use_merge=True, 
    teacher_lm = tlm,
    set_for_merge_minibatch='val', 
    track_scores_on='val',
    max_iterations=15,
    num_dspy_examples_per_gepa_step=7,
    run_dir=runs_dir,
    logger=gepa_logger,
    num_threads=8
    )

base_program = program.deepcopy()
