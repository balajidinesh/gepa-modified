from lm_setup import tlm, os , dspy , lm , time, threading, tiktoken, deque, BaseCallback



from gepa_artifact.benchmarks.super_bench.super_utils import FinishResponse
from gepa_artifact.benchmarks.super_bench import benchmark_with_gold as sb_metas


bench = sb_metas[0].benchmark(with_gold=True)
# len(bench.train_set), len(bench.val_set), len(bench.test_set)


import pprint
# pprint.pprint(bench.train_set[0])

program = sb_metas[0].program[0]
# program




import dspy
from gepa_artifact.gepa.gepa import GEPA,GEPAState
from gepa_artifact.utils.capture_stream_logger import Logger

import time




runs_dir = os.path.join(os.getcwd(), "runs", time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(runs_dir, exist_ok=True)
gepa_logger = Logger(os.path.join(runs_dir, "run_log.txt"))


if sb_metas[0].feedback_fn_maps is None or sb_metas[0].feedback_fn_maps[0] is None:
    def feedback_func(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
        pred = sb_metas[0].metric_with_feedback(module_inputs, module_outputs, None)
        return {
            "feedback_score": pred.score,
            "feedback_text": pred.feedback,
        }

    feedback_fn_map = {k:feedback_func for k, v in program.named_predictors()}
else:
    feedback_fn_map = sb_metas[0].feedback_fn_maps[0]


optimizer = GEPA(
    named_predictor_to_feedback_fn_map=feedback_fn_map,
    knowledgebase_qe=None,
    metric=sb_metas[0].metric,
    run_linearized_gepa=False,
    use_merge=True, 
    teacher_lm = tlm,
    set_for_merge_minibatch='val', 
    track_scores_on='val',
    max_iterations=20,
    run_dir=runs_dir,
    logger=gepa_logger,
    num_threads=9)
## Optimize the program with GEPA

base_program = program.deepcopy()

