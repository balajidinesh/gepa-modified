import dspy

from ..benchmark import BenchmarkMeta
from .rcb_data import RCBenchmark
from .rcb_program import #implement 

from .rcb_utils import rcb_score, rcb_score_with_feedback, rcb_score_with_gold_feedback

benchmark = [
    BenchmarkMeta(
        RCBenchmark,
        [
            SuperReactAgent(),
        ],
        rcb_score,
        metric_with_feedback=rcb_score_with_feedback,
    )
]


benchmark_with_gold = [
    BenchmarkMeta(
        RCBenchmark,
        [
            SuperReactAgent(),
        ],
        rcb_score,
        metric_with_feedback=rcb_score_with_gold_feedback,
    )
]