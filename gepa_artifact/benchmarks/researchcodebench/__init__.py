import dspy

from ..benchmark import BenchmarkMeta
from .rcb_data import RCBenchmark
from .rcb_program import RCB

from .rcb_utils import rcb_score, rcb_score_with_feedback, rcb_score_with_gold_feedback



benchmark = [
    BenchmarkMeta(
        RCBenchmark,
        [
            RCB(),
        ],
        rcb_score,
        metric_with_feedback=rcb_score_with_feedback,
    )
]


benchmark_with_gold = [
    BenchmarkMeta(
        RCBenchmark,
        [
            RCB(),
        ],
        rcb_score,
        metric_with_feedback=rcb_score_with_gold_feedback,
    )
]