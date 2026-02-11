import dspy

from ..benchmark import BenchmarkMeta
from .sab_data import SABenchmark
from .sab_program import SABReactAgent
from .sab_utils import sab_score, sab_score_with_feedback, sab_score_with_gold_feedback

benchmark = [
    BenchmarkMeta(
        SABenchmark,
        [
            SABReactAgent(),
        ],
        sab_score,
        metric_with_feedback=sab_score_with_feedback,
    )
]


benchmark_with_gold = [
    BenchmarkMeta(
        SABenchmark,
        [
            SABReactAgent(),
        ],
        sab_score,
        metric_with_feedback=sab_score_with_gold_feedback,
    )
]