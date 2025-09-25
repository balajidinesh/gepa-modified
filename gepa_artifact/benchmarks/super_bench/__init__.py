import dspy

from ..benchmark import BenchmarkMeta
from .super_data import SuperBenchmark
from .super_program import SuperReactAgent
from .super_utils import super_score, super_score_with_feedback

benchmark = [
    BenchmarkMeta(
        SuperBenchmark,
        [
            SuperReactAgent(),
        ],
        super_score,
        metric_with_feedback=super_score_with_feedback,
    )
]