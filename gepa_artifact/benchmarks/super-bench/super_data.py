from ..benchmark import Benchmark
import dspy
from datasets import load_dataset


class SuperBenchmark(Benchmark):
    def __init__(self, dataset_mode="lite", instance_ids=None):
        self.instance_ids = instance_ids or ['mera', 'team', 'dir-gnn']
        super().__init__(dataset_mode)

    def init_dataset(self):
        data_split = 'Expert'
        try:
            super_bench = load_dataset('allenai/super', data_split, split="all_examples")
            super_bench = super_bench.to_pandas()
            super_bench.rename(columns={'task_id': 'instance_id'}, inplace=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

        keys_to_remove = ['solution_dependencies', 'solution']
        instances = super_bench.to_dict('records')
        
        self.dataset = []
        for instance in instances:
            if instance['instance_id'] not in self.instance_ids:
                continue
            
            for key in keys_to_remove:
                instance.pop(key, None)
            
            ex = dspy.Example(**instance).with_inputs("instance_id", "query", "github_repo", "git_commit")
            self.dataset.append(ex)