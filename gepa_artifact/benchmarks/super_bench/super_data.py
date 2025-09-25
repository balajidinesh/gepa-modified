from ..benchmark import Benchmark
import dspy
from datasets import load_dataset


class SuperBenchmark(Benchmark):
    def init_dataset(self):
        # Hardcoded instance ID splits
        train_instance_ids = ['mera', 'team']
        val_instance_ids = ['hype', 'textbox']
        test_instance_ids = ['dir-gnn', 'mbib']
        all_instance_ids = train_instance_ids + val_instance_ids + test_instance_ids

        # Load the Super dataset
        data_split = 'Expert'
        super_bench = load_dataset('allenai/super', data_split, split="all_examples")
        super_bench = super_bench.to_pandas()
        super_bench.rename(columns={'task_id': 'instance_id'}, inplace=True)

        # Process data and remove unnecessary keys
        keys_to_remove = ['solution_dependencies', 'solution']
        instances = super_bench.to_dict('records')

        # Create examples for all relevant instances
        all_examples = []
        for instance in instances:
            if instance['instance_id'] in all_instance_ids:
                for key in keys_to_remove:
                    instance.pop(key, None)

                # Use single input field pattern like other benchmarks
                ex = dspy.Example(**instance).with_inputs("query")
                all_examples.append(ex)

        # Create splits based on hardcoded instance IDs
        self.train_set = [ex for ex in all_examples if ex.instance_id in train_instance_ids]
        self.val_set = [ex for ex in all_examples if ex.instance_id in val_instance_ids]
        self.test_set = [ex for ex in all_examples if ex.instance_id in test_instance_ids]

        # Set the combined dataset
        self.dataset = self.train_set + self.val_set + self.test_set