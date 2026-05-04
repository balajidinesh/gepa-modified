from ..benchmark import Benchmark
import dspy
from datasets import load_dataset


class SuperBenchmark(Benchmark):
    def init_dataset(self, with_gold=False):
        # Hardcoded instance ID splits [THE DATASPLIT IS MADE USING : Stratified Sampling ]
        train_instance_ids = ['glee','hype','pie-perf','rah-kbqa','safetybench','acqsurvey','dir-gnn','amos','discodisco']
        val_instance_ids = ['quantifying-stereotypes-in-language','mera','textbox','colbert','powerfulpromptft','logme-nlp','amrbart','curriculum_learning','multi3woz']
        test_instance_ids = ['team','cet','paraphrase-nli','galore','memorizing-transformers-pytorch','spa','unsupervisedhierarchicalsymbolicregression','robust_prompt_classifier','g-transformer','bert-lnl','upet','data_label_alignment','mode-connectivity-plm','blockskim','mbib','transpolymer','pira','dpt','conv_graph','inbedder','linkbert','mezo','pet','transnormerllm','align-to-distill','mixup-amp','parallel-context-windows']
        
        all_instance_ids = train_instance_ids + val_instance_ids + test_instance_ids

        # Load the Super dataset
        data_split = 'Expert'
        super_bench = load_dataset('allenai/super', data_split, split="all_examples")
        super_bench = super_bench.to_pandas()
        super_bench.rename(columns={'task_id': 'instance_id'}, inplace=True)

        # Process data and remove unnecessary keys
        keys_to_remove = ['solution_dependencies', 'solution']
        if with_gold : 
            keys_to_remove = []
        instances = super_bench.to_dict('records')

        # Create examples for all relevant instances
        all_examples = []
        for instance in instances:
            if instance['instance_id'] in all_instance_ids:
                for key in keys_to_remove:
                    instance.pop(key, None)

                # Use single input field pattern like other benchmarks
                # ex = dspy.Example(**instance).with_inputs("query",'git_commit','github_repo','instance_id','landmarks','answer')
                ex = dspy.Example(**instance).with_inputs("query",'git_commit','github_repo','instance_id')
                all_examples.append(ex)

        # Create splits based on hardcoded instance IDs
        self.train_set = [ex for ex in all_examples if ex.instance_id in train_instance_ids]
        self.val_set = [ex for ex in all_examples if ex.instance_id in val_instance_ids]
        self.test_set = [ex for ex in all_examples if ex.instance_id in test_instance_ids]

        # Set the combined dataset
        self.dataset = self.train_set + self.val_set + self.test_set