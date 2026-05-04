Purpose
- Standalone, single-image evaluation for SAB instances. No CSV reads, no persistent logs.

Build Image
- Run from repository root (pwd must be the repo root): (or directory containing parent folder of docker file)
  - `docker build -t sab-eval:latest -f base/Dockerfile .`

  - also addiyionally build the `benchmarks/scienceagentbench/base/eval-docker/Dockerfile` for running our coding agent

Inputs
- Download the zip from the scienceagentbench github and provide the paths in the `sab_utils.py` 

we used : 

```
abs_benchmark_path = '/mnt/d/sab/ScienceAgentBench/benchmark'
abs_dataset_path = '/mnt/d/sab/ScienceAgentBench/benchmark/datasets'
```

- `gold_program_name`, `task_inst`, `output_fname`, `eval_script_name`.
- `predicted_code`: the agent’s predicted program source (string).
