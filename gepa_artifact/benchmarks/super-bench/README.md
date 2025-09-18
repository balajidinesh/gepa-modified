# Super Benchmark

A clean, organized benchmark implementation for the Super dataset using DSPy React agents with fresh Docker containers.

## Structure

```
super-bench/
├── __init__.py           # Package exports
├── super_data.py         # SuperBenchmark class for data loading
├── super_program.py      # SuperReactAgent implementation
├── super_utils.py        # Evaluation functions and utilities
├── run_super_bench.py    # Main runner script
└── README.md            # This file
```

## Features

- **Fresh Docker containers** for each task evaluation
- **Configurable task IDs** - run specific Super benchmark instances
- **Clean evaluation metrics** - submission rate, output matching, landmark tracking
- **Organized code structure** following existing benchmark patterns
- **Easy-to-use runner script** with command line options

## Usage

### Basic Usage

```bash
# Run default tasks (mera, team, dir-gnn)
python run_super_bench.py

# Run specific task IDs
python run_super_bench.py --task_ids mera team

# Run a single task
python run_super_bench.py --single mera

# Run all available tasks
python run_super_bench.py --task_ids all

# Customize max iterations
python run_super_bench.py --task_ids mera --max_iters 50
```

### Available Task IDs

- `mera` - Meta-reasoning tasks
- `team` - Team collaboration tasks  
- `dir-gnn` - Directory GNN tasks
- `scicode` - Scientific code tasks
- `math-comp` - Mathematical computation tasks
- `ml-engineering` - ML engineering tasks

### Programmatic Usage

```python
from super_bench import SuperBenchmark, SuperReactAgent, super_metric
import dspy

# Setup DSPy
lm = dspy.LM("azure/gpt-4o")
dspy.configure(lm=lm)

# Create benchmark with specific task IDs
benchmark = SuperBenchmark(
    dataset_mode="test", 
    instance_ids=['mera', 'team']
)

# Get test examples
test_set = benchmark.get_test_set()

# Create agent
agent = SuperReactAgent(max_iters=100)

# Run on single example
example = test_set[0]
result = agent(
    query=example.query,
    github_repo=example.github_repo,
    git_commit=example.git_commit,
    instance_id=example.instance_id
)

# Evaluate
metrics = super_metric(example, result)
print(metrics)
```

## Requirements

- Python 3.8+
- DSPy framework
- aicodetools (for Docker runtime)
- datasets library
- pydantic
- dotenv
- Docker with `superbench:latest` image

## Environment Setup

Create a `.env` file with your Azure OpenAI credentials:

```
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
```

## Docker Container Management

The benchmark automatically:
- Creates fresh Docker containers for each task
- Provides clean tool instances (read_file, write_file, edit_file, run_command)
- Handles container cleanup between evaluations
- Logs all agent interactions and results

## Output

Results are saved to:
- `runs/` directory - individual task traces and metrics
- Console output - real-time progress and final summary

## Metrics

The benchmark tracks:
- **Submission rate**: % of tasks that produced valid outputs
- **Output match**: % of correct answers vs gold standard
- **Landmarks**: % of required checkpoints hit during execution
- **Overall score**: Average of output match and landmarks

## Integration

This benchmark is designed to integrate with:
- GEPA optimization framework (future)
- Existing benchmark evaluation pipelines
- Custom metric functions and feedback systems