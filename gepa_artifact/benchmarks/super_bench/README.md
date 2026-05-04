# Super Benchmark

A clean, organized benchmark implementation for the Super dataset using DSPy React agents with fresh Docker containers.

## Structure

```
super-bench/
├── __init__.py           # Package exports
├── super_data.py         # SuperBenchmark class for data loading
├── super_program.py      # SuperReactAgent implementation
├── super_utils.py        # Evaluation functions and utilities
├── run_super_bench.py    # Main runner script [not validated]
└── README.md            # This file
```




## Requirements

- Python 3.8+ [please check we used Python 3.12.11 ]
- DSPy framework
- aicodetools (for Docker runtime) need this for coding agent
- datasets library
- pydantic
- dotenv
- Docker with `superbench:latest` image


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


## Environment Setup

Create a `.env` file with your Azure OpenAI credentials:

```env
# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_openai_api_key
AZURE_API_BASE=https://your-resource-name.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
AZURE_AD_TOKEN=your_azure_ad_token_if_using_aad_auth
AZURE_API_TYPE=azure
```

### DSPy Azure Configuration

In your code, use the deployment name with the `azure/` prefix:

```python
import dspy

# Configure DSPy with Azure OpenAI
lm = dspy.LM("azure/your_deployment_name")
dspy.configure(lm=lm)
```


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
