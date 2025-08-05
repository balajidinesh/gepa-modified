# GEPA (Code Artifact)
This repository hosts the GEPA experiment reproduction code. This is only intended as a checkpoint to reproduce the results presented in the GEPA paper. To use GEPA, please refer to the main GEPA repository [https://github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa).

## Running GEPA Experiments
### Setup
Before starting, please ensure that you have `uv` installed:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Next, clone the DSPy and Arbor dependencies by executing the following command (we used a slightly modified version of DSPy and Arbor for our experiments, which will be downloaded by the script):
```
bash setup_gepa_repo.sh
```

(Only for running GRPO experiments) You must use the `pyproject_grpo.toml` instead of the default `pyproject.toml`:
```
mv pyproject_grpo.toml pyproject.toml
uv sync --no-install-package flash-attn
```

Now, run let's ensure the environment is synced:
```
uv sync
```

### Obtaining experiment data
The various data artifacts for all the experiments (like log files and DSPy .cache files) are hosted in Zenodo at [](). Download the file, and extract it to this directory under `experiment_runs_data`:
```
TODO: Add command to download and extract
```

### Configuring the experiments
[scripts/experiment_configs.py](scripts/experiment_configs.py) lists all the experiment configurations that can be modified. Specifically:
1. `BASE_EXPERIMENT_DIR`: This should point to the directory containing the extracted data artifacts
2. `LM_CONFIGS`: The list of language model configurations to run
3. `get_benchmarks`: List of benchmarks to run
4. `get_optimizers`: List of optimizer configurations to run (including MIPRO, GRPO, GEPA, and ablations)

### Generating experiment commands
After configuring the experiment, generate the list of launch commands by execution:
```
uv run python -m scripts.generate_launch_commands > launch_commands
```

Running this will produce a file `launch_commands` containing the command to be executed to perform each of the experiment.

### Executing the commands
The commands need to be executed in an environment, where `OPENAI_API_KEY` and `WANDB_API_KEY` are defined. [Arbor](https://github.com/Ziems/arbor) is used for local inference and GRPO training. For GPU configuration, have a look at the `.yaml` files in `gepa_artifact/utils/arbor`.

### Visualizing the results
Once the experiment commands have completed (or if you directly obtained the saved artifacts from Zenodo), `experiment_runs_data/` should contain the final result logs. Executing [scripts/generate_figures.ipynb](scripts/generate_figures.ipynb) will reproduce all the figures from the GEPA paper.

## Citation
If you use this repository, we kindly request that you cite:
```
@misc{agrawal2025gepareflectivepromptevolution,
      title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning}, 
      author={Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab},
      year={2025},
      eprint={2507.19457},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.19457}, 
}
```
