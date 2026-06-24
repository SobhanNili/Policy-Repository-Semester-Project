<h1 align='center'>
Learning a Policy Repository through Adaptive Environment Assignment
</h1>

# Experiment Overview
In open-ended learning environments, agents are gradually exposed to novel challenges over time. Unlike conventional multi-task settings, it is often infeasible to train a generalist policy on all encountered challenges in parallel, a strategy commonly used to improve training stability and mitigate forgetting. In this project, we introduce a policy repository framework that partitions the heterogeneous environment space by allowing different policies to specialize in distinct niches. Specialization is implicitely encouraged through an adaptive environment assignment mechanism, where each policy is preferentially trained on environments that align with its area of competence. We investigate several assignment strategies by varying the scoring function used to estimate how well a given environment matches a policy's niche.

For more details on the methodology and results, please consult the [slides here](https://apr-amg.my.canva.site/). 

Semester project at [Mathis Group](cnai.epfl.ch), by Sobhan Nili Ahmadabadi under supervision of Chengkun Li & Merkourios Simos.

# Setting Up Development Environment
Steps:
1. Have uv package manager installed on the machine.
2. Run `uv sync` at project root to install dependencies.
3. Activate the venv by `source .venv/bin/activate`

> [!TIP]
> To (heavily) speed up experiments, set the `JAX_COMPILATION_CACHE_DIR` env var so that jax doesn't have to recompile the code every time.

The following need to be installed after `uv sync` based on the available cuda version:
- Jax: `uv pip install -U "jax[cuda12]"`
- Torch: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

> [!NOTE]
> For running the DINO experiment, if using [DINOv3](https://huggingface.co/docs/transformers/en/model_doc/dinov3) you need to first request for getting access on huggningface website, and export your personal `HF_TOKEN` in terminal running the scripts. 

> [!NOTE]
> For running the policy latent experiment, first train a transformer policy and save the checkpoint using `scripts/train_ppo.py`.

# Scripts
- `scripts/run.py`: Run the policy repository training and plot results for a single configuration. Run `python scripts/run.py -h` for details and possible kwarg overrides. The configuration is detailed in [here](docs/config.md).
- `scripts/plot_results.py`: Re-generate the plots for a specific experiment. Run `python scripts/plot_results.py -h` for details.
- `scripts/compare_runs.py`: Generate stats and plot for comparing different repository logics / hyperparameters. Run `python scripts/compare_runs.py -h` for details.
- `scripts/train_ppo.py`: based on the `third_party/kinetix/experiments/ppo.py` to train a policy on training environments in parallel, producing a checkpoint to be used in policy-latent similarity scoring function.
- `scripts/runai_sweep.py`: Submit a batch of parallel jobs to EPFL RCP. Expects the code to be at user home directory on cluster.

# Reproduction
First put the desired config values in [configs/config.yaml](configs/config.yaml) and corresponding repository configs. Consult the [config readme](docs/config.md) for overview of the params. For repoducing the results from the slides, use the config file in its current state in the repo with across 4 random seeds (0, 2, 90, 42).
Run the `scripts/train_ppo.py` after adjusting destired training envrionments on its [corresponding config](./configs/ppo_training/config.yaml), to get a policy checkpoint to use policy latent metching repo.  

Then use `scripts/run.py` (possibly with kwarg overrides) or run a hyperparam sweep using `scripts/sweep.sh`. By default, output will be stored at `outputs/experiments`.
After that for generating the comparison plots across different repository logics, run `scripts/compare_runs` which generates outputs at `outputs/run_comparisons`.

# Legacy Scripts
To train the transformer policy network on randomly generated environments with periodic checkpointing, use `thirdparty/kinetix/experiments/bacbone.py` script. Corresponding config is at `../configs/backbone.yaml`

To load a pre-trained policy and finetune it on a list of defined levels, use `thirdparty/kinetix/experiments/finetune.py` with corresponding config at `../configs/finetune.yaml`. It enables to choose among different strategies for finetuning (actor / critic heads only, freezing the trunk and etc.). Refer to config file for options.