<h1 align='center'>
Learning a Policy Repository through Adaptive Environment Assignment
</h1>

# Experiment Overview

In open-ended learning environments, agents are gradually exposed to novel challenges over time. Unlike conventional multi-task settings, it is often infeasible to train a generalist policy on all encountered challenges in parallel, a strategy commonly used to improve training stability and mitigate forgetting. In this project, we introduce a policy repository framework that partitions the heterogeneous environment space by allowing different policies to specialize in distinct niches. Specialization is implicitely encouraged through an adaptive environment assignment mechanism, where each policy is preferentially trained on environments that align with its area of competence. We investigate several assignment strategies by varying the scoring function used to estimate how well a given environment matches a policy's niche.

For more details on the methodology and results, please consult the slides bellow.
<div style="position: relative; width: 100%; height: 0; padding-top: 56.2500%;
 padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
 border-radius: 8px; will-change: transform;">
  <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
    src="https://www.canva.com/design/DAHNLnLiW6Q/wI_59BRKBRtChhki0LC3Kg/view?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
  </iframe>
</div>
<a href="https:&#x2F;&#x2F;www.canva.com&#x2F;design&#x2F;DAHNLnLiW6Q&#x2F;wI_59BRKBRtChhki0LC3Kg&#x2F;view?utm_content=DAHNLnLiW6Q&amp;utm_campaign=designshare&amp;utm_medium=embeds&amp;utm_source=link" target="_blank" rel="noopener">AMG Final Presentation</a> by Sobhan Nili

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
> For running the DINO expierment, if using [DINOv3](https://huggingface.co/docs/transformers/en/model_doc/dinov3) you need to first request for getting access on huggningface website, and export your personal `HF_TOKEN` in terminal running the scripts. 

# Scripts
- `scripts/run.py`: Run the policy repository training and plot results for a single configuration. Run `python scripts/run.py -h` for details and possible kwarg overrides.
- `scripts/plot_results.py`: Re-generate the plots for a specific experiment. Run `python scripts/plot_results.py -h` for details.
- `scripts/compare_runs.py`: Generate stats and plot for comparing different repository logics / hyperparameters. Run `python scripts/compare_runs.py -h` for details.
- `scripts/runai_sweep.py`: Submit a batch of parallel jobs to EPFL RCP. Expects the code to be at user home directory on cluster.

# Analytics


# Reproduction
First put the desired config values in [configs/config.yaml](configs/config.yaml) and corresponding repository configs. Consult the [config readme](configs/config.yaml) for overview of the params. For repoducing the results from the slides, use the config file in its current state in the repo with across 4 random seeds (0, 2, 90, 42).

Then use `scripts/run.py` (possibly with kwarg overrides) or run a hyperparam sweep using `scripts/sweep.sh`. By default, output will be stored at `outputs/experiments`.
After that for generating the comparison plots across different repository logics, run `scripts/compare_runs` which generates outputs at `outputs/run_comparisons`.

# Legacy Scripts
> To train the transformer policy network on randomly generated environments with periodic checkpointing, use `thirdparty/kinetix/experiments/bacbone.py` script. Corresponding config is at `../configs/backbone.yaml`

> To load a pre-trained policy and finetune it on a list of defined levels, use `thirdparty/kinetix/experiments/finetune.py` with corresponding config at `../configs/finetune.yaml`. It enables to choose among different strategies for finetuning (actor / critic heads only, freezing the trunk and etc.). Refer to config file for options.