# Reproduction
- Have uv package manager installed on the machine.
- Run `uv sync` to install dependencies.
- Activate the venv by `source .venv/bin/activate`
- To speed up jax compilation, set the `JAX_COMPILATION_CACHE_DIR` env var.

The following need to be installed based on the available cuda version:
- Jax: `uv pip install -U "jax[cuda12]"`

- Torch: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`


> To train the transformer policy network on randomly generated environments with periodic checkpointing, use `thirdparty/kinetix/experiments/bacbone.py` script. Corresponding config is at `../configs/backbone.yaml`

> To load a pre-trained policy and finetune it on a list of defined levels, use `thirdparty/kinetix/experiments/finetune.py` with corresponding config at `../configs/finetune.yaml`. It enables to choose among different strategies for finetuning (actor / critic heads only, freezing the trunk and etc.). Refer to config file for options.