# Configuration

## Kinetix Base Config
The configuration for running the policy repository experiments is built on top of the configurations from Kinetix (composed using hydra). 

The configs from Kinetix determine the policy / PPO params and also the training levels. For detailed information on what they do, consult their documentation: [Kinetix Config](../third_party/kinetix/docs/configs.md).

## Common Repository Experiment Configs
- `seed`: The seed to used as the inital jax PRNG key which sets the course for entire randomization in the experiments, <ins>including train / eval environemnet splitting.</ins>

- `defaults/repo`: The hyperparams for repository experiments. See [here](#repository-specific-params).

- `num_copies_per_env`: For each env, number of dublicates to create when training. The number of copies is passed to the trianing loop, to be stepped in parallel. This is done to increase the stability of PPO training.

- `num_updates`: Total number of PPO updates per repository iteration. This is per each environment (for each of the copies).

- `num_repo_updates`: Number of repository training loops, per which the env sampling -> assginment -> PPO training -> evaluation is done.

- `num_policies`: Number of policies inside the repository.

- `candidate_pool_size`: Number of envs sampled from the training env pool (`train_levels`) at each repository step. Should be adjusted in a way that the total number of envs passed for PPO training is divisable by number of PPO minibatches `learning/num_minibatches`.

- `eval_ratio`: When `defaults/eval` is null, this is used to randomely split the `defaults/train_levels` into trian and evaluation sets.

- `eval_rollout_steps`: Number of env steps (per each parallel copy) for frozen policy evaluation stats. Note that each kinetix episode by default can take max 256 steps. 

- `evaluate_finetune_steps`: Number of PPO update steps for fintuning the policy before frozen evaluation during the repository evaluation stage. The budget shouldn't be so high so that it enables the policy to learn the task from very scratch, defying the repositorie's purpose of providing starting points.

- `pretrain_eval_finetune_steps`: Number of steps to finetune the policy on the env before frozen evaluation. Used for affinity score - performance correlation diagnostics and doesn't affect the end results.

- `diagnostic_finetune_frequency`: frequency (w.r.t `num_repo_updates`) to run the policy evaluation with finetuning. Serves for diagnostic puproses. Takes some time, so reduce if needed.

- `assignment_mode`: How to derive the env <-> policy assginemnt mask based on affinity scores. Applies along the env score axis. 
Possible modes:
    - `greedy`: random argmax (argmax with random tie-breaking). This is what used in experiments.
    - `softmax`: softmax with temprature coming from `assignment_temperature`
    - `epsilon_greedy`: epsilon greedy with exploration rate controlled via `assignment_epsilon`.

## Repository-Specific Params

Located in [`configs/repo`](../configs/repo/).

- Common configurations between environment-embedding based approaches (dino / policy latent and morphology):
    - `history_decay`: 
    - `sigma`: The RBF kernel sigma, operating on eclucidian distance of env embeddings.

- `dino_matching`:
    - `model_name`: The dino checkpoint path in huggingface. Make sure to have set `HF_TOKEN ` env variable if it is DINOv3 as it requires authentication.
    - 
- `policy_latent_matching`: 
    - `checkpoint_path`: Path to the trained checkpoint used for extraction of policy latent embeddings. To train such a checkpoint, use [this training script](../scripts/train_ppo.py) while using appropriate `train_levels` IN [`configs/ppo_training/config.yaml`](../configs/ppo_training/config.yaml). For this experiment, `train_levels` was set to `train_all` there.

- `morphology`: 
    - `feature_weights`: per-feature multipliers applied before distance. Feature ordering: `[polygons, circles, joints, thrusters, green_shapes, blue_shapes]`

- `performance`:
    - `mode`: The way the score is calculated based on frozen evaluation. Possible values:
        - `return_rank`: rank policies per env by avg episode return (works the best here)
        - `solve_rate`: fraction of episodes solved
        - `hybrid`: solve_rate + small return fraction as a tiebreaker. The coef for return is `hybrid_return_weight`
    - `policy_eval_ep_count`: number of frozen evaluation steps is this value x 256, which is default kinetix maximum episode length. 