import math
import numpy as np
import matplotlib.pyplot as plt

def append_trajectory_metrics(traj_info, episode_history, num_copies_per_env=None):
    """Accumulate episode metrics from a PPO rollout into per-policy (and optionally per-env) lists.

    Args:
        traj_info: info dict from PPO rollout with keys like ``returned_episode``, ``returned_episode_solved``, etc.
                   Each value has shape ``(num_policies, num_steps, batch_size)``.
        episode_history: mutable dict mapping metric names to ``num_policies`` inner lists.
        num_copies_per_env: if provided, also groups episodes by which *distinct* env they belong to
                            within the flattened batch.  Batch layout is assumed to be
                            ``[env_0 × C, env_1 × C, …, env_{S-1} × C]`` where ``S = batch_size // C``.
                            Results are stored under ``episode_history["per_env"]`` as
                            ``per_env[metric][policy_idx][env_pool_idx] = [float, …]``.

    Returns:
        The updated ``episode_history`` dict (mutated in-place).
    """
    done_mask = np.array(traj_info["returned_episode"])
    returns   = np.array(traj_info["returned_episode_returns"])
    lengths   = np.array(traj_info["returned_episode_lengths"])
    timesteps = np.array(traj_info["timestep"])
    solved    = np.array(traj_info["returned_episode_solved"])

    G, T, B = done_mask.shape  # G=num_policies, T=num_steps, B=batch_size

    # Initialise per-distinct-env lists if grouping is requested
    if num_copies_per_env is not None:
        S = B // num_copies_per_env
        if "per_env" not in episode_history:
            episode_history["per_env"] = {
                k: [[[] for _ in range(S)] for _ in range(G)]
                for k in ("return", "length", "solved")
            }

    for g in range(G):
        # Flatten (T, B) → (T*B,) in row-major order: flat_idx = t * B + b
        g_done  = done_mask[g].reshape(-1)
        g_ret   = returns[g].reshape(-1)
        g_len   = lengths[g].reshape(-1)
        g_tstep = timesteps[g].reshape(-1)
        g_solved= solved[g].reshape(-1)

        idx = np.where(g_done)[0]               # flat indices of completed episodes
        if idx.size == 0:
            continue

        ep_returns = g_ret[idx]
        ep_lengths = g_len[idx]
        ep_end     = g_tstep[idx]
        ep_start   = ep_end - ep_lengths
        ep_solved  = g_solved[idx]

        # Flat (aggregated over all env copies) — used by log_step2wandb
        episode_history["return"][g].extend(ep_returns.tolist())
        episode_history["length"][g].extend(ep_lengths.tolist())
        episode_history["start_step"][g].extend(ep_start.tolist())
        episode_history["end_step"][g].extend(ep_end.tolist())
        episode_history.setdefault("solved", [list() for _ in range(G)])[g].extend(ep_solved.tolist())

        # Per-distinct-env grouping: recover batch index b = flat_idx % B, then env_i = b // C
        if num_copies_per_env is not None:
            batch_indices = idx % B                        # which env copy inside the batch
            env_indices = batch_indices // num_copies_per_env  # which distinct env in the pool

            for ep_i, env_i in enumerate(env_indices):
                episode_history["per_env"]["return"][g][int(env_i)].append(float(ep_returns[ep_i]))
                episode_history["per_env"]["length"][g][int(env_i)].append(float(ep_lengths[ep_i]))
                episode_history["per_env"]["solved"][g][int(env_i)].append(float(ep_solved[ep_i]))

    return episode_history

def plot_env_pixel_grid(env_reset_obs_pixel, cols=4, rotate=True, title=None):
    images = np.asarray(env_reset_obs_pixel)
    num_envs = images.shape[0]
    cols = min(cols, num_envs)
    rows = math.ceil(num_envs / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.6 * cols, 3.8 * rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes):
        if idx >= num_envs:
            ax.axis("off")
            continue

        img = images[idx]
        if rotate:
            img = np.rot90(img, k=1)

        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("#9a9a9a")

        ax.set_xlabel(f"Env {idx}", labelpad=8, fontsize=10)

    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="semibold")

    return fig, axes