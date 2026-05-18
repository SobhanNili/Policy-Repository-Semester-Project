import math
import numpy as np
import matplotlib.pyplot as plt

def append_trajectory_metrics(traj_info, episode_history):
    done_mask = np.array(traj_info["returned_episode"])
    returns   = np.array(traj_info["returned_episode_returns"])
    lengths   = np.array(traj_info["returned_episode_lengths"])
    timesteps = np.array(traj_info["timestep"])
    solved    = np.array(traj_info["returned_episode_solved"])

    G, T, B = done_mask.shape
    for g in range(G):
        g_done  = done_mask[g].reshape(-1)
        g_ret   = returns[g].reshape(-1)
        g_len   = lengths[g].reshape(-1)
        g_tstep = timesteps[g].reshape(-1)
        g_solved= solved[g].reshape(-1)

        idx = np.where(g_done)[0]
        if idx.size == 0:
            continue

        ep_returns = g_ret[idx]
        ep_lengths = g_len[idx]
        ep_end     = g_tstep[idx]
        ep_start   = ep_end - ep_lengths
        ep_solved  = g_solved[idx]

        episode_history["return"][g].extend(ep_returns.tolist())
        episode_history["length"][g].extend(ep_lengths.tolist())
        episode_history["start_step"][g].extend(ep_start.tolist())
        episode_history["end_step"][g].extend(ep_end.tolist())
        episode_history.setdefault("solved", [list() for _ in range(G)])[g].extend(ep_solved.tolist())

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