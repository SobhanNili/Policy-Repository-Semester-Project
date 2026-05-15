import numpy as np

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
