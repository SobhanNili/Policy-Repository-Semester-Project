"""Analysis and plotting utilities for policy repository experiments.

This module is the single source of truth for diagnostic plotting logic.
It is used by both ``scripts/run.py`` (during training) and
``scripts/plot_results.py`` (standalone post-hoc analysis).

Generates 5 diagnostic plots:
    1. Policy Performance (per-policy mean solve rate & avg return over steps)
    2. Active Environments per Policy (stackplot)
    3. Assignment Certainty (CV = σ/μ per env, box plot per step)
    4. Greedy Assignment Heatmap (env × step, colored by assigned policy)
    5. Affinity–Solve Rate Spearman Correlation
"""

import matplotlib
matplotlib.use("Agg")  # headless — must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from pathlib import Path
from scipy.stats import spearmanr
import jax


# ═══════════════════════════════════════════════════════════════════════════════
# Data unpacking
# ═══════════════════════════════════════════════════════════════════════════════

def _unpack_step_logs(step_logs: list) -> dict:
    """Convert a list of ``RepoStepLog`` namedtuples into a dictionary of
    numpy arrays plus precomputed derived metrics.

    Returns keys:
        T, P, E_subset, num_envs  — dimensions
        affinities       (T, P, E_subset)
        active_masks     (T, E_subset, P)
        solve_rates      (T, P, E_subset)
        avg_returns      (T, P, E_subset)
        candidate_env_indices  (T, E_subset)
        mean_solve_rate  (T, P)  — mean over active envs only
        mean_avg_return  (T, P)  — mean over active envs only
        n_active_envs    (T, P)  — count of active envs
        cv_per_env       (T, E_subset)  — σ_p / μ_p per candidate env
        spearman_rhos    (T,)    — Spearman ρ per step (NaN if degenerate)
        assignment       (num_envs, T)  — greedy argmax per (env, step), NaN if not sampled
    """
    T = len(step_logs)
    P = step_logs[0].affinity_scores.shape[0]
    E_subset = step_logs[0].affinity_scores.shape[1]

    # ── Raw arrays ──────────────────────────────────────────────────────
    affinities = np.stack(
        [np.array(log.affinity_scores) for log in step_logs]
    )  # (T, P, E_subset)
    active_masks = np.stack(
        [np.array(log.active_mask) for log in step_logs]
    )  # (T, E_subset, P)
    solve_rates = np.stack(
        [np.array(log.solve_rate) for log in step_logs]
    )  # (T, P, E_subset)
    avg_returns = np.stack(
        [np.array(log.avg_return) for log in step_logs]
    )  # (T, P, E_subset)
    candidate_env_indices = np.stack(
        [np.array(log.candidate_env_indices) for log in step_logs]
    )  # (T, E_subset)

    # ── Per-policy active metrics ───────────────────────────────────────
    active_for_metric = active_masks.transpose(0, 2, 1)  # (T, P, E_subset)

    mean_solve_rate = np.full((T, P), np.nan)
    mean_avg_return = np.full((T, P), np.nan)
    n_active_envs = np.zeros((T, P), dtype=int)

    for t in range(T):
        for p in range(P):
            active = active_for_metric[t, p]
            n_active_envs[t, p] = active.sum()
            if active.any():
                mean_solve_rate[t, p] = np.nanmean(
                    np.where(active, solve_rates[t, p], np.nan)
                )
                mean_avg_return[t, p] = np.nanmean(
                    np.where(active, avg_returns[t, p], np.nan)
                )

    # ── CV per env: σ_p(affinity) / μ_p(affinity) ───────────────────────
    mean_p = affinities.mean(axis=1)   # (T, E_subset)
    std_p  = affinities.std(axis=1)    # (T, E_subset)
    cv_per_env = std_p / (mean_p + 1e-8)

    # ── Spearman ρ (affinity vs solve rate, pooled over P×E_subset) ─────
    spearman_rhos = np.full(T, np.nan)
    for t in range(T):
        aff_flat = affinities[t].ravel()
        sr_flat = solve_rates[t].ravel()
        valid = ~np.isnan(sr_flat)
        if (
            valid.sum() > 2
            and np.ptp(aff_flat[valid]) > 0
            and np.ptp(sr_flat[valid]) > 0
        ):
            rho, _ = spearmanr(aff_flat[valid], sr_flat[valid])
            spearman_rhos[t] = rho

    # ── Greedy-assignment matrix: (num_envs, T) ─────────────────────────
    num_envs = int(candidate_env_indices.max()) + 1
    assignment = np.full((num_envs, T), np.nan)
    for t in range(T):
        for local_i in range(E_subset):
            env_idx = int(candidate_env_indices[t, local_i])
            policy_idx = int(np.argmax(affinities[t, :, local_i]))
            assignment[env_idx, t] = policy_idx

    return {
        "T": T,
        "P": P,
        "E_subset": E_subset,
        "num_envs": num_envs,
        "affinities": affinities,
        "active_masks": active_masks,
        "solve_rates": solve_rates,
        "avg_returns": avg_returns,
        "candidate_env_indices": candidate_env_indices,
        "mean_solve_rate": mean_solve_rate,
        "mean_avg_return": mean_avg_return,
        "n_active_envs": n_active_envs,
        "cv_per_env": cv_per_env,
        "spearman_rhos": spearman_rhos,
        "assignment": assignment,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _thin_ticks(ax, steps, max_ticks: int = 15):
    """Limit x-axis tick labels to at most *max_ticks* evenly-spaced positions."""
    T = len(steps)
    if T <= max_ticks:
        return
    step = max(1, T // max_ticks)
    ax.set_xticks(steps[::step])
    ax.set_xticklabels(steps[::step])


def _plot_performance(data, plots_dir, colors):
    """Plot 1: per-policy mean solve rate and avg return over steps."""
    T, P = data["T"], data["P"]
    steps = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for p in range(P):
        ax1.plot(
            steps, data["mean_solve_rate"][:, p],
            color=colors[p], lw=2, label=f"Policy {p}",
            marker=".", markersize=4,
        )
    ax1.set_ylabel("Solve Rate")
    ax1.set_title("Policy Performance on Active Envs Over Training")
    ax1.legend(ncol=P, loc="lower right")
    ax1.set_ylim(-0.05, 1.05)

    for p in range(P):
        ax2.plot(
            steps, data["mean_avg_return"][:, p],
            color=colors[p], lw=2, label=f"Policy {p}",
            marker=".", markersize=4,
        )
    ax2.set_ylabel("Avg Return")
    ax2.set_xlabel("Repo Step")
    ax2.legend(ncol=P, loc="lower right")

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_1_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_active_envs(data, plots_dir, colors):
    """Plot 2: stackplot of active environment counts per policy."""
    T, P = data["T"], data["P"]
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.stackplot(
        steps, data["n_active_envs"].T,
        labels=[f"Policy {p}" for p in range(P)],
        colors=colors, alpha=0.85,
    )
    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Number of Active Envs")
    ax.set_title("Environments Assigned Per Policy Over Training")
    ax.legend(loc="upper right")
    ax.set_xlim(0, T - 1)

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_2_active_envs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_certainty(data, plots_dir):
    """Plot 3: box plot of per-environment CV (σ/μ) at each repo step.

    Each box shows the distribution of CV across the E_subset candidate
    environments for that step.  A rising trend indicates that affinity
    scores are spreading out — policies are developing differentiated
    "opinions" about individual environments.
    """
    cv = data["cv_per_env"]        # (T, E_subset)
    T = data["T"]
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5))

    box_data = [cv[t] for t in range(T)]
    bp = ax.boxplot(
        box_data, positions=steps, widths=0.6,
        patch_artist=True, showfliers=True,
        boxprops={"facecolor": "lightblue", "alpha": 0.6},
        medianprops={"color": "darkblue", "lw": 1.5},
        whiskerprops={"color": "gray"},
        capprops={"color": "gray"},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4, "markeredgewidth": 0},
    )

    # Overlay the mean CV per step
    mean_cv = np.nanmean(cv, axis=1)
    ax.plot(steps, mean_cv, "o-", color="darkred", lw=2, markersize=4,
            label="Mean CV")

    ax.set_xlabel("Repo Step")
    ax.set_ylabel(r"CV ($\sigma$ / $\mu$) per Environment")
    ax.set_title("Assignment Certainty Over Training")
    ax.legend(loc="upper left")

    _thin_ticks(ax, steps)

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_3_certainty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_assignment_heatmap(data, plots_dir, colors):
    """Plot 4: env × step heatmap of greedy argmax assignments.

    Rows = individual training environments (0 … E−1).
    Columns = repo steps.
    Cell color = which policy would be assigned by pure argmax (before
    safety nets).  White cells = env was not in the candidate set that step.

    Solid horizontal bands indicate stable ownership; flickering rows
    indicate contested environments.
    """
    assignment = data["assignment"]   # (num_envs, T) — NaN where not sampled
    num_envs, T = data["num_envs"], data["T"]
    P = data["P"]

    cmap = ListedColormap(colors)
    cmap.set_bad("whitesmoke")

    # Proportional height so individual rows stay legible
    fig_height = np.clip(num_envs * 0.15, 4, 20)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    X = np.arange(T + 1)
    Y = np.arange(num_envs + 1)
    masked = np.ma.masked_invalid(assignment)

    mesh = ax.pcolormesh(
        X, Y, masked, cmap=cmap,
        vmin=-0.5, vmax=P - 0.5,
        edgecolors="none", linewidth=0, shading="flat",
    )

    cbar = plt.colorbar(mesh, ax=ax, ticks=np.arange(P))
    cbar.set_label("Assigned Policy")
    cbar.ax.set_yticklabels([f"Policy {p}" for p in range(P)])

    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Environment Index")
    ax.set_title("Greedy Assignment per Environment Over Training")

    # Tick layout
    _thin_ticks(ax, np.arange(T) + 0.5, max_ticks=15)
    ax.set_yticks(np.arange(num_envs) + 0.5)
    y_fontsize = 8 if num_envs <= 50 else (6 if num_envs <= 100 else 4)
    ax.set_yticklabels(np.arange(num_envs), fontsize=y_fontsize)
    ax.set_xlim(0, T)
    ax.set_ylim(0, num_envs)
    ax.invert_yaxis()  # env 0 at top

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_4_assignment_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_spearman(data, plots_dir):
    """Plot 5: Spearman ρ between affinity scores and solve rates over steps."""
    T = data["T"]
    rho = data["spearman_rhos"]
    steps = np.arange(T)
    valid = ~np.isnan(rho)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        steps[valid], rho[valid],
        color="tab:green", lw=2, marker=".", markersize=4,
    )
    ax.axhline(0, color="gray", ls="--", lw=1, label=r"$\rho = 0$ (no correlation)")
    ax.set_xlabel("Repo Step")
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title("Affinity–Solve Rate Correlation Over Training")
    ax.legend(loc="lower right")

    _thin_ticks(ax, steps)

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_5_spearman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_plots(step_logs: list, output_dir: Path) -> None:
    """Generate all 5 diagnostic plots from step_logs and save to output_dir/plots/.

    Plots produced:
        1. Policy Performance — per-policy mean solve rate & avg return
        2. Active Environments — stackplot of env counts per policy
        3. Assignment Certainty — box plot of CV (σ/μ) per candidate env
        4. Assignment Heatmap — env × step greedy-argmax assignment matrix
        5. Spearman Correlation — affinity vs solve rate rank correlation

    Args:
        step_logs: list of RepoStepLog namedtuples from a completed run.
        output_dir: path to the run's output directory.
    """
    if not step_logs:
        print("Warning: empty step_logs — nothing to plot.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    data = _unpack_step_logs(step_logs)
    colors = sns.color_palette("muted", data["P"])

    _plot_performance(data, plots_dir, colors)
    _plot_active_envs(data, plots_dir, colors)
    _plot_certainty(data, plots_dir)
    _plot_assignment_heatmap(data, plots_dir, colors)
    _plot_spearman(data, plots_dir)

    print(f"Saved 5 plots to {plots_dir}")

# ═══════════════════════════════════════════════════════════════════════════════
# Environment observation visualization
# ═══════════════════════════════════════════════════════════════════════════════

def save_env_observations(env_params, static_env_params,
                           env_reset_states, output_dir: Path,
                           folder: str = "env_train_observations") -> None:
    """Render and save first-observation images for a set of environments.

    Each image is saved as ``env_{index:03d}.png`` in
    ``output_dir / folder /``.

    This is a standalone utility that creates its own pixel renderer — it
    does not interact with the repository logic.  The render+rotate pattern
    matches ``DinoMatchingRepo._encode_pixels``.

    Args:
        env_params: Kinetix environment parameters.
        static_env_params: Kinetix static environment parameters.
        env_reset_states: Pytree of ``LogEnvState`` with leading dimension E.
        output_dir: Path to the run's output directory.
        folder: Subfolder name inside ``output_dir`` for the images.
                 Default ``"env_train_observations"``.
    """
    from kinetix.render.renderer_pixels import make_render_pixels_rl
    renderer = make_render_pixels_rl(env_params, static_env_params)
    pixels = jax.vmap(lambda s: renderer(s.env_state))(env_reset_states).image
    pixels_np = np.array(pixels)
    pixels_np = np.rot90(pixels_np, k=1, axes=(1, 2))

    out_dir = output_dir / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    E = pixels_np.shape[0]
    for i in range(E):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(pixels_np[i])
        ax.axis("off")
        fig.savefig(out_dir / f"env_{i:03d}.png",
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"Saved {E} environment observation images to {out_dir}")
