"""Analysis and plotting utilities for policy repository experiments.

This module is the single source of truth for diagnostic plotting logic.
It is used by both ``scripts/run.py`` (during training) and
``scripts/plot_results.py`` (standalone post-hoc analysis).

Generates the following diagnostic plots:
    - Policy Performance (per-policy mean solve rate & avg return over steps)
    - Active Environments per Policy (stackplot, post-mask)
    - Attracted Environments per Policy (stackplot, raw argmax before safety net)
    - Assignment Certainty — CV (σ/μ per env, box plot per step)
    - Assignment Certainty — Normalized Margin (box plot per step)
    - Greedy Assignment Heatmap (env × step, colored by assigned policy)
    - Affinity–Solve Rate Spearman ρ (Pre-Training Frozen)
    - Affinity–Solve Rate Spearman ρ (Pre-Training Finetuned)
    - Policy–Environment Assignment Grids (per-step, with solve & greedy indicators)
"""

import matplotlib
matplotlib.use("Agg")  # headless — must be set before importing pyplot

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from pathlib import Path
from scipy.stats import spearmanr
import json
import pickle
import pandas as pd
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
        solve_rates      (T, P, E_subset)  — post-training, unmasked
        avg_returns      (T, P, E_subset)  — post-training, unmasked
        candidate_env_indices  (T, E_subset)
        mean_solve_rate  (T, P)  — mean over active envs only
        mean_avg_return  (T, P)  — mean over active envs only
        n_active_envs    (T, P)  — count of active envs (post-mask)
        n_greedy_envs    (T, P)  — count of envs attracted by raw argmax
        cv_per_env       (T, E_subset)  — σ_p / μ_p per candidate env
        score_margin_per_env  (T, E_subset)  — normalized top-2 margin per env
        spearman_pre_frozen    (T, E_subset)  — per-env ρ, pre-training frozen
        spearman_pre_finetune  (T, E_subset)  — per-env ρ, pre-training finetuned
        spearman_post_frozen   (T, E_subset)  — per-env ρ, post-training frozen
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
    )  # (T, P, E_subset)  — post-training, unmasked
    avg_returns = np.stack(
        [np.array(log.avg_return) for log in step_logs]
    )  # (T, P, E_subset)  — post-training, unmasked
    pre_frozen_solve_rates = np.stack(
        [np.array(log.pre_frozen_solve_rate) for log in step_logs]
    )  # (T, P, E_subset)
    pre_frozen_avg_returns = np.stack(
        [np.array(log.pre_frozen_avg_return) for log in step_logs]
    )  # (T, P, E_subset)
    pre_finetune_solve_rates = np.stack(
        [np.array(log.pre_finetune_solve_rate) for log in step_logs]
    )  # (T, P, E_subset)
    pre_finetune_avg_returns = np.stack(
        [np.array(log.pre_finetune_avg_return) for log in step_logs]
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

    # ── Raw greedy argmax counts: (T, P) ────────────────────────────────
    # How many candidate envs would choose each policy based purely on the
    # highest affinity score — before any exploration (softmax / ε-greedy)
    # or safety-net corrections.  When multiple policies share the maximum
    # score for an env, ALL are counted as greedy.
    n_greedy_envs = np.zeros((T, P), dtype=int)
    for t in range(T):
        max_per_col = np.max(affinities[t], axis=0)           # (E_subset,)
        is_max = np.isclose(affinities[t], max_per_col)       # (P, E_subset)
        n_greedy_envs[t] = np.sum(is_max, axis=1)             # (P,)

    # ── CV per env: σ_p(affinity) / μ_p(affinity) ───────────────────────
    mean_p = affinities.mean(axis=1)   # (T, E_subset)
    std_p  = affinities.std(axis=1)    # (T, E_subset)
    cv_per_env = std_p / (mean_p + 1e-8)

    # ── Normalized margin per env: (s_0 - s_1) / (s_0 - s_{P-1}) ───────
    # Measures how decisively the best policy dominates each environment.
    #  0 ≈ all policies equally scored (contested env)
    #  1 ≈ one policy clearly stands out (confident assignment).
    sorted_aff = np.sort(affinities, axis=1)  # (T, P, E_subset), ascending
    score_range = sorted_aff[:, -1, :] - sorted_aff[:, 0, :]  # (T, E_subset)
    top2_gap = sorted_aff[:, -1, :] - sorted_aff[:, -2, :]     # (T, E_subset)
    score_margin_per_env = np.divide(
        top2_gap, score_range,
        out=np.zeros_like(top2_gap),
        where=(score_range > 1e-12),
    )  # (T, E_subset)

    # ── Per-environment Spearman ρ (affinity vs solve rate per column) ───
    spearman_pre_frozen = _compute_per_env_spearman(
        affinities, pre_frozen_solve_rates, T, P, E_subset
    )  # (T, E_subset)
    spearman_pre_finetune = _compute_per_env_spearman(
        affinities, pre_finetune_solve_rates, T, P, E_subset
    )  # (T, E_subset)
    spearman_post_frozen = _compute_per_env_spearman(
        affinities, solve_rates, T, P, E_subset
    )  # (T, E_subset)

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
        "n_greedy_envs": n_greedy_envs,
        "cv_per_env": cv_per_env,
        "score_margin_per_env": score_margin_per_env,
        "spearman_pre_frozen": spearman_pre_frozen,
        "spearman_pre_finetune": spearman_pre_finetune,
        "spearman_post_frozen": spearman_post_frozen,
        "assignment": assignment,
    }


def _compute_per_env_spearman(affinities, solve_rates, T, P, E_subset):
    """Compute per-environment Spearman ρ between affinity and solve rate.

    For each step *t* and candidate environment *e*, computes
    ρ = Spearman(affinity[:, e], solve_rate[:, e]) across the P policies.
    This avoids cross-environment scale artifacts that plague pooled ρ.

    Args:
        affinities: (T, P, E_subset) array.
        solve_rates: (T, P, E_subset) array (may contain NaN).
        T, P, E_subset: dimensions.

    Returns:
        (T, E_subset) array; entry (t, e) is NaN when fewer than 2 valid
        pairs or zero variance in either variable.
    """
    rho = np.full((T, E_subset), np.nan)
    for t in range(T):
        for e in range(E_subset):
            aff_col = affinities[t, :, e]
            sr_col = solve_rates[t, :, e]
            valid = ~np.isnan(sr_col)
            if (
                valid.sum() >= 2
                and np.ptp(aff_col[valid]) > 0
                and np.ptp(sr_col[valid]) > 0
            ):
                rho[t, e], _ = spearmanr(aff_col[valid], sr_col[valid])
    return rho


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
    """Per-policy mean solve rate and avg return over steps."""
    T, P = data["T"], data["P"]
    steps = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for p in range(P):
        ax1.plot(
            steps, data["mean_solve_rate"][:, p],
            color=colors[p], lw=2, label=f"Policy {p}",
            marker=".", markersize=4,
        )
    ax1.set_ylabel("Episode Solve Rate")
    ax1.set_title("Policy Performance on Assigned Environments")
    ax1.legend(ncol=P, loc="lower right")
    ax1.set_ylim(-0.05, 1.05)

    for p in range(P):
        ax2.plot(
            steps, data["mean_avg_return"][:, p],
            color=colors[p], lw=2, label=f"Policy {p}",
            marker=".", markersize=4,
        )
    ax2.set_ylabel("Avg. Episode Return")
    ax2.set_xlabel("Repository Training Step")
    ax2.legend(ncol=P, loc="lower right")

    plt.tight_layout()
    fig.savefig(plots_dir / "performance_over_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_active_envs(data, plots_dir, colors):
    """Stackplot of active environment counts per policy."""
    T, P = data["T"], data["P"]
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.stackplot(
        steps, data["n_active_envs"].T,
        labels=[f"Policy {p}" for p in range(P)],
        colors=colors, alpha=0.85,
    )
    ax.set_xlabel("Repository Training Step")
    ax.set_ylabel("# Assigned Environments")
    ax.set_title("Number of Assigned Environments to Each Policy")
    ax.legend(loc="upper right")
    ax.set_xlim(0, T - 1)

    plt.tight_layout()
    fig.savefig(plots_dir / "assigned_envs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_env_greedy_policy_choice(data, plots_dir, colors):
    """Stackplot of environment counts per policy by raw argmax (no exploration, no safety net).

    Counts how many candidate environments would choose each policy based
    purely on the highest affinity score — before any assignment-mode
    stochasticity (softmax / ε-greedy) or safety-net corrections.

    Contrast with ``_plot_active_envs``: if this plot shows one policy
    hoarding all envs while the active-envs plot is balanced, the safety
    net is doing heavy lifting.
    """
    T, P = data["T"], data["P"]
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.stackplot(
        steps, data["n_greedy_envs"].T,
        labels=[f"Policy {p}" for p in range(P)],
        colors=colors, alpha=0.85,
    )
    ax.set_xlabel("Repository Training Step")
    ax.set_ylabel("# Attracted Environments")
    ax.set_title("Number of Attracted Environments per Policy (Greedy Affinity Choice)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, T - 1)

    # Reference line for candidate pool size
    ax.axhline(data["E_subset"], color="gray", ls="--", lw=1, alpha=0.5,
               label=f"Candidate pool ({data['E_subset']})")

    plt.tight_layout()
    fig.savefig(plots_dir / "env_greedy_policy_choices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_affinity_score_spread_cv(data, plots_dir):
    """Box plot of per-environment CV (σ/μ) at each repo step.

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

    ax.set_xlabel("Repository Training Step")
    ax.set_ylabel(r"CV ($\sigma$ / $\mu$) per Environment")
    ax.set_title("Affinity Score Spread — Coefficient of Variation Over Training")
    ax.legend(loc="upper left")

    _thin_ticks(ax, steps)

    plt.tight_layout()
    fig.savefig(plots_dir / "score_spread_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_affinity_score_spread_margin(data, plots_dir):
    """Box plot of per-environment Normalized Margin at each repo step.

    Each box shows the distribution of the normalized top-2 margin across the
    E_subset candidate environments for that step.

    - 0   = all policies score the environment equally (contested).
    - 1   = one policy clearly dominates (confident assignment).

    A rising trend indicates that policies are developing more decisive
    preferences — one policy stands out per environment.

    The Normalized Margin is scale-invariant, unlike CV, making it comparable
    across different affinity-scoring methods (RBF, rank-based, etc.).
    """
    margin = data["score_margin_per_env"]  # (T, E_subset)
    T = data["T"]
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5))

    box_data = [margin[t] for t in range(T)]
    ax.boxplot(
        box_data, positions=steps, widths=0.6,
        patch_artist=True, showfliers=True,
        boxprops={"facecolor": "lightgreen", "alpha": 0.6},
        medianprops={"color": "darkgreen", "lw": 1.5},
        whiskerprops={"color": "gray"},
        capprops={"color": "gray"},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4,
                    "markeredgewidth": 0},
    )

    # Overlay the mean margin per step
    mean_margin = np.nanmean(margin, axis=1)
    ax.plot(steps, mean_margin, "o-", color="darkred", lw=2, markersize=4,
            label="Mean Normalized Margin")

    ax.set_xlabel("Repository Training Step")
    ax.set_ylabel("Normalized Margin per Environment")
    ax.set_title(r"Affinity Score Spread — Normalized Top-2 Margin")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")

    _thin_ticks(ax, steps)

    plt.tight_layout()
    fig.savefig(plots_dir / "score_spread_margin.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_assignment_heatmap(data, plots_dir, colors):
    """Env × step heatmap of greedy argmax assignments.

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

    ax.set_xlabel("Repository Training Step")
    ax.set_ylabel("Environment Index")
    ax.set_title("Greedy Policy Assignment per Environment Over Training")

    # Tick layout
    _thin_ticks(ax, np.arange(T) + 0.5, max_ticks=15)
    ax.set_yticks(np.arange(num_envs) + 0.5)
    y_fontsize = 8 if num_envs <= 50 else (6 if num_envs <= 100 else 4)
    ax.set_yticklabels(np.arange(num_envs), fontsize=y_fontsize)
    ax.set_xlim(0, T)
    ax.set_ylim(0, num_envs)
    ax.invert_yaxis()  # env 0 at top

    plt.tight_layout()
    fig.savefig(plots_dir / "assignment_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_per_env_spearman(data, plots_dir, rho_key, title, filename):
    """Box plot of per-environment Spearman ρ at each repo step.

    Each box shows the distribution of ρ across the E_subset candidate
    environments for that step.  A rising trend indicates the affinity
    metric is becoming more predictive of performance *within* individual
    environments.

    Args:
        data: dict from ``_unpack_step_logs``.
        plots_dir: Path to save the plot.
        rho_key: key in ``data`` for the (T, E_subset) ρ array.
        title: plot title string.
        filename: output filename (e.g. ``"spearman_pre_frozen.png"``).
    """
    rho_per_env = data[rho_key]  # (T, E_subset)
    T = data["T"]
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Build box data: one list per step, dropping NaN entries
    box_data = []
    mean_rho = np.full(T, np.nan)
    for t in range(T):
        row = rho_per_env[t]
        valid = row[~np.isnan(row)]
        box_data.append(valid if len(valid) > 0 else [])
        if len(valid) > 0:
            mean_rho[t] = np.mean(valid)

    ax.boxplot(
        box_data, positions=steps, widths=0.6,
        patch_artist=True, showfliers=True,
        boxprops={"facecolor": "lightblue", "alpha": 0.6},
        medianprops={"color": "darkblue", "lw": 1.5},
        whiskerprops={"color": "gray"},
        capprops={"color": "gray"},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4,
                    "markeredgewidth": 0},
    )

    # Overlay the mean ρ per step (NaN where no valid data exists)
    ax.plot(steps, mean_rho, "o-", color="darkred", lw=2, markersize=4,
            label=r"Mean $\rho$")

    ax.axhline(0, color="gray", ls="--", lw=1, label=r"$\rho = 0$")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Repo Step")
    ax.set_ylabel(r"Spearman $\rho$ per Environment")
    ax.set_title(title)
    ax.legend(loc="lower right")

    _thin_ticks(ax, steps)

    plt.tight_layout()
    fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-step policy–environment assignment grids
# ═══════════════════════════════════════════════════════════════════════════════

def generate_env_assignment_grids(
    step_logs: list,
    output_dir: Path,
    obs_dir: str = "env_train_observations",
    step_interval: int = 1,
    max_envs_per_policy: int | None = None,
    solve_threshold: float = 0.0,
    cell_size_inches: float = 1.4,
) -> None:
    """Generate per-step grid images showing which environments each policy
    was assigned to, what those environments look like, and whether the
    policy solved them.

    Grid layout
    -----------
    * Rows    = policies (fixed).
    * Columns = env slots (fixed width — policies with fewer envs have
      grey trailing cells).
    * Each cell contains the environment observation image, framed by a
      coloured border.

    Border encoding
    ---------------
    ======== ======== ====================================================
    Colour   Style    Meaning
    ======== ======== ====================================================
    Green    Solid    Greedy choice (highest affinity) — solved
    Red      Solid    Greedy choice — did **not** solve
    Green    Dashed   Safety-net / exploration — solved anyway
    Red      Dashed   Safety-net / exploration — did **not** solve
    ======== ======== ====================================================

    Images saved to ``output_dir/plots/assignment_grids/step_XXXX.png``.
    """
    from matplotlib.lines import Line2D

    data = _unpack_step_logs(step_logs)
    T, P, E_subset = data["T"], data["P"], data["E_subset"]
    obs_path = output_dir / obs_dir

    if not obs_path.is_dir():
        print(f"Warning: observation directory not found — skipping assignment grids: {obs_path}")
        return

    if max_envs_per_policy is None:
        max_envs_per_policy = min(E_subset, 10)
    n_cols = max_envs_per_policy
    if n_cols == 0:
        return

    grids_dir = output_dir / "plots" / "assignment_grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-load env images ────────────────────────────────────────────
    num_envs = data["num_envs"]
    env_images: dict[int, np.ndarray] = {}
    for i in range(num_envs):
        p = obs_path / f"env_{i:03d}.png"
        if p.exists():
            env_images[i] = plt.imread(str(p))

    GREEN = "#5b9a5e"
    RED = "#d35e5e"
    GREY = "#e8e8e8"

    steps_to_plot = list(range(0, T, step_interval))

    for t in steps_to_plot:
        # All policies tied for the maximum score count as greedy (solid border).
        aff_t = data["affinities"][t]                              # (P, E_subset)
        max_per_col = np.max(aff_t, axis=0)                        # (E_subset,)
        is_greedy = np.isclose(aff_t, max_per_col)                 # (P, E_subset) bool

        # ── Collect assigned envs per policy ─────────────────────────
        policy_envs: list[list[tuple[int, float, bool]]] = []
        for p in range(P):
            active = data["active_masks"][t, :, p].astype(bool)
            idx = np.where(active)[0]
            gids = data["candidate_env_indices"][t][idx]
            srs = data["solve_rates"][t, p, idx]
            greedy = is_greedy[p, idx]
            info = [(int(g), float(s), bool(ig)) for g, s, ig in zip(gids, srs, greedy)]
            info.sort(key=lambda x: x[0])
            policy_envs.append(info)

        # ── Figure ──────────────────────────────────────────────────
        fig, axes = plt.subplots(
            P, n_cols,
            figsize=(n_cols * cell_size_inches, P * cell_size_inches + 0.55),
            squeeze=False,
            facecolor="white",
        )
        fig.subplots_adjust(
            left=0.09, right=0.98, top=0.92, bottom=0.10,
            wspace=0.10, hspace=0.14,
        )

        fig.text(0.5, 0.965, f"Step {t}",
                 ha="center", va="top", fontsize=12, fontweight="bold")

        for p in range(P):
            envs = policy_envs[p]
            n_envs = len(envs)
            n_greedy = sum(1 for _, _, ig in envs if ig)
            truncated = n_envs > n_cols
            shown = envs[:n_cols]

            for col in range(n_cols):
                ax = axes[p, col]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False)

                if col < len(shown):
                    gid, sr, is_greedy = shown[col]
                    solved = (not np.isnan(sr)) and (sr > solve_threshold)

                    if gid in env_images:
                        ax.imshow(env_images[gid])
                    else:
                        ax.set_facecolor("#ddd")

                    # Border
                    c = GREEN if solved else RED
                    lw = 3.0 if is_greedy else 1.6
                    ls = "-" if is_greedy else "--"
                    for spine in ax.spines.values():
                        spine.set_color(c)
                        spine.set_linewidth(lw)
                        spine.set_linestyle(ls)
                        spine.set_visible(True)

                    # Label
                    sr_str = f"{sr:.2f}" if not np.isnan(sr) else "—"
                    ax.text(0.5, 0.015, f"E{gid}  {sr_str}",
                            ha="center", va="bottom", fontsize=6,
                            fontweight="bold" if is_greedy else "normal",
                            color=c, transform=ax.transAxes)
                else:
                    ax.set_facecolor(GREY)

            # Row label
            parts = [f"Policy {p}", f"{n_envs} envs"]
            if n_greedy:
                parts.append(f"{n_greedy} greedy")
            if truncated:
                parts.append(f"+{n_envs - n_cols} more")
            axes[p, 0].set_ylabel("\n".join(parts),
                                   fontsize=7.5, fontweight="bold",
                                   rotation=0, ha="right", va="center",
                                   labelpad=38)

        # ── Legend ──────────────────────────────────────────────────
        legend_handles = [
            Line2D([0], [0], color=GREEN, lw=2.5, label="Greedy, solved"),
            Line2D([0], [0], color=RED,   lw=2.5, label="Greedy, fail"),
            Line2D([0], [0], color=GREEN, lw=1.4, ls="--", label="Safety-net, solved"),
            Line2D([0], [0], color=RED,   lw=1.4, ls="--", label="Safety-net, fail"),
        ]
        fig.legend(
            handles=legend_handles, loc="lower center",
            ncol=4, fontsize=7, frameon=False,
            handlelength=1.2, handletextpad=0.5,
            columnspacing=1.0,
        )

        fig.savefig(grids_dir / f"step_{t:04d}.png",
                    dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)

    print(f"Saved {len(steps_to_plot)} assignment grid images to {grids_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation assignment bar plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eval_assignment_bars(
    env2policy: np.ndarray,
    num_policies: int,
    plots_dir: Path,
) -> None:
    """Bar plot: absolute number of evaluation environments assigned to
    each policy.

    Assignment is determined by greedy argmax over affinity scores, so it
    is identical for frozen and finetuned evaluation.

    Args:
        env2policy: (E_eval,) integer array — which policy each eval
            environment was assigned to.
        num_policies: total number of policies *P*.
        plots_dir: directory to save ``eval_assignment_bars.png``.
    """
    env2policy = np.asarray(env2policy, dtype=int)
    counts = np.bincount(env2policy, minlength=num_policies)

    P = num_policies
    x = np.arange(P)

    fig, ax = plt.subplots(figsize=(max(6, P * 0.9), 4.8))
    fig.patch.set_facecolor("white")

    colors = sns.color_palette("pastel", P)

    bars = ax.bar(
        x, counts,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Policy {p}" for p in range(P)], fontsize=10)
    ax.set_ylabel("# of Environments Assigned", fontsize=11)
    ax.set_title(
        "Evaluation Environment Assignment Distribution",
        fontsize=13, fontweight="bold",
    )

    y_max = counts.max()
    ax.set_ylim(0, y_max + max(2, y_max * 0.15))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        plots_dir / "eval_env_assignment_dist.png",
        dpi=150, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def _plot_eval_assignment_from_run(run_dir: Path) -> None:
    """Load eval assignment from saved pickles and generate the bar plot."""
    import json

    frozen_path = run_dir / "eval_frozen_results.pkl"
    config_path = run_dir / "config.json"
    if not frozen_path.exists() or not config_path.exists():
        return

    with open(frozen_path, "rb") as f:
        _, env2policy, _ = pickle.load(f)

    with open(config_path) as f:
        config = json.load(f)
    num_policies = int(config.get("num_policies", 0))
    if num_policies == 0:
        return

    plot_eval_assignment_bars(
        np.array(env2policy),
        num_policies,
        run_dir / "plots",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_plots(step_logs: list, output_dir: Path) -> None:
    """Generate all diagnostic plots from step_logs and save to output_dir/plots/.

    Plots produced:
        - Policy Performance — per-policy mean solve rate & avg return
        - Active Environments — stackplot of env counts per policy (post-mask)
        - Attracted Environments — stackplot of raw argmax env counts per policy
        - Assignment Certainty (CV) — box plot of σ/μ per candidate env
        - Assignment Certainty (Norm. Margin) — box plot of normalized
          top-2 gap per candidate env
        - Assignment Heatmap — env × step greedy-argmax assignment matrix
        - Spearman ρ (Pre-Training Frozen) — per-env box plot
        - Spearman ρ (Pre-Training Finetuned) — per-env box plot
        - Policy–Environment Assignment Grids — per-step thumbnails with
          solve (green/red) and greedy (solid/dashed) indicators

    Each Spearman plot shows the distribution (across candidate environments)
    of within-column rank correlation between affinity scores and solve rates,
    avoiding cross-environment scale artifacts that plague pooled correlation.

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
    _plot_env_greedy_policy_choice(data, plots_dir, colors)
    _plot_affinity_score_spread_cv(data, plots_dir)
    _plot_affinity_score_spread_margin(data, plots_dir)
    _plot_assignment_heatmap(data, plots_dir, colors)
    _plot_per_env_spearman(
        data, plots_dir,
        rho_key="spearman_pre_frozen",
        title="Affinity Score - Solve Rate Spearman Correlation (Frozen Evaluation)",
        filename="affinity_sr_spearman_frozen.png",
    )
    _plot_per_env_spearman(
        data, plots_dir,
        rho_key="spearman_pre_finetune",
        title="Affinity Score - Solve Rate Spearman Correlation (Finetuned Evaluation)",
        filename="affinity_sr_spearman_finetune.png",
    )
    # _plot_per_env_spearman(
    #     data, plots_dir,
    #     rho_key="spearman_post_frozen",
    #     title="Affinity–Solve Rate Spearman ρ (Post-Training Frozen)",
    #     filename="spearman_post_frozen.png",
    # )

    # ── Per-step assignment grid images ─────────────────────────────────
    # Use a sensible default interval: roughly 20 grids across training
    grid_interval = max(1, data["T"] // 20)
    generate_env_assignment_grids(
        step_logs, output_dir,
        step_interval=grid_interval,
    )

    # ── Eval assignment bar plot ─────────────────────────────────────────
    _plot_eval_assignment_from_run(output_dir)

    print(f"Saved all plots to {plots_dir}")

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


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-experiment comparison
# ═══════════════════════════════════════════════════════════════════════════════

REPO_COLORS: dict[str, str] = {
    "random_matching":          "#999999",
    "morphology_matching":      "#E69F00",
    "dino_matching":            "#56B4E9",
    "policy_latent_matching":   "#009E73",
    "performance_matching":     "#CC79A7",
}

REPO_DISPLAY_NAMES: dict[str, str] = {
    "random_matching":          "Random",
    "morphology_matching":      "Morphology",
    "dino_matching":            "DINO",
    "policy_latent_matching":   "Policy Latent",
    "performance_matching":     "Performance",
}


def _parse_repo_type_from_name(dirname: str) -> str | None:
    """Extract repo_type from directory name like 'dino_matching_2026-06-07_22-56-21'.

    The last two underscore-delimited segments are date and time; everything
    before them is the repo_type (which may itself contain underscores, e.g.
    ``policy_latent_matching``).
    """
    parts = dirname.split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[:-2])


def _get_color_for_repo(repo_type: str) -> str:
    """Return the hex color for a repo_type, with a dark fallback."""
    return REPO_COLORS.get(repo_type, "#555555")


def _get_display_name(repo_type: str) -> str:
    """Return a human-readable short name for a repo_type."""
    return REPO_DISPLAY_NAMES.get(repo_type, repo_type)


# ── Dynamic hyperparam discovery ─────────────────────────────────────────────

_IDENTITY_COLS = {"run_dir", "repo_type", "repo_display"}
_METRIC_PREFIXES = ("frozen_", "finetune_")


def _get_varying_hyperparam_cols(df: pd.DataFrame) -> list[str]:
    """Return columns that represent varying hyperparameters.

    Excludes identity columns (run_dir, repo_type, repo_display) and
    metric columns (frozen_*, finetune_*).  Returns only columns with
    more than one unique non-NaN value — constant columns provide no
    grouping or labelling signal.

    This is the single source of truth for hyperparam discovery;
    nowhere else in the module should a hyperparam name be hardcoded.
    """
    varying: list[str] = []
    for col in df.columns:
        if col in _IDENTITY_COLS:
            continue
        if col.startswith(_METRIC_PREFIXES):
            continue
        series = df[col].dropna()
        if series.nunique() > 1:
            varying.append(col)
    return varying


def _select_best_per_repo(
    df: pd.DataFrame,
    metric: str = "finetune_solve_rate_mean",
) -> pd.DataFrame:
    """Select the best hyperparameter configuration per repo type.

    For each ``repo_type``, picks the row with the highest value of
    *metric* (default: finetuned solve rate).  If *metric* is missing
    for a repo, falls back to ``frozen_solve_rate_mean``.

    Returns a DataFrame with one row per repo type.
    """
    best_rows: list[dict] = []
    for repo_type, group in df.groupby("repo_type", dropna=False):
        if metric in group.columns and group[metric].notna().any():
            idx = group[metric].idxmax()
        elif "frozen_solve_rate_mean" in group.columns:
            idx = group["frozen_solve_rate_mean"].idxmax()
        else:
            idx = group.index[0]
        best_rows.append(group.loc[idx].to_dict())
    return pd.DataFrame(best_rows)


# ── Data collection ──────────────────────────────────────────────────────────

def collect_experiment_results(
    outputs_root: Path | None = None,
    run_dirs: list[Path] | None = None,
) -> pd.DataFrame:
    """Scan experiment folders and return a DataFrame with one row per run.

    Each row contains aggregate metrics from ``eval_frozen.csv`` /
    ``eval_finetune.csv`` plus hyperparameters from ``config.json`` and
    ``repo_kwargs.json`` for grouping and labeling.

    Two input modes:
      - **Scan mode**: provide ``outputs_root`` — finds all subdirectories
        that contain a ``config.json``.
      - **Explicit mode**: provide ``run_dirs`` — uses only those directories.

    Args:
        outputs_root: parent directory to scan for experiment subdirectories.
        run_dirs: explicit list of run directory paths (takes precedence).

    Returns:
        DataFrame with columns: run_dir, repo_type, repo_display,
        frozen_solve_rate_mean, frozen_solve_rate_std,
        frozen_avg_return_mean, frozen_avg_return_std,
        finetune_solve_rate_mean, finetune_solve_rate_std,
        finetune_avg_return_mean, finetune_avg_return_std,
        plus hyperparameter columns (seed, sigma, history_decay, mode, …).
    """
    if run_dirs is None and outputs_root is not None:
        run_dirs = sorted(
            d for d in outputs_root.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )
    if run_dirs is None:
        return pd.DataFrame()

    rows: list[dict] = []
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        kwargs_path = run_dir / "repo_kwargs.json"
        frozen_csv = run_dir / "eval_frozen.csv"
        finetune_csv = run_dir / "eval_finetune.csv"

        if not config_path.exists() or not frozen_csv.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)

        # ── repo_type: prefer config.json, fall back to directory name ──
        repo_type = config.get("repo_type")
        if repo_type is None:
            repo_type = _parse_repo_type_from_name(run_dir.name)
        if repo_type is None:
            continue

        kwargs = {}
        if kwargs_path.exists():
            with open(kwargs_path) as f:
                kwargs = json.load(f)

        # ── Aggregate metrics from CSV (first two rows are mean / std) ──
        frozen_df = pd.read_csv(frozen_csv, index_col=0)

        row: dict = {
            "run_dir": str(run_dir),
            "repo_type": repo_type,
            "frozen_solve_rate_mean": frozen_df.loc["mean", "solve_rate"],
            "frozen_solve_rate_std":  frozen_df.loc["std", "solve_rate"],
            "frozen_avg_return_mean": frozen_df.loc["mean", "avg_return"],
            "frozen_avg_return_std":  frozen_df.loc["std", "avg_return"],
        }

        if finetune_csv.exists():
            ft_df = pd.read_csv(finetune_csv, index_col=0)
            if "mean" in ft_df.index:
                row.update({
                    "finetune_solve_rate_mean": ft_df.loc["mean", "solve_rate"],
                    "finetune_solve_rate_std":  ft_df.loc["std", "solve_rate"],
                    "finetune_avg_return_mean": ft_df.loc["mean", "avg_return"],
                    "finetune_avg_return_std":  ft_df.loc["std", "avg_return"],
                })

        # ── Flatten hyperparams: take ALL scalar values from both files ──
        for key, val in kwargs.items():
            if not isinstance(val, (list, dict)):
                row[key] = val
        for key, val in config.items():
            if key not in row and not isinstance(val, (list, dict)):
                row[key] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    if "repo_type" in df.columns:
        df["repo_display"] = df["repo_type"].apply(_get_display_name)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison plot functions
# ═══════════════════════════════════════════════════════════════════════════════

def _meta_group_keys(df: pd.DataFrame) -> list[str]:
    """Return grouping columns: repo_type, display name, plus any
    hyperparameter that varies (discovered dynamically)."""
    keys = ["repo_type", "repo_display"]
    keys.extend(_get_varying_hyperparam_cols(df))
    return keys


def _build_run_label(row, varying_cols: list[str] | None = None,
                     max_dir_chars: int = 18) -> str:
    """Compact label for a single run: repo display name + tail of dir name.

    If *varying_cols* is provided, only those columns are used for the
    parameter suffix.  Otherwise all non-identity, non-metric columns
    with a non-NaN value in *row* are used.
    """
    dname = Path(row["run_dir"]).name
    tail = dname[-max_dir_chars:] if len(dname) > max_dir_chars else dname
    display = row.get("repo_display", row["repo_type"])
    if varying_cols is None:
        varying_cols = [
            k for k in row.index
            if k not in _IDENTITY_COLS
            and not k.startswith(_METRIC_PREFIXES)
            and pd.notna(row[k])
        ]
    param_parts = []
    for k in varying_cols:
        if k in row and pd.notna(row[k]):
            param_parts.append(f"{k}={row[k]}")
    if param_parts:
        return f"{display} ({', '.join(param_parts)})\n{tail}"
    return f"{display}\n{tail}"


# ── Plot 1: Grouped bar chart ────────────────────────────────────────────────

def plot_comparison_bars(
    df: pd.DataFrame,
    plots_dir: Path,
    title: str | None = None,
    filename: str = "comparison_bars.png",
) -> None:
    """Grouped bar chart: mean ± std by (repo_type, hyperparams).

    Produces a single figure with two subplots: solve rate (top) and
    avg return (bottom).  Frozen and finetuned results are shown
    side-by-side.  If multiple runs share the same config (different
    seeds), they are averaged and std is computed across seeds.

    Args:
        df: DataFrame from ``collect_experiment_results``.
        plots_dir: directory to save the plot.
        title: optional figure title (default: auto-generated).
        filename: output filename (default: ``"comparison_bars.png"``).
    """
    group_keys = _meta_group_keys(df)
    varying_cols = _get_varying_hyperparam_cols(df)

    def _build_agg(metric: str) -> pd.DataFrame:
        agg = df.groupby(group_keys, dropna=False).agg(
            frozen_mean=(f"frozen_{metric}_mean", "mean"),
            frozen_std=(f"frozen_{metric}_mean", "std"),
            finetune_mean=(f"finetune_{metric}_mean", "mean"),
            finetune_std=(f"finetune_{metric}_mean", "std"),
            n=(f"frozen_{metric}_mean", "count"),
        ).reset_index()
        agg["frozen_std"] = agg["frozen_std"].fillna(0)
        agg["finetune_std"] = agg["finetune_std"].fillna(0)
        return agg

    agg_sr = _build_agg("solve_rate")
    agg_ar = _build_agg("avg_return")

    # Sort both by frozen solve rate (descending) for consistency
    order = agg_sr.sort_values("frozen_mean", ascending=False).index
    agg_sr = agg_sr.loc[order].reset_index(drop=True)
    agg_ar = agg_ar.loc[order].reset_index(drop=True)

    N = len(agg_sr)
    if N == 0:
        return

    # Build labels from dynamic varying columns
    labels = []
    for _, row in agg_sr.iterrows():
        label = row["repo_display"]
        extras = []
        for k in varying_cols:
            if k in row and pd.notna(row[k]):
                extras.append(f"{k}={row[k]}")
        if extras:
            label += f"  [{', '.join(extras)}]"
        labels.append(label)

    x = np.arange(N)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(8, N * 0.9), 8), sharex=True,
    )

    colors = [_get_color_for_repo(t) for t in agg_sr["repo_type"]]

    def _draw_bars(ax, agg, metric_label: str):
        ax.bar(
            x - width / 2, agg["frozen_mean"], width,
            yerr=agg["frozen_std"], capsize=4,
            color=colors, edgecolor="white", linewidth=0.8,
        )
        ax.bar(
            x + width / 2, agg["finetune_mean"], width,
            yerr=agg["finetune_std"], capsize=4,
            color=[_desaturate(c, 0.5) for c in colors],
            edgecolor="white", linewidth=0.8,
            hatch="//",
        )
        # Annotate seed count
        for i, (m, s, n) in enumerate(zip(agg["frozen_mean"],
                                           agg["frozen_std"],
                                           agg["n"])):
            if n > 1:
                ax.annotate(f"n={n}", (i - width / 2, m + s + 0.015),
                            ha="center", fontsize=7, color="gray")
        ax.set_ylabel(metric_label)
        all_means = np.concatenate([agg["frozen_mean"].values,
                                     agg["finetune_mean"].values])
        all_stds = np.concatenate([agg["frozen_std"].values,
                                    agg["finetune_std"].values])
        ymax = (all_means + all_stds).max() + 0.05
        ax.set_ylim(0, max(1.05, ymax))

    _draw_bars(ax1, agg_sr, "Solve Rate")
    _draw_bars(ax2, agg_ar, "Avg Return")

    # ── Shared legend with pattern-based handles (not colour-based) ───
    legend_handles = [
        mpatches.Patch(facecolor="#888888", edgecolor="white",
                       linewidth=0.8, label="Frozen"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="white",
                       linewidth=0.8, hatch="//", label="Finetuned"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=2, fontsize=9, frameon=False,
        handlelength=1.4, handleheight=1.2,
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)

    if title is None:
        title = "Policy Repo Comparison"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 1b: Per-repo-type bar charts ────────────────────────────────────────

def plot_per_repo_comparison_bars(
    df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """For each repo type, produce a dual-subplot bar chart comparing
    its own hyperparameter configurations (frozen vs finetuned).

    One ``comparison_{repo_type}.png`` file is saved per repo type.
    Repo types with only a single configuration still get a plot for
    consistency.
    """
    for repo_type in sorted(df["repo_type"].unique()):
        sub_df = df[df["repo_type"] == repo_type]
        if sub_df.empty:
            continue
        display = _get_display_name(repo_type)
        safe_name = repo_type  # already filesystem-safe
        plot_comparison_bars(
            sub_df,
            plots_dir,
            title=f"{display} — Hyperparameter Comparison",
            filename=f"comparison_{safe_name}.png",
        )


def _desaturate(hex_color: str, amount: float = 0.4) -> str:
    """Lighten/desaturate a hex color by blending with white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Plot 2: Frozen vs Finetuned scatter ──────────────────────────────────────

def plot_frozen_vs_finetune_scatter(
    df: pd.DataFrame,
    plots_dir: Path,
    metric: str = "solve_rate",
) -> None:
    """Scatter of frozen vs finetuned metric, one point per run.

    Points above the diagonal indicate that per-env fine-tuning improves
    the metric.  Points near the diagonal mean the frozen policy already
    generalises well to its assigned environments.

    Color-coded by repo type.  Marker size reflects the frozen std (larger =
    more variable per-env performance).
    """
    frozen_col = f"frozen_{metric}_mean"
    ft_col = f"finetune_{metric}_mean"
    df_plot = df.dropna(subset=[frozen_col, ft_col])
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    repo_types = sorted(df_plot["repo_type"].unique())

    for rt in repo_types:
        sub = df_plot[df_plot["repo_type"] == rt]
        color = _get_color_for_repo(rt)
        label = _get_display_name(rt)
        sizes = np.clip(sub.get(f"frozen_{metric}_std", 0.2).fillna(0.2) * 300, 30, 200)
        ax.scatter(
            sub[frozen_col], sub[ft_col],
            c=color, label=label, s=sizes,
            edgecolors="white", linewidth=0.5, zorder=3, alpha=0.85,
        )

    # Diagonal
    lims = [-0.02, 1.05]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.35, label="No change")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    metric_label = "Solve Rate" if metric == "solve_rate" else "Avg Return"
    ax.set_xlabel(f"Frozen {metric_label}")
    ax.set_ylabel(f"Finetuned {metric_label}")
    ax.set_title(f"Effect of Per-Env Finetuning on {metric_label}")
    ax.legend(loc="lower right", fontsize=9, markerscale=0.7)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(plots_dir / "frozen_vs_finetune_scatter.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 3: Per-environment distribution ─────────────────────────────────────

def plot_per_env_distributions(
    df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Violin plot of per-environment solve rates, one violin per run.

    Reads the full per-env rows (skipping the mean/std summary) from each
    run's ``eval_frozen.csv``.  Wide violins with high median = consistent;
    narrow violins with long tails = fragile (driven by a few easy envs).
    """
    records: list[dict] = []
    varying_cols = _get_varying_hyperparam_cols(df)
    for _, row in df.iterrows():
        csv_path = Path(row["run_dir"]) / "eval_frozen.csv"
        if not csv_path.exists():
            continue
        full = pd.read_csv(csv_path)
        env_rows = full[~full["env"].isin(["mean", "std"])]
        label = _build_run_label(row, varying_cols=varying_cols)
        for _, erow in env_rows.iterrows():
            records.append({
                "run_label": label,
                "repo_type": row["repo_type"],
                "solve_rate": erow["solve_rate"],
            })

    if not records:
        return

    plot_df = pd.DataFrame(records)
    order = (plot_df.groupby("run_label")["solve_rate"]
             .median().sort_values(ascending=False).index.tolist())
    palette = {
        label: _get_color_for_repo(
            plot_df[plot_df["run_label"] == label]["repo_type"].iloc[0]
        )
        for label in order
    }

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.7), 5))
    sns.violinplot(
        data=plot_df, x="run_label", y="solve_rate", hue="run_label",
        order=order, palette=palette, ax=ax, legend=False,
        inner="quartile", cut=0, linewidth=0.8, density_norm="width",
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Per-Environment Solve Rate")
    ax.set_title("Distribution of Solve Rates Across Evaluation Environments")
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(plots_dir / "per_env_distributions.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Public entry point ───────────────────────────────────────────────────────

def generate_comparison_plots(
    df: pd.DataFrame,
    outputs_root: Path | None,
    plots_dir: Path,
) -> None:
    """Generate all cross-experiment comparison plots and a summary CSV.

    Produces:
        - ``comparison_{repo_type}.png`` — one per repo type, comparing
          hyperparam configs within that repo (dual subplot).
        - ``comparison_best.png`` — best config per repo type, side by
          side (dual subplot).
        - ``frozen_vs_finetune_scatter.png`` — all runs scatter.
        - ``per_env_distributions.png`` — violin plot (all runs).
        - ``comparison_summary.csv`` — all metrics in one table.

    Args:
        df: DataFrame from ``collect_experiment_results``.
        outputs_root: root directory containing run folders (used for
            per-env distribution lookups; ignored if None).
        plots_dir: directory to save plots and ``comparison_summary.csv``.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    # ── Per-repo-type plots ────────────────────────────────────────────
    plot_per_repo_comparison_bars(df, plots_dir)

    # ── Best-of-each cross-repo plot ────────────────────────────────────
    best_df = _select_best_per_repo(df)
    if not best_df.empty:
        plot_comparison_bars(
            best_df,
            plots_dir,
            title="Best Configuration Per Repo Type",
            filename="comparison_best.png",
        )

    # ── Scatter plots (all runs) ────────────────────────────────────────
    has_ft = ("finetune_solve_rate_mean" in df.columns
              and df["finetune_solve_rate_mean"].notna().any())
    if has_ft:
        plot_frozen_vs_finetune_scatter(df, plots_dir, metric="solve_rate")
        plot_frozen_vs_finetune_scatter(df, plots_dir, metric="avg_return")

    # ── Per-env distributions (all runs) ────────────────────────────────
    if outputs_root is not None:
        plot_per_env_distributions(df, plots_dir)

    # ── Save summary CSV ────────────────────────────────────────────────
    metric_cols = [
        "repo_type", "repo_display",
        "frozen_solve_rate_mean", "frozen_solve_rate_std",
        "finetune_solve_rate_mean", "finetune_solve_rate_std",
        "frozen_avg_return_mean", "frozen_avg_return_std",
        "finetune_avg_return_mean", "finetune_avg_return_std",
    ]
    extra = [c for c in df.columns if c not in metric_cols
             and c not in ("run_dir",)]
    out_cols = metric_cols + ["run_dir"] + extra
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(plots_dir / "comparison_summary.csv", index=False)

    print(f"Saved comparison plots and summary to {plots_dir}")
