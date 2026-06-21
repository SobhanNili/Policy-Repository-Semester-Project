"""Diagnostic plots for a single experiment run.

Produces all single-experiment plots from saved ``step_logs.pkl`` data.
Orchestrated by ``generate_plots``. Also provides ``save_env_observations``
for rendering environment thumbnail images.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from apa.utils.analysis.common import (
    _figure_saver,
    _plot_boxplot_over_time,
    _thin_ticks,
)

# data container for stuff to plot
@dataclass
class _StepLogData:
    """Precomputed metrics from a list of ``RepoStepLog`` namedtuples.

    Created by ``_unpack_step_logs``; consumed by all ``_plot_*`` functions.
    """
    T: int
    P: int
    E_subset: int
    num_envs: int
    affinities: np.ndarray            # (T, P, E_subset)
    active_masks: np.ndarray          # (T, E_subset, P)
    solve_rates: np.ndarray           # (T, P, E_subset)  post-training
    avg_returns: np.ndarray           # (T, P, E_subset)  post-training
    candidate_env_indices: np.ndarray # (T, E_subset)
    mean_solve_rate: np.ndarray       # (T, P)
    mean_avg_return: np.ndarray       # (T, P)
    n_active_envs: np.ndarray         # (T, P)
    n_greedy_envs: np.ndarray         # (T, P)
    cv_per_env: np.ndarray            # (T, E_subset)
    score_margin_per_env: np.ndarray  # (T, E_subset)
    spearman_pre_frozen: np.ndarray   # (T, E_subset)
    spearman_pre_finetune: np.ndarray # (T, E_subset)
    spearman_post_frozen: np.ndarray  # (T, E_subset)
    assignment: np.ndarray            # (num_envs, T)  -1=contested, NaN=not sampled


def _unpack_step_logs(step_logs: list) -> _StepLogData:
    """Convert a list of ``RepoStepLog`` namedtuples into a ``_StepLogData``
    with all raw arrays stacked plus precomputed derived metrics."""
    T = len(step_logs)
    P = step_logs[0].affinity_scores.shape[0]
    E_subset = step_logs[0].affinity_scores.shape[1]

    # ── Raw arrays ──────────────────────────────────────────────────────
    _s = lambda field: np.stack([np.array(getattr(log, field)) for log in step_logs])

    affinities = _s("affinity_scores")       # (T, P, E_subset)
    active_masks = _s("active_mask")          # (T, E_subset, P)
    solve_rates = _s("solve_rate")            # (T, P, E_subset)  post-training
    avg_returns = _s("avg_return")            # (T, P, E_subset)  post-training
    pre_frozen_solve_rates = _s("pre_frozen_solve_rate")     # (T, P, E_subset)
    pre_finetune_solve_rates = _s("pre_finetune_solve_rate") # (T, P, E_subset)
    candidate_env_indices = _s("candidate_env_indices")      # (T, E_subset)

    # ── Derived metrics ─────────────────────────────────────────────────
    mean_solve_rate, mean_avg_return, n_active_envs = _compute_per_policy_active_metrics(
        active_masks, solve_rates, avg_returns, T, P
    )
    n_greedy_envs = _compute_greedy_env_counts(affinities, T, P)
    cv_per_env = _compute_cv_per_env(affinities)
    score_margin_per_env = _compute_score_margin_per_env(affinities)

    # Per-environment Spearman ρ (affinity vs solve rate, within each column)
    spearman_pre_frozen = _compute_per_env_spearman(
        affinities, pre_frozen_solve_rates, T, P, E_subset
    )
    spearman_pre_finetune = _compute_per_env_spearman(
        affinities, pre_finetune_solve_rates, T, P, E_subset
    )
    spearman_post_frozen = _compute_per_env_spearman(
        affinities, solve_rates, T, P, E_subset
    )

    # Greedy-assignment matrix
    num_envs = int(candidate_env_indices.max()) + 1
    assignment = _compute_assignment_matrix(affinities, candidate_env_indices,
                                            num_envs, T, E_subset)

    return _StepLogData(
        T=T, P=P, E_subset=E_subset, num_envs=num_envs,
        affinities=affinities, active_masks=active_masks,
        solve_rates=solve_rates, avg_returns=avg_returns,
        candidate_env_indices=candidate_env_indices,
        mean_solve_rate=mean_solve_rate, mean_avg_return=mean_avg_return,
        n_active_envs=n_active_envs, n_greedy_envs=n_greedy_envs,
        cv_per_env=cv_per_env, score_margin_per_env=score_margin_per_env,
        spearman_pre_frozen=spearman_pre_frozen,
        spearman_pre_finetune=spearman_pre_finetune,
        spearman_post_frozen=spearman_post_frozen,
        assignment=assignment,
    )


# ── Derived-metric helpers ───────────────────────────────────────────────────

def _compute_per_policy_active_metrics(active_masks, solve_rates, avg_returns,
                                        T, P):
    """Per-policy mean solve rate & avg return over *active* envs only."""
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
    return mean_solve_rate, mean_avg_return, n_active_envs


def _compute_greedy_env_counts(affinities, T, P):
    """Count of candidate envs per policy by raw argmax (before safety nets).

    When multiple policies share the maximum for an env, ALL are counted.
    """
    n_greedy_envs = np.zeros((T, P), dtype=int)
    for t in range(T):
        max_per_col = np.max(affinities[t], axis=0)       # (E_subset,)
        is_max = np.isclose(affinities[t], max_per_col)   # (P, E_subset)
        n_greedy_envs[t] = np.sum(is_max, axis=1)
    return n_greedy_envs


def _compute_cv_per_env(affinities):
    """Coefficient of variation (σ/μ) per candidate env across policies."""
    mean_p = affinities.mean(axis=1)   # (T, E_subset)
    std_p  = affinities.std(axis=1)    # (T, E_subset)
    return std_p / (mean_p + 1e-8)


def _compute_score_margin_per_env(affinities):
    """Normalized top-2 margin: (s_max - s_second) / (s_max - s_min).

    0 ≈ all policies score equally; 1 ≈ one policy clearly dominates.
    """
    sorted_aff = np.sort(affinities, axis=1)  # (T, P, E_subset), ascending
    score_range = sorted_aff[:, -1, :] - sorted_aff[:, 0, :]
    top2_gap = sorted_aff[:, -1, :] - sorted_aff[:, -2, :]
    return np.divide(top2_gap, score_range, out=np.zeros_like(top2_gap),
                     where=(score_range > 1e-12))


def _compute_assignment_matrix(affinities, candidate_env_indices,
                                num_envs, T, E_subset):
    """Greedy-assignment matrix (num_envs, T): policy index or -1 (tied) or NaN."""
    assignment = np.full((num_envs, T), np.nan)
    for t in range(T):
        for local_i in range(E_subset):
            env_idx = int(candidate_env_indices[t, local_i])
            col = affinities[t, :, local_i]
            tied = np.where(np.isclose(col, col.max()))[0]
            assignment[env_idx, t] = -1 if len(tied) > 1 else int(tied[0])
    return assignment


def _compute_per_env_spearman(affinities, solve_rates, T, P, E_subset):
    """Per-environment Spearman ρ between affinity and solve rate.

    For each (t, e), computes ρ = Spearman(affinity[:, e], solve_rate[:, e])
    across the P policies.  This avoids cross-environment scale artifacts that
    plague a single pooled correlation across all envs.
    """
    rho = np.full((T, E_subset), np.nan)
    for t in range(T):
        for e in range(E_subset):
            aff_col = affinities[t, :, e]
            sr_col = solve_rates[t, :, e]
            valid = ~np.isnan(sr_col)
            if (valid.sum() >= 2
                    and np.ptp(aff_col[valid]) > 0
                    and np.ptp(sr_col[valid]) > 0):
                rho[t, e], _ = spearmanr(aff_col[valid], sr_col[valid])
    return rho


# ═══════════════════════════════════════════════════════════════════════════════
# Plot functions
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_performance(data: _StepLogData, plots_dir: Path, colors):
    """Per-policy mean solve rate and avg return over steps."""
    T, P = data.T, data.P
    steps = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for p in range(P):
        ax1.plot(steps, data.mean_solve_rate[:, p],
                 color=colors[p], lw=2, label=f"Policy {p}",
                 marker=".", markersize=4)
    ax1.set_ylabel("Episode Solve Rate")
    ax1.set_title("Policy Performance on Assigned Environments")
    ax1.legend(ncol=P, loc="lower right")
    ax1.set_ylim(-0.05, 1.05)

    for p in range(P):
        ax2.plot(steps, data.mean_avg_return[:, p],
                 color=colors[p], lw=2, label=f"Policy {p}",
                 marker=".", markersize=4)
    ax2.set_ylabel("Avg. Episode Return")
    ax2.set_xlabel("Repository Training Step")
    ax2.legend(ncol=P, loc="lower right")

    fig.tight_layout()
    fig.savefig(plots_dir / "performance_over_training.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_active_envs(data: _StepLogData, plots_dir: Path, colors):
    """Stackplot of active environment counts per policy (post-mask)."""
    with _figure_saver(plots_dir, "assigned_envs.png", figsize=(10, 4)) as (fig, ax):
        ax.stackplot(np.arange(data.T), data.n_active_envs.T,
                     labels=[f"Policy {p}" for p in range(data.P)],
                     colors=colors, alpha=0.85)
        ax.set_xlabel("Repository Training Step")
        ax.set_ylabel("# Assigned Environments")
        ax.set_title("Number of Assigned Environments to Each Policy")
        ax.legend(loc="upper right")
        ax.set_xlim(0, data.T - 1)


def _plot_env_greedy_policy_choice(data: _StepLogData, plots_dir: Path, colors):
    """Stackplot of raw argmax env counts per policy (before safety nets).

    Contrast with ``_plot_active_envs``: if this plot shows one policy
    hoarding all envs while the active-envs plot is balanced, the safety
    net is doing heavy lifting.
    """
    steps = np.arange(1, data.T)
    with _figure_saver(plots_dir, "env_greedy_policy_choices.png",
                       figsize=(10, 4)) as (fig, ax):
        ax.stackplot(steps, data.n_greedy_envs[1:].T,
                     labels=[f"Policy {p}" for p in range(data.P)],
                     colors=colors, alpha=0.85)
        ax.set_xlabel("Repository Training Step")
        ax.set_ylabel("# Attracted Environments")
        ax.set_title("Number of Attracted Environments per Policy (Greedy Affinity Choice)")
        ax.legend(loc="upper right")
        ax.set_xlim(1, data.T - 1)
        ax.axhline(data.E_subset, color="gray", ls="--", lw=1, alpha=0.5,
                   label=f"Candidate pool ({data.E_subset})")


def _plot_affinity_score_spread_cv(data: _StepLogData, plots_dir: Path):
    """Box plot of per-environment CV (σ/μ) at each repo step."""
    _plot_boxplot_over_time(
        data.cv_per_env, plots_dir, "score_spread_cv.png",
        title="Affinity Score Spread — Coefficient of Variation Over Training",
        ylabel=r"CV ($\sigma$ / $\mu$) per Environment",
        ylim=(data.cv_per_env.min() - 0.1, data.cv_per_env.max() + 0.1),
        facecolor="lightblue", overlay_label="Mean CV",
    )


def _plot_affinity_score_spread_margin(data: _StepLogData, plots_dir: Path):
    """Box plot of per-environment Normalized Margin at each repo step.

    0 = all policies score equally (contested).
    1 = one policy clearly dominates (confident assignment).
    """
    _plot_boxplot_over_time(
        data.score_margin_per_env, plots_dir, "score_spread_margin.png",
        title="Affinity Score Spread — Normalized Top-2 Margin",
        ylabel="Normalized Margin per Environment",
        ylim=(-0.05, 1.05),
        facecolor="lightgreen", overlay_label="Mean Normalized Margin",
    )


def _plot_assignment_heatmap(data: _StepLogData, plots_dir: Path, colors):
    """Env x step heatmap of greedy argmax assignments.

    Rows = envs (0 … E-1); Columns = repo steps.
    Grey = contested; White = not sampled.
    """
    num_envs, T, P = data.num_envs, data.T, data.P
    contested_color = "#bdbdbd"
    cmap = ListedColormap([contested_color] + list(colors))
    cmap.set_bad("whitesmoke")

    fig_height = np.clip(num_envs * 0.15, 4, 20)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    masked = np.ma.masked_invalid(data.assignment)
    mesh = ax.pcolormesh(np.arange(T + 1), np.arange(num_envs + 1),
                         masked, cmap=cmap, vmin=-1.5, vmax=P - 0.5,
                         edgecolors="none", linewidth=0, shading="flat")

    cbar = plt.colorbar(mesh, ax=ax, ticks=np.arange(-1, P))
    cbar.set_label("Assigned Policy")
    cbar.ax.set_yticklabels(["Contested"] + [f"Policy {p}" for p in range(P)])

    ax.set_xlabel("Repository Training Step")
    ax.set_ylabel("Environment Index")
    ax.set_title("Greedy Policy Assignment per Environment Over Training")
    _thin_ticks(ax, np.arange(T) + 0.5, max_ticks=15)
    ax.set_yticks(np.arange(num_envs) + 0.5)
    y_fs = 8 if num_envs <= 50 else (6 if num_envs <= 100 else 4)
    ax.set_yticklabels(np.arange(num_envs), fontsize=y_fs)
    ax.set_xlim(0, T)
    ax.set_ylim(0, num_envs)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(plots_dir / "assignment_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_env_spearman(data: _StepLogData, plots_dir: Path,
                            rho_key: str, title: str, filename: str):
    """Box plot of per-environment Spearman ρ at each repo step."""
    rho_per_env = getattr(data, rho_key)  # (T, E_subset)
    _plot_boxplot_over_time(
        rho_per_env, plots_dir, filename,
        title=title,
        ylabel=r"Spearman $\rho$ per Environment",
        ylim=(-1.05, 1.05),
        facecolor="lightblue", overlay_label=r"Mean $\rho$",
        reference_line=0, reference_label=r"$\rho = 0$",
    )


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
    """Generate per-step grid images showing assigned environments.

    Rows = policies; Columns = env slots.  Each cell shows the env
    observation image with a coloured border:

    ======== ======== ==========================
    Colour   Style    Meaning
    ======== ======== ==========================
    Green    Solid    Greedy choice — solved
    Red      Solid    Greedy choice — did not solve
    Green    Dashed   Safety-net / exploration — solved
    Red      Dashed   Safety-net / exploration — did not solve
    ======== ======== ==========================

    Images saved to ``output_dir/plots/assignment_grids/step_XXXX.png``.
    """
    data = _unpack_step_logs(step_logs)
    T, P, E_subset = data.T, data.P, data.E_subset
    obs_path = output_dir / obs_dir

    if not obs_path.is_dir():
        print(f"Warning: observation directory not found — "
              f"skipping assignment grids: {obs_path}")
        return

    n_cols = max_envs_per_policy or min(E_subset, 10)
    if n_cols == 0:
        return

    grids_dir = output_dir / "plots" / "assignment_grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load env images
    num_envs = data.num_envs
    env_images: dict[int, np.ndarray] = {}
    for i in range(num_envs):
        p = obs_path / f"env_{i:03d}.png"
        if p.exists():
            env_images[i] = plt.imread(str(p))

    GREEN, RED, GREY = "#5b9a5e", "#d35e5e", "#e8e8e8"
    steps_to_plot = list(range(0, T, step_interval))

    for t in steps_to_plot:
        aff_t = data.affinities[t]  # (P, E_subset)
        max_per_col = np.max(aff_t, axis=0)
        is_greedy = np.isclose(aff_t, max_per_col)  # (P, E_subset) bool

        # Collect assigned envs per policy
        policy_envs: list[list[tuple[int, float, bool]]] = []
        for p in range(P):
            active = data.active_masks[t, :, p].astype(bool)
            idx = np.where(active)[0]
            gids = data.candidate_env_indices[t][idx]
            srs = data.solve_rates[t, p, idx]
            greedy = is_greedy[p, idx]
            info = [(int(g), float(s), bool(ig))
                    for g, s, ig in zip(gids, srs, greedy)]
            info.sort(key=lambda x: x[0])
            policy_envs.append(info)

        fig, axes = plt.subplots(
            P, n_cols,
            figsize=(n_cols * cell_size_inches, P * cell_size_inches + 0.55),
            squeeze=False, facecolor="white",
        )
        fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.10,
                            wspace=0.10, hspace=0.14)
        fig.text(0.5, 0.965, f"Step {t}", ha="center", va="top",
                 fontsize=12, fontweight="bold")

        for p in range(P):
            envs = policy_envs[p]
            n_envs = len(envs)
            n_greedy = sum(1 for _, _, ig in envs if ig)
            truncated = n_envs > n_cols
            shown = envs[:n_cols]

            for col in range(n_cols):
                ax = axes[p, col]
                ax.set_xticks([]), ax.set_yticks([])
                ax.tick_params(left=False, bottom=False)

                if col < len(shown):
                    gid, sr, is_greedy = shown[col]
                    solved = (not np.isnan(sr)) and (sr > solve_threshold)

                    if gid in env_images:
                        ax.imshow(env_images[gid])
                    else:
                        ax.set_facecolor("#ddd")

                    c = GREEN if solved else RED
                    lw = 3.0 if is_greedy else 1.6
                    ls = "-" if is_greedy else "--"
                    for spine in ax.spines.values():
                        spine.set_color(c)
                        spine.set_linewidth(lw)
                        spine.set_linestyle(ls)
                        spine.set_visible(True)

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
            axes[p, 0].set_ylabel("\n".join(parts), fontsize=7.5,
                                   fontweight="bold", rotation=0,
                                   ha="right", va="center", labelpad=38)

        # Legend
        legend_handles = [
            Line2D([0], [0], color=GREEN, lw=2.5, label="Greedy, solved"),
            Line2D([0], [0], color=RED,   lw=2.5, label="Greedy, fail"),
            Line2D([0], [0], color=GREEN, lw=1.4, ls="--", label="Safety-net, solved"),
            Line2D([0], [0], color=RED,   lw=1.4, ls="--", label="Safety-net, fail"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                   fontsize=7, frameon=False, handlelength=1.2,
                   handletextpad=0.5, columnspacing=1.0)

        fig.savefig(grids_dir / f"step_{t:04d}.png", dpi=150,
                    bbox_inches="tight", facecolor="white", edgecolor="none")
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
    """Bar plot: number of evaluation environments assigned to each policy.

    Assignment is determined by greedy argmax over affinity scores, so it
    is identical for frozen and finetuned evaluation.
    """
    env2policy = np.asarray(env2policy, dtype=int)
    counts = np.bincount(env2policy, minlength=num_policies)
    P = num_policies
    x = np.arange(P)

    fig, ax = plt.subplots(figsize=(max(6, P * 0.9), 4.8))
    fig.patch.set_facecolor("white")

    ax.bar(x, counts, color=sns.color_palette("pastel", P),
           edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Policy {p}" for p in range(P)], fontsize=10)
    ax.set_ylabel("# of Environments Assigned", fontsize=11)
    ax.set_title("Evaluation Environment Assignment Distribution",
                 fontsize=13, fontweight="bold")

    y_max = counts.max()
    ax.set_ylim(0, y_max + max(2, y_max * 0.15))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "eval_env_assignment_dist.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_eval_assignment_from_run(run_dir: Path) -> None:
    """Load eval assignment from saved pickle and generate the bar plot."""
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

    plot_eval_assignment_bars(np.array(env2policy), num_policies,
                              run_dir / "plots")


def _group_eval_env_images_by_policy(run_dir: Path) -> None:
    """Copy eval env images into per-policy subfolders for visual browsing.

    Reads ``env2policy`` from ``eval_frozen_results.pkl`` and copies each
    ``env_{i:03d}.png`` into ``eval_env_observations/by_policy/policy_{p}/``.
    """
    frozen_path = run_dir / "eval_frozen_results.pkl"
    config_path = run_dir / "config.json"
    obs_dir = run_dir / "eval_env_observations"

    if not frozen_path.exists() or not config_path.exists():
        return
    if not obs_dir.is_dir():
        return

    with open(frozen_path, "rb") as f:
        _, env2policy, _ = pickle.load(f)
    with open(config_path) as f:
        config = json.load(f)
    num_policies = int(config.get("num_policies", 0))
    if num_policies == 0:
        return

    env2policy = np.asarray(env2policy, dtype=int)
    by_policy_dir = obs_dir / "by_policy"
    if by_policy_dir.exists():
        shutil.rmtree(by_policy_dir)

    copied = 0
    for i, p in enumerate(env2policy):
        src = obs_dir / f"env_{i:03d}.png"
        if not src.exists():
            continue
        dst_dir = by_policy_dir / f"policy_{p}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / f"env_{i:03d}.png")
        copied += 1

    print(f"Grouped {copied} eval env images into {num_policies} policy "
          f"subfolders under {by_policy_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Environment observation visualization
# ═══════════════════════════════════════════════════════════════════════════════

def save_env_observations(env_params, static_env_params,
                           env_reset_states, output_dir: Path,
                           folder: str = "env_train_observations") -> None:
    """Render and save first-observation images for a set of environments.

    Each image is saved as ``env_{index:03d}.png`` in ``output_dir / folder/``.
    The render+rotate pattern matches ``DinoMatchingRepo._encode_pixels``.
    """
    from kinetix.render.renderer_pixels import make_render_pixels_rl

    renderer = make_render_pixels_rl(env_params, static_env_params)
    pixels = jax.vmap(lambda s: renderer(s.env_state))(env_reset_states).image
    pixels_np = np.array(pixels)
    pixels_np = np.rot90(pixels_np, k=1, axes=(1, 2))

    out_dir = output_dir / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(pixels_np.shape[0]):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(pixels_np[i])
        ax.axis("off")
        fig.savefig(out_dir / f"env_{i:03d}.png",
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"Saved {pixels_np.shape[0]} environment observation images to {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Public orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_plots(step_logs: list, output_dir: Path) -> None:
    """Generate all diagnostic plots from step_logs, saved to output_dir/plots/.

    Produces:
        - Policy Performance (per-policy mean solve rate & avg return)
        - Active/Attracted Environments stackplots
        - Assignment Certainty (CV and Normalized Margin box plots)
        - Greedy Assignment Heatmap (env x step)
        - Affinity-Solve Rate Spearman ρ (Pre-Training, Frozen + Finetuned)
        - Policy-Environment Assignment Grids (per-step thumbnails)
        - Evaluation Assignment Distribution bar plot
    """
    if not step_logs:
        print("Warning: empty step_logs — nothing to plot.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _unpack_step_logs(step_logs)
    colors = sns.color_palette("muted", data.P)

    # Core diagnostics
    _plot_performance(data, plots_dir, colors)
    _plot_active_envs(data, plots_dir, colors)
    _plot_env_greedy_policy_choice(data, plots_dir, colors)
    _plot_affinity_score_spread_cv(data, plots_dir)
    _plot_affinity_score_spread_margin(data, plots_dir)
    _plot_assignment_heatmap(data, plots_dir, colors)

    # Spearman correlation
    _plot_per_env_spearman(
        data, plots_dir, rho_key="spearman_pre_frozen",
        title="Affinity–Solve Rate Spearman ρ (Frozen, Pre-Training)",
        filename="affinity_sr_spearman_frozen.png",
    )
    _plot_per_env_spearman(
        data, plots_dir, rho_key="spearman_pre_finetune",
        title="Affinity–Solve Rate Spearman ρ (Finetuned, Pre-Training)",
        filename="affinity_sr_spearman_finetune.png",
    )

    # Per-step assignment grids
    grid_interval = max(1, data.T // 20)
    generate_env_assignment_grids(step_logs, output_dir, step_interval=grid_interval)

    # Eval assignment plots (require saved evaluation results, not step logs)
    _plot_eval_assignment_from_run(output_dir)
    _group_eval_env_images_by_policy(output_dir)

    print(f"Saved all plots to {plots_dir}")
