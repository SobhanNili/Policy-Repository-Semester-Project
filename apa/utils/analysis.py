"""Analysis and plotting utilities for policy repository experiments.

This module is the single source of truth for diagnostic plotting logic.
It is used by both ``scripts/run.py`` (during training) and
``scripts/plot_results.py`` (standalone post-hoc analysis).
"""

import matplotlib
matplotlib.use("Agg")  # headless — must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr


def generate_plots(step_logs: list, output_dir: Path) -> None:
    """Generate all 5 diagnostic plots from step_logs and save to output_dir/plots/.

    Args:
        step_logs: list of RepoStepLog namedtuples from a completed run.
        output_dir: path to the run's output directory.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    T = len(step_logs)
    P = step_logs[0].affinity_scores.shape[0]

    # ── Unpack step_logs into numpy arrays ──────────────────────────────────
    affinities = np.stack([np.array(log.affinity_scores) for log in step_logs])  # (T, P, E)
    active_masks = np.stack([np.array(log.active_mask) for log in step_logs])  # (T, E, P)
    solve_rates = np.stack([np.array(log.solve_rate) for log in step_logs])  # (T, P, E)
    avg_returns = np.stack([np.array(log.avg_return) for log in step_logs])  # (T, P, E)

    # ── Derived metrics ─────────────────────────────────────────────────────
    active_for_metric = active_masks.transpose(0, 2, 1)  # (T, P, E)

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

    # ── Utility margin: per-env gap between best and second-best raw score ──
    # Sorted along policy axis: best (index -1), second-best (index -2)
    sorted_aff = np.sort(affinities, axis=1)  # (T, P, E) — ascending
    per_env_margins = sorted_aff[:, -1, :] - sorted_aff[:, -2, :]  # (T, E)
    mean_margin = per_env_margins.mean(axis=1)
    std_margin = per_env_margins.std(axis=1)

    # Spearman correlation
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

    steps = np.arange(T)
    colors = sns.color_palette("muted", P)

    # ── Plot 1: Policy Performance Over Steps ───────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for p in range(P):
        ax1.plot(
            steps,
            mean_solve_rate[:, p],
            color=colors[p],
            lw=2,
            label=f"Policy {p}",
            marker=".",
            markersize=4,
        )
    ax1.set_ylabel("Solve Rate")
    ax1.set_title("Policy Performance on Active Envs Over Training")
    ax1.legend(ncol=P, loc="lower right")
    ax1.set_ylim(-0.05, 1.05)

    for p in range(P):
        ax2.plot(
            steps,
            mean_avg_return[:, p],
            color=colors[p],
            lw=2,
            label=f"Policy {p}",
            marker=".",
            markersize=4,
        )
    ax2.set_ylabel("Avg Return")
    ax2.set_xlabel("Repo Step")
    ax2.legend(ncol=P, loc="lower right")

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_1_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: Active Envs Per Policy ──────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ax.stackplot(
        steps,
        n_active_envs.T,
        labels=[f"Policy {p}" for p in range(P)],
        colors=colors,
        alpha=0.85,
    )
    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Number of Active Envs")
    ax.set_title("Environments Assigned Per Policy Over Training")
    ax.legend(loc="upper right")
    ax.set_xlim(0, T - 1)

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_2_active_envs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Assignment Decisiveness (Utility Margin) ────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ax.plot(steps, mean_margin, color="tab:blue", lw=2, label="Mean margin")
    ax.fill_between(
        steps,
        mean_margin - std_margin,
        mean_margin + std_margin,
        alpha=0.2,
        color="tab:blue",
        label="±1 std",
    )
    ax.axhline(0, color="gray", ls="--", lw=1, label="Zero margin (ties)")
    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Mean Utility Margin")
    ax.set_title("Assignment Decisiveness Over Training")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_3_margin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: Solve Rate by Affinity Quantile ─────────────────────────────
    n_bins = 5
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binned_solve_rate = np.full((T, n_bins), np.nan)
    bin_labels = [
        f"[{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f})" for i in range(n_bins)
    ]

    for t in range(T):
        aff_flat = affinities[t].ravel()
        sr_flat = solve_rates[t].ravel()
        valid = ~np.isnan(sr_flat)
        if valid.any():
            aff_v = aff_flat[valid]
            sr_v = sr_flat[valid]
            for b in range(n_bins):
                lo, hi = bin_edges[b], bin_edges[b + 1]
                if b == n_bins - 1:
                    in_bin = (aff_v >= lo) & (aff_v <= hi)
                else:
                    in_bin = (aff_v >= lo) & (aff_v < hi)
                if in_bin.sum() > 0:
                    binned_solve_rate[t, b] = sr_v[in_bin].mean()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    palette = sns.color_palette("viridis", n_bins)

    for b in range(n_bins):
        ax.plot(
            steps,
            binned_solve_rate[:, b],
            color=palette[b],
            lw=2,
            label=bin_labels[b],
            marker=".",
            markersize=4,
        )
    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Mean Solve Rate")
    ax.set_title("Solve Rate by Affinity Quintile Over Training")
    ax.legend(title="Affinity bin", loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(
        plots_dir / "plot_4_affinity_quantiles.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ── Plot 5: Affinity–Performance Spearman Correlation ───────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    valid = ~np.isnan(spearman_rhos)
    ax.plot(
        steps[valid],
        spearman_rhos[valid],
        color="tab:green",
        lw=2,
        marker=".",
        markersize=4,
    )
    ax.axhline(0, color="gray", ls="--", lw=1, label="ρ = 0 (no correlation)")
    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Affinity–Solve Rate Correlation Over Training")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_5_spearman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 5 plots to {plots_dir}")
