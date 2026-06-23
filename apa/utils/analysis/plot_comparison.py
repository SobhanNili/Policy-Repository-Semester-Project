"""Cross-experiment comparison plots and data collection.

Scans experiment output directories, aggregates metrics, and produces
comparison plots across different repo types and hyperparams, and also a summary CSV.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
from matplotlib.colors import ListedColormap

from apa.utils.analysis.common import (
    _build_run_label,
    _desaturate,
    _figure_saver,
    _get_color_for_repo,
    _get_display_name,
    _get_varying_hyperparam_cols,
    _parse_repo_type_from_name,
)

def collect_experiment_results(
    outputs_root: Path | None = None,
    run_dirs: list[Path] | None = None,
) -> pd.DataFrame:
    """Scan experiment folders and return a DataFrame with one row per run.

    Two input modes:
      - **Scan mode**: provide ``outputs_root`` — finds all subdirectories
        that contain a ``config.json``.
      - **Explicit mode**: provide ``run_dirs`` — uses only those directories.

    Returns:
        DataFrame with columns: run_dir, repo_type, repo_display,
        frozen_solve_rate_mean, frozen_solve_rate_std, frozen_avg_return_mean,
        frozen_avg_return_std, finetune_solve_rate_mean, finetune_solve_rate_std,
        finetune_avg_return_mean, finetune_avg_return_std, plus hyperparameter
        columns (seed, sigma, history_decay, mode, …).
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

        # repo_type: prefer config.json, fall back to directory name
        repo_type = config.get("repo_type")
        if repo_type is None:
            repo_type = _parse_repo_type_from_name(run_dir.name)
        if repo_type is None:
            continue

        kwargs = {}
        if kwargs_path.exists():
            with open(kwargs_path) as f:
                kwargs = json.load(f)

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

        # Flatten scalar hyperparams from both files
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

def _meta_group_keys(df: pd.DataFrame) -> list[str]:
    """Grouping columns: repo_type, display name, plus varying hyperparams."""
    keys = ["repo_type", "repo_display"]
    keys.extend(_get_varying_hyperparam_cols(df))
    return keys


def _select_best_config_per_repo(
    df: pd.DataFrame,
    metric: str = "finetune_solve_rate_mean",
    all_seeds: bool = False,
) -> pd.DataFrame:
    """Select the best hyperparameter configuration per repo type.

    1. Groups runs by ``(repo_type, hyperparams)``, **excluding seed**, and
       averages all metrics across seeds within each group.
    2. Picks the group with the highest mean *metric* per repo type.
    3. Finds the smallest seed common to **all** repo types (so evaluation
       environments are comparable across repos).
    4. Returns either:
       - The single original DataFrame row for the common seed
         (*all_seeds* = False, default; for checkerboards / ARI — one column
         per repo).
       - All original DataFrame rows for the winning config
         (*all_seeds* = True; for bar plots — enables cross-seed error bars).

    Falls back to ``frozen_solve_rate_mean`` if *metric* is unavailable.
    When ``seed`` is absent from the DataFrame (older experiments without
    seed logging), falls back to the old per-row best-pick behaviour.
    """
    if df.empty:
        return pd.DataFrame()

    # ── Determine grouping columns (hyperparams only, not seed) ────────
    # seed is excluded from varying cols because it's in _IDENTITY_COLS.
    varying = _get_varying_hyperparam_cols(df)
    group_cols = ["repo_type"] + [c for c in varying if c in df.columns]

    has_seed = "seed" in df.columns

    if not has_seed or len(group_cols) <= 1:
        # No seed column or no hyperparams → fall back to old row-level pick.
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

    # ── Aggregate metrics across seeds within each config ──────────────
    metric_patterns = ("frozen_", "finetune_")
    metric_cols = [c for c in df.columns if c.startswith(metric_patterns)]

    agg_dict: dict = {c: "mean" for c in metric_cols}
    # Collect all seeds belonging to this config
    agg_dict["seed"] = lambda x: sorted(x.dropna().unique().tolist())
    agg_dict["run_dir"] = "first"  # placeholder for later lookup

    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    # ── Find smallest seed common to ALL repo types ────────────────────
    seeds_by_repo: dict[str, set] = {}
    for rt in df["repo_type"].unique():
        seeds_by_repo[rt] = set(
            df.loc[df["repo_type"] == rt, "seed"].dropna().unique()
        )
    if seeds_by_repo:
        common = set.intersection(*seeds_by_repo.values())
        common_seed = min(common) if common else None
    else:
        common_seed = None

    # ── Pick the best config per repo type ─────────────────────────────
    best_rows: list[dict] = []
    for repo_type, group in grouped.groupby("repo_type", dropna=False):
        if metric in group.columns and group[metric].notna().any():
            idx = group[metric].idxmax()
        elif "frozen_solve_rate_mean" in group.columns:
            idx = group["frozen_solve_rate_mean"].idxmax()
        else:
            idx = group.index[0]
        best = group.loc[idx].to_dict()

        # Choose representative seed for this config (used only when
        # all_seeds=False — for checkerboards / ARI).
        available = best.get("seed", [])
        if common_seed is not None and common_seed in available:
            chosen = common_seed
        elif available:
            chosen = available[0]  # smallest available as fallback
        else:
            chosen = None
        best["_chosen_seed"] = chosen
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows)

    # ── Map back to original DataFrame rows ────────────────────────────
    result_rows: list[dict] = []
    for _, best in best_df.iterrows():
        # Build mask for the winning config (repo_type + hyperparams)
        mask = df["repo_type"] == best["repo_type"]
        for col in group_cols:
            if col == "repo_type" or col not in df.columns:
                continue
            val = best[col]
            if pd.isna(val):
                mask &= df[col].isna()
            else:
                mask &= df[col] == val

        if all_seeds:
            # Bar plots: return all seeds so error bars show cross-seed std.
            for _, row in df[mask].iterrows():
                result_rows.append(row.to_dict())
        else:
            # Checkerboard / ARI: only the common-seed row (one column per repo).
            chosen_seed = best.get("_chosen_seed")
            if chosen_seed is not None and has_seed:
                mask &= df["seed"] == chosen_seed
            match = df[mask]
            if not match.empty:
                result_rows.append(match.iloc[0].to_dict())

    return pd.DataFrame(result_rows)


def _collect_eval_assignments(df: pd.DataFrame) -> dict | None:
    """Load ``env2policy`` from each run's ``eval_frozen_results.pkl``.

    Returns:
        dict with keys assignment_matrix (max_E, N), run_labels (list[str]),
        repo_types (list[str]), num_policies (int).  None if no data found.
    """
    records: list[dict] = []
    max_E = 0

    for _, row in df.iterrows():
        pkl_path = Path(row["run_dir"]) / "eval_frozen_results.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            _, env2policy, _ = pickle.load(f)
        env2policy = np.asarray(env2policy, dtype=int)
        max_E = max(max_E, len(env2policy))
        records.append({
            "env2policy": env2policy,
            "repo_type": row["repo_type"],
            "row": row,
        })

    if not records:
        return None

    N = len(records)
    assignment = np.full((max_E, N), np.nan)
    for j, rec in enumerate(records):
        E = len(rec["env2policy"])
        assignment[:E, j] = rec["env2policy"]

    varying_cols = _get_varying_hyperparam_cols(df)
    run_labels = [_build_run_label(rec["row"], varying_cols=varying_cols,
                                   max_dir_chars=14) for rec in records]

    return {
        "assignment_matrix": assignment,
        "run_labels": run_labels,
        "repo_types": [rec["repo_type"] for rec in records],
        "num_policies": int(np.nanmax(assignment)) + 1,
    }


def _draw_column_group_backgrounds(
    ax: plt.Axes,
    repo_types: np.ndarray,
    unique_repos: np.ndarray,
    N: int,
    E: int,
) -> None:
    """Subtle alternating backgrounds and coloured accent lines for repo groups."""
    for i, rt in enumerate(unique_repos):
        idx = np.where(repo_types == rt)[0]
        if len(idx) == 0:
            continue
        x_start, x_end = idx[0], idx[-1] + 1
        if i % 2 == 1:
            ax.axvspan(x_start, x_end, ymin=0, ymax=1,
                       facecolor="#f4f4f4", zorder=0, linewidth=0)
        color = _get_color_for_repo(rt)
        ax.plot([x_start, x_end], [E, E], color=color, lw=2.5,
                solid_capstyle="butt", clip_on=False, zorder=8)


# Grouped bar chart
def plot_comparison_bars(
    df: pd.DataFrame,
    plots_dir: Path,
    title: str | None = None,
    filename: str = "comparison_bars.svg",
    finetune_only: bool = False,
    reference_repos: dict[str, str] | None = None,
) -> None:
    """Grouped bar chart: frozen vs finetuned mean ± std per (repo_type, config).

    If multiple runs share the same config (different seeds), they are
    averaged and std is computed across seeds.

    Args:
        finetune_only: if True, only finetuned bars are shown (no frozen).
        reference_repos: optional ``{repo_type: legend_label}`` mapping for
            horizontal reference lines (e.g. baseline / control markers).
    """
    group_keys = _meta_group_keys(df)
    varying_cols = _get_varying_hyperparam_cols(df)

    def _build_agg(metric: str) -> pd.DataFrame:
        if finetune_only:
            agg = df.groupby(group_keys, dropna=False).agg(
                finetune_mean=(f"finetune_{metric}_mean", "mean"),
                finetune_std=(f"finetune_{metric}_mean", "std"),
                n=(f"finetune_{metric}_mean", "count"),
            ).reset_index()
            agg["finetune_std"] = agg["finetune_std"].fillna(0)
        else:
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
    sort_col = "finetune_mean" if finetune_only else "frozen_mean"
    order = agg_sr.sort_values(sort_col, ascending=False).index
    agg_sr = agg_sr.loc[order].reset_index(drop=True)
    agg_ar = agg_ar.loc[order].reset_index(drop=True)

    N = len(agg_sr)
    if N == 0:
        return

    # Build labels from dynamic varying columns
    labels = []
    for _, row in agg_sr.iterrows():
        label = row["repo_display"]
        extras = [f"{k}={row[k]}" for k in varying_cols
                  if k in row and pd.notna(row[k])]
        if extras:
            label += f"  [{', '.join(extras)}]"
        labels.append(label)

    x = np.arange(N)
    colors = [_get_color_for_repo(t) for t in agg_sr["repo_type"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, N * 0.9), 8),
                                   sharex=True)

    if finetune_only:
        # ── Finetune-only mode: single bar per group ──────────────────
        bar_width = 0.5

        def _draw_bars(ax, agg, metric_label: str):
            ax.bar(x, agg["finetune_mean"], bar_width,
                   yerr=agg["finetune_std"], capsize=4,
                   color=[_desaturate(c, 0.5) for c in colors],
                   edgecolor="white", linewidth=0.8, hatch="//")
            for i, (m, s, n) in enumerate(zip(agg["finetune_mean"],
                                               agg["finetune_std"], agg["n"])):
                if n > 1:
                    ax.annotate(f"n={n}", (i, m + s + 0.015),
                                ha="center", fontsize=7, color="gray")
            ax.set_ylabel(metric_label)
            all_means = agg["finetune_mean"].values
            all_stds = agg["finetune_std"].values
            ax.set_ylim(
                min(0, (all_means - all_stds).min() - 0.05),
                max(1.05, (all_means + all_stds).max() + 0.05),
            )
            return _draw_reference_lines(ax, agg, "finetune_mean")

    else:
        # ── Normal mode: frozen + finetuned side by side ─────────────
        width = 0.35

        def _draw_bars(ax, agg, metric_label: str):
            ax.bar(x - width / 2, agg["frozen_mean"], width,
                   yerr=agg["frozen_std"], capsize=4,
                   color=colors, edgecolor="white", linewidth=0.8)
            ax.bar(x + width / 2, agg["finetune_mean"], width,
                   yerr=agg["finetune_std"], capsize=4,
                   color=[_desaturate(c, 0.5) for c in colors],
                   edgecolor="white", linewidth=0.8, hatch="//")
            for i, (m, s, n) in enumerate(zip(agg["frozen_mean"],
                                               agg["frozen_std"], agg["n"])):
                if n > 1:
                    ax.annotate(f"n={n}", (i - width / 2, m + s + 0.015),
                                ha="center", fontsize=7, color="gray")
            ax.set_ylabel(metric_label)
            all_means = np.concatenate([agg["frozen_mean"].values,
                                         agg["finetune_mean"].values])
            all_stds = np.concatenate([agg["frozen_std"].values,
                                        agg["finetune_std"].values])
            ax.set_ylim(
                min(0, (all_means - all_stds).min() - 0.05),
                max(1.05, (all_means + all_stds).max() + 0.05),
            )
            return _draw_reference_lines(ax, agg, "finetune_mean")

    def _draw_reference_lines(ax, agg, value_col: str) -> list:
        """Draw horizontal reference lines; return legend handles."""
        if not reference_repos:
            return []
        handles = []
        for rt, ref_label in reference_repos.items():
            mask = agg["repo_type"] == rt
            if not mask.any():
                continue
            val = agg.loc[mask, value_col].values[0]
            if pd.isna(val):
                continue
            color = _get_color_for_repo(rt)
            ax.axhline(val, color=color, linestyle="--", linewidth=1.2,
                       alpha=0.85, zorder=5)
            handles.append(
                mpatches.Patch(facecolor=color, edgecolor="white",
                               linewidth=0.5, label=ref_label)
            )
        return handles

    ref_sr = _draw_bars(ax1, agg_sr, "Solve Rate")
    ref_ar = _draw_bars(ax2, agg_ar, "Avg Return")

    # Collect unique reference legend handles (deduplicate across axes).
    all_ref = ref_sr + ref_ar
    unique_ref: list = []
    if all_ref:
        seen: set[str] = set()
        for h in all_ref:
            lbl = h.get_label()
            if lbl not in seen:
                seen.add(lbl)
                unique_ref.append(h)

    # ── Single figure legend (frozen/finetuned + reference lines) ─────
    fig_handles: list = []
    if not finetune_only:
        fig_handles.extend([
            mpatches.Patch(facecolor="#888888", edgecolor="white",
                           linewidth=0.8, label="Frozen"),
            mpatches.Patch(facecolor="#cccccc", edgecolor="white",
                           linewidth=0.8, hatch="//", label="Finetuned"),
        ])
    if unique_ref:
        fig_handles.extend(unique_ref)

    if fig_handles:
        fig.legend(handles=fig_handles, loc="lower center",
                   ncol=len(fig_handles), fontsize=8.5,
                   frameon=True, edgecolor="#e0e0e0",
                   facecolor="white", framealpha=0.9,
                   handlelength=1.2, handletextpad=0.5,
                   columnspacing=1.2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    fig.suptitle(title or "Policy Repo Comparison", fontsize=13, fontweight="bold")

    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_repo_comparison_bars(df: pd.DataFrame, plots_dir: Path) -> None:
    """One dual-subplot bar chart per repo type comparing its own configs."""
    for repo_type in sorted(df["repo_type"].unique()):
        sub_df = df[df["repo_type"] == repo_type]
        if sub_df.empty:
            continue
        plot_comparison_bars(
            sub_df, plots_dir,
            title=f"{_get_display_name(repo_type)} — Hyperparameter Comparison",
            filename=f"comparison_{repo_type}.svg",
        )

# Frozen vs Finetuned scatter
def plot_frozen_vs_finetune_scatter(
    df: pd.DataFrame,
    plots_dir: Path,
    metric: str = "solve_rate",
) -> None:
    """Scatter of frozen vs finetuned metric, one point per run.

    Points above the diagonal = fine-tuning helps.
    Color = repo type; marker size ∝ frozen std (larger = more variable).
    """
    frozen_col = f"frozen_{metric}_mean"
    ft_col = f"finetune_{metric}_mean"
    df_plot = df.dropna(subset=[frozen_col, ft_col])
    if df_plot.empty:
        return

    with _figure_saver(plots_dir, f"frozen_vs_finetune_{metric}_scatter.svg",
                       figsize=(7, 7)) as (fig, ax):
        for rt in sorted(df_plot["repo_type"].unique()):
            sub = df_plot[df_plot["repo_type"] == rt]
            sizes = np.clip(
                sub.get(f"frozen_{metric}_std", 0.2).fillna(0.2) * 300, 30, 200
            )
            ax.scatter(sub[frozen_col], sub[ft_col],
                       c=_get_color_for_repo(rt), label=_get_display_name(rt),
                       s=sizes, edgecolors="white", linewidth=0.5,
                       zorder=3, alpha=0.85)

        lims = [-0.02, 1.05]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.35, label="No change")
        ax.set_xlim(lims), ax.set_ylim(lims)
        metric_label = "Solve Rate" if metric == "solve_rate" else "Avg Return"
        ax.set_xlabel(f"Frozen {metric_label}")
        ax.set_ylabel(f"Finetuned {metric_label}")
        ax.set_title(f"Effect of Per-Env Finetuning on {metric_label}")
        ax.legend(loc="lower right", fontsize=9, markerscale=0.7)
        ax.set_aspect("equal")

# Per-environment distribution plot
def plot_per_env_distributions(df: pd.DataFrame, plots_dir: Path) -> None:
    """Violin plot of per-environment solve rates, one violin per run.

    Wide violins with high median = consistent; narrow with long tails = fragile.
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

    with _figure_saver(plots_dir, "per_env_distributions.svg",
                       figsize=(max(8, len(order) * 0.7), 5)) as (fig, ax):
        sns.violinplot(data=plot_df, x="run_label", y="solve_rate",
                       hue="run_label", order=order, palette=palette,
                       ax=ax, legend=False, inner="quartile", cut=0,
                       linewidth=0.8, density_norm="width")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Per-Environment Solve Rate")
        ax.set_title("Distribution of Solve Rates Across Evaluation Environments")
        ax.set_ylim(-0.05, 1.05)

# Eval assignment checkerboard
def plot_eval_assignment_checkerboard(
    df: pd.DataFrame,
    plots_dir: Path,
    title: str | None = None,
    filename: str = "eval_assignment_checkerboard.svg",
) -> None:
    """Checkerboard heatmap: which policy each eval env was assigned to.

    Rows = eval environments; Columns = experiment runs (grouped by repo_type).
    Cell colour = assigned policy (greedy argmax).
    """
    data = _collect_eval_assignments(df)
    if data is None:
        print("Warning: no eval assignment data — skipping checkerboard plot.")
        return

    assignment = data["assignment_matrix"]
    E, N = assignment.shape
    P = data["num_policies"]
    repo_types = np.array(data["repo_types"])
    run_labels = np.array(data["run_labels"])

    # Sort columns by repo_type, then label
    sort_idx = np.lexsort((run_labels, repo_types))
    assignment = assignment[:, sort_idx]
    repo_types = repo_types[sort_idx]
    run_labels = run_labels[sort_idx]

    unique_repos = np.unique(repo_types)
    num_repos = len(unique_repos)
    boundaries = [i for i in range(1, N) if repo_types[i] != repo_types[i - 1]]

    policy_palette = sns.color_palette("pastel", max(P, 3))[:P]
    policy_cmap = ListedColormap(policy_palette)
    policy_cmap.set_bad("#fafafa")

    col_width = max(0.28, min(0.52, 10.0 / max(N, 1)))
    fig_width = max(7, N * col_width + 2.2)
    row_height = max(0.20, min(0.42, 8.0 / max(E, 1)))
    fig_height = max(4, E * row_height + 1.3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.91, top=0.93, bottom=0.14)

    if num_repos > 1:
        _draw_column_group_backgrounds(ax, repo_types, unique_repos, N, E)

    masked = np.ma.masked_invalid(assignment)
    mesh = ax.pcolormesh(np.arange(N + 1), np.arange(E + 1), masked,
                         cmap=policy_cmap, vmin=-0.5, vmax=P - 0.5,
                         edgecolors="white", linewidth=0.5,
                         shading="flat", antialiased=True)

    for b in boundaries:
        ax.axvline(b, color="#ffffff", lw=3.5, zorder=5)
        ax.axvline(b, color="#cccccc", lw=0.5, alpha=0.6, zorder=6)

    label_fs = 8.0 if N <= 12 else (7.0 if N <= 25 else 6.0)
    ax.set_xticks(np.arange(N) + 0.5)
    ax.set_xticklabels(run_labels, rotation=50, ha="right", fontsize=label_fs)
    ax.set_yticks(np.arange(E) + 0.5)
    y_fs = 8.5 if E <= 30 else (7.0 if E <= 60 else 5.5)
    ax.set_yticklabels([f"E{i}" for i in range(E)], fontsize=y_fs)

    cbar_ax = fig.add_axes([0.925, 0.16, 0.020, 0.70])
    cbar = fig.colorbar(mesh, cax=cbar_ax, ticks=np.arange(P))
    cbar.set_label("Policy", fontsize=9, labelpad=4)
    cbar.ax.set_yticklabels([f"P{p}" for p in range(P)], fontsize=7.5)
    cbar.outline.set_visible(False)

    if num_repos > 1:
        legend_handles = [
            mpatches.Patch(facecolor=_get_color_for_repo(rt),
                           edgecolor="#dddddd", linewidth=0.5,
                           label=_get_display_name(rt))
            for rt in unique_repos
        ]
        leg = fig.legend(handles=legend_handles, loc="lower center",
                         ncol=min(num_repos, 6), fontsize=7.5, frameon=True,
                         handlelength=1.0, handleheight=0.9, columnspacing=1.0,
                         edgecolor="#e0e0e0", facecolor="white",
                         borderpad=0.4, labelspacing=0.3)
        leg.get_frame().set_linewidth(0.5)

    ax.set_ylabel("Evaluation Environment", fontsize=10, labelpad=4)
    ax.set_title(title or "Evaluation Environment → Policy Assignment (Greedy Argmax)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(0, N), ax.set_ylim(0, E)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / filename, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved eval assignment checkerboard to {plots_dir / filename}")


def plot_per_repo_checkerboards(df: pd.DataFrame, plots_dir: Path) -> None:
    """One eval assignment checkerboard per repo type (comparing its configs)."""
    for repo_type in sorted(df["repo_type"].unique()):
        sub_df = df[df["repo_type"] == repo_type]
        if sub_df.empty:
            continue
        plot_eval_assignment_checkerboard(
            sub_df, plots_dir,
            title=f"{_get_display_name(repo_type)} — Eval Assignment per Configuration",
            filename=f"eval_checkerboard_{repo_type}.svg",
        )


def plot_adjusted_rand_score_heatmap(
    df: pd.DataFrame,
    plots_dir: Path,
    metric: str = "finetune_solve_rate_mean",
    filename: str = "adjusted_rand_heatmap.svg",
) -> None:
    """Heatmap of pairwise Adjusted Rand Index between eval assignments
    of the best configuration per repo type.

    Args:
        df: DataFrame from ``collect_experiment_results``.
        plots_dir: directory to save the plot to.
        metric: which metric to use for selecting the best config
            (e.g. ``"frozen_solve_rate_mean"`` or ``"finetune_solve_rate_mean"``).
        filename: output filename.
    """
    best_df = _select_best_config_per_repo(df, metric=metric)
    # Exclude random and generalist — their assignments carry no meaningful
    # signal for similarity analysis.
    best_df = best_df[~best_df["repo_type"].isin(
        ["random_matching", "generalist_matching"]
    )]
    data = _collect_eval_assignments(best_df)
    if data is None or data["assignment_matrix"].shape[1] < 2:
        return

    assignment = data["assignment_matrix"].astype(int)
    N = assignment.shape[1]
    labels = [_get_display_name(rt) for rt in data["repo_types"]]

    ari = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            ari[i, j] = adjusted_rand_score(assignment[:, i], assignment[:, j])
            ari[j, i] = ari[i, j]

    criteria_label = "Frozen" if "frozen" in metric else "Finetune"

    with sns.plotting_context("notebook"), sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(max(6, N * 1.0), max(5, N * 0.85)))
        sns.heatmap(ari, annot=False, fmt=".3f", cmap="RdYlBu_r",
                    vmin=-0.5, vmax=1, center=0, square=True,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "Adjusted Rand Index", "shrink": 0.8},
                    ax=ax)
        ax.set_title(
            f"Similarity in Environment Grouping - Adjusted Rand Index"
        )

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_active_policies_bar(
    df: pd.DataFrame,
    plots_dir: Path,
    metric: str = "finetune_solve_rate_mean",
    filename: str = "active_policies.svg",
) -> None:
    """Bar chart: number of active (assigned) policies during evaluation
    for the best configuration per repo type.

    An "active" policy is one that gets at least one evaluation environment
    assigned via greedy argmax affinity matching.  This reveals whether
    a repo's affinity scoring leads to policy collapse (few policies
    dominate) or balanced specialization.

    The ``random_matching`` repo is excluded because its assignments are
    inherently random and carry no meaningful signal.

    Args:
        df: DataFrame from ``collect_experiment_results``.
        plots_dir: directory to save the plot to.
        metric: which metric to use for selecting the best config
            (default ``"finetune_solve_rate_mean"``).
        filename: output filename.
    """
    best_df = _select_best_config_per_repo(df, metric=metric)
    # Exclude random_matching — random assignments are meaningless here.
    best_df = best_df[best_df["repo_type"] != "random_matching"]
    if best_df.empty:
        print("Warning: no (non-random) best-config data — skipping active-policies plot.")
        return

    records: list[dict] = []
    total_policies: int | None = None

    for _, row in best_df.iterrows():
        pkl_path = Path(row["run_dir"]) / "eval_frozen_results.pkl"
        if not pkl_path.exists():
            pkl_path = Path(row["run_dir"]) / "eval_finetune_results.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            affinity, env2policy, _ = pickle.load(f)
        n_active = len(np.unique(env2policy))
        records.append({
            "repo_type": row["repo_type"],
            "repo_display": row.get("repo_display", _get_display_name(row["repo_type"])),
            "n_active_policies": n_active,
        })
        # Infer total policies from affinity shape: (P, E_eval)
        if total_policies is None:
            total_policies = affinity.shape[0]

    if not records:
        print("Warning: no eval assignment data — skipping active-policies plot.")
        return

    plot_df = pd.DataFrame(records)
    # Sort by repo_type for consistent ordering
    plot_df = plot_df.sort_values("repo_type").reset_index(drop=True)

    N = len(plot_df)
    colors = [_get_color_for_repo(rt) for rt in plot_df["repo_type"]]

    metric_label = "Finetune" if "finetune" in metric else "Frozen"

    with _figure_saver(
        plots_dir, filename, figsize=(max(6, N * 0.9), 5), dpi=150
    ) as (fig, ax):
        x = np.arange(N)
        bars = ax.bar(
            x, plot_df["n_active_policies"],
            color=colors, edgecolor="white", linewidth=0.8, width=0.55,
        )

        # Annotate each bar with the count
        for i, (bar, count) in enumerate(zip(bars, plot_df["n_active_policies"])):
            ax.annotate(
                str(count),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                xytext=(0, 4), textcoords="offset points",
            )

        # Horizontal line for total number of policies in the repository
        if total_policies is not None:
            ax.axhline(
                total_policies, color="#555555", linestyle="--",
                linewidth=1.2, alpha=0.7,
                label=f"Total policies (P={total_policies})",
            )
            ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(
            plot_df["repo_display"], rotation=25, ha="right", fontsize=10,
        )
        ax.set_ylabel("Number of Active Policies", fontsize=11)
        ax.set_title(
            f"Number of Active Policies During Evaluation",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylim(
            0,
            max(plot_df["n_active_policies"].max() + 1,
                (total_policies or 0) + 1),
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    print(f"Saved active-policies bar chart to {plots_dir / filename}")


# Public orchestrator API
def generate_comparison_plots(
    df: pd.DataFrame,
    outputs_root: Path | None,
    plots_dir: Path,
) -> None:
    """Generate all cross-experiment comparison plots and a summary CSV.

    Produces:
        - ``comparison_{repo_type}.svg`` — per repo type, comparing its configs
          (averaged across seeds).
        - ``comparison_best_frozen.svg`` — best config per repo type by frozen perf.
        - ``comparison_best_finetune.svg`` — best config per repo type by finetune perf.
        - ``frozen_vs_finetune_scatter.svg`` — all runs scatter.
        - ``per_env_distributions.svg`` — violin plot (all runs).
        - ``eval_checkerboard_{repo_type}.svg`` — assignment heatmaps.
        - ``eval_checkerboard_best_frozen.svg`` / ``_finetune.svg`` — best-config
          assignment heatmaps for each criterion.
        - ``adjusted_rand_heatmap_frozen.svg`` / ``_finetune.svg`` — ARI similarity
          between best configs for each criterion.
        - ``active_policies_frozen.svg`` / ``_finetune.svg`` — number of active
          (assigned) policies per repo type (random_matching excluded).
        - ``comparison_summary.csv`` — all metrics in one table.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Per-repo hyperparam comparison (auto-averages across seeds)
    plot_per_repo_comparison_bars(df, plots_dir)

    has_ft = ("finetune_solve_rate_mean" in df.columns
              and df["finetune_solve_rate_mean"].notna().any())

    # Reference lines for baseline / control repos.
    # Drawn as horizontal dashed lines at the bar-top value.
    _REFERENCE_REPOS: dict[str, str] = {
        "generalist_matching": "Baseline (Generalist)",
        "random_matching": "Control (Random)",
    }

    # ── Best by frozen criterion ───────────────────────────────────────
    # Bar plots use all_seeds=True → cross-seed error bars are shown.
    # Checkerboard & ARI use default all_seeds=False → one column per repo.
    best_frozen_bars = _select_best_config_per_repo(
        df, metric="frozen_solve_rate_mean", all_seeds=True
    )
    if not best_frozen_bars.empty:
        plot_comparison_bars(
            best_frozen_bars, plots_dir,
            title="Best Configuration Per Repo Type (Frozen Criterion)",
            filename="comparison_best_frozen.svg",
            reference_repos=_REFERENCE_REPOS,
        )
    best_frozen_single = _select_best_config_per_repo(
        df, metric="frozen_solve_rate_mean"
    )
    if not best_frozen_single.empty:
        plot_eval_assignment_checkerboard(
            best_frozen_single, plots_dir,
            title="Best Config Per Repo Type (Frozen) — Eval Assignment",
            filename="eval_checkerboard_best_frozen.svg",
        )
        plot_adjusted_rand_score_heatmap(
            df, plots_dir, metric="frozen_solve_rate_mean",
            filename="adjusted_rand_heatmap_frozen.svg",
        )

    # ── Best by finetune criterion ─────────────────────────────────────
    if has_ft:
        best_finetune_bars = _select_best_config_per_repo(
            df, metric="finetune_solve_rate_mean", all_seeds=True
        )
        if not best_finetune_bars.empty:
            plot_comparison_bars(
                best_finetune_bars, plots_dir,
                title="Best Configuration Per Repo Type (Finetune Criterion)",
                filename="comparison_best_finetune.svg",
                reference_repos=_REFERENCE_REPOS,
            )
            # Clean finetune-only version (no frozen bars)
            plot_comparison_bars(
                best_finetune_bars, plots_dir,
                title="Performance of Starting Point Proposed by Repository (After Finetuning)",
                filename="comparison_best_finetune_only.svg",
                finetune_only=True,
                reference_repos=_REFERENCE_REPOS,
            )
        best_finetune_single = _select_best_config_per_repo(
            df, metric="finetune_solve_rate_mean"
        )
        if not best_finetune_single.empty:
            plot_eval_assignment_checkerboard(
                best_finetune_single, plots_dir,
                title="Best Config Per Repo Type (Finetune) - Eval Assignment",
                filename="eval_checkerboard_best_finetune.svg",
            )
            plot_adjusted_rand_score_heatmap(
                df, plots_dir, metric="finetune_solve_rate_mean",
                filename="adjusted_rand_heatmap_finetune.svg",
            )

    # ── All-runs scatter (every seed shown as one point) ───────────────
    if has_ft:
        plot_frozen_vs_finetune_scatter(df, plots_dir, metric="solve_rate")
        plot_frozen_vs_finetune_scatter(df, plots_dir, metric="avg_return")

    # ── Per-environment distributions ─────────────────────────────────
    if outputs_root is not None:
        plot_per_env_distributions(df, plots_dir)

    # ── Per-repo checkerboards (all configs, all seeds shown) ─────────
    plot_per_repo_checkerboards(df, plots_dir)

    # ── Active policies bar chart (random_matching excluded) ──────────
    plot_active_policies_bar(df, plots_dir, metric="frozen_solve_rate_mean",
                             filename="active_policies_frozen.svg")
    if has_ft:
        plot_active_policies_bar(df, plots_dir, metric="finetune_solve_rate_mean",
                                 filename="active_policies_finetune.svg")

    # Summary CSV
    metric_cols = [
        "repo_type", "repo_display",
        "frozen_solve_rate_mean", "frozen_solve_rate_std",
        "finetune_solve_rate_mean", "finetune_solve_rate_std",
        "frozen_avg_return_mean", "frozen_avg_return_std",
        "finetune_avg_return_mean", "finetune_avg_return_std",
    ]
    extra = [c for c in df.columns if c not in metric_cols and c not in ("run_dir",)]
    out_cols = metric_cols + ["run_dir"] + extra
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(plots_dir / "comparison_summary.csv", index=False)

    print(f"Saved comparison plots and summary to {plots_dir}")