"""Shared utilities for analysis plotting modules.

Internal helpers used by both ``plot_runs`` and ``plot_comparison``
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_COLORS: dict[str, str] = {
    "random_matching":          "#999999",
    "morphology_matching":      "#E69F00",
    "dino_matching":            "#56B4E9",
    "policy_latent_matching":   "#009E73",
    "performance_matching":     "#CC79A7",
    "generalist_matching":      "#882255",
}

REPO_DISPLAY_NAMES: dict[str, str] = {
    "random_matching":          "Random",
    "morphology_matching":      "Morphology",
    "dino_matching":            "DINO",
    "policy_latent_matching":   "Policy Latent",
    "performance_matching":     "Performance",
    "generalist_matching":      "Generalist",
}

_IDENTITY_COLS = {"run_dir", "repo_type", "repo_display", "seed"}
_METRIC_PREFIXES = ("frozen_", "finetune_")

def _get_color_for_repo(repo_type: str) -> str:
    """Hex colour for a repo_type, with a dark fallback."""
    return REPO_COLORS.get(repo_type, "#555555")

def _get_display_name(repo_type: str) -> str:
    """Human-readable short name for a repo_type."""
    return REPO_DISPLAY_NAMES.get(repo_type, repo_type)

def _parse_repo_type_from_name(dirname: str) -> str | None:
    """Extract repo_type from a timestamped directory name.

    Names follow the pattern ``<repo_type>_YYYY-MM-DD_HH-MM-SS``.
    The last two underscore-delimited segments are date and time; everything
    before them is the repo_type (which may itself contain underscores, e.g.
    ``policy_latent_matching``).
    """
    parts = dirname.split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[:-2])

# Hyperparameter discovery
def _get_varying_hyperparam_cols(df: pd.DataFrame) -> list[str]:
    """Return columns that represent varying hyperparameters.

    Excludes identity columns (run_dir, repo_type, repo_display) and
    metric columns (frozen_*, finetune_*).  Returns only columns with
    more than one unique non-NaN value — constant columns provide no
    grouping or labelling signal.
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

def _build_run_label(
    row,
    varying_cols: list[str] | None = None,
    max_dir_chars: int = 18,
) -> str:
    """Compact label for a single run: display name + key params + dir tail.

    Args:
        row: dict-like with keys ``run_dir``, ``repo_display``, ``repo_type``,
            and optional hyperparameter columns.
        varying_cols: hyperparam columns to include in the suffix.  If None,
            auto-detected as all non-identity, non-metric columns with non-NaN
            values in *row*.
        max_dir_chars: maximum characters to keep from the end of the
            directory name.
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

    param_parts = [f"{k}={row[k]}" for k in varying_cols
                   if k in row and pd.notna(row[k])]
    if param_parts:
        return f"{display} ({', '.join(param_parts)})\n{tail}"
    return f"{display}\n{tail}"


# Axes helpers
def _thin_ticks(ax, steps, max_ticks: int = 15):
    """Limit x-axis tick labels to at most *max_ticks* evenly-spaced positions."""
    T = len(steps)
    if T <= max_ticks:
        return
    step = max(1, T // max_ticks)
    ax.set_xticks(steps[::step])
    ax.set_xticklabels(steps[::step])

def _desaturate(hex_color: str, amount: float = 0.4) -> str:
    """Lighten/desaturate a hex colour by blending with white.

    Args:
        hex_color: hex string, e.g. ``"#56B4E9"``.
        amount: blend factor; 0 = original, 1 = white.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"

@contextmanager
def _figure_saver(plots_dir: Path, filename: str, figsize=(10, 5), dpi=150,
                  format: str | None = None):
    """Context manager that creates a figure, then saves and closes it.

    If *format* is provided, the file extension is replaced (e.g. ``"svg"``).

    Usage::

        with _figure_saver(plots_dir, "my_plot.png", figsize=(8, 4)) as (fig, ax):
            ax.plot(...)
            ax.set_xlabel("Step")
    """
    if format is not None:
        filename = str(Path(filename).with_suffix(f".{format}"))
    fig, ax = plt.subplots(figsize=figsize)
    try:
        yield fig, ax
    finally:
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

# Generic box-plot-over-time factory
def _plot_boxplot_over_time(
    values: np.ndarray,       # (T, E_subset)
    plots_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    facecolor: str = "lightblue",
    overlay_label: str = "Mean",
    median_color: str = "darkblue",
    overlay_color: str = "darkred",
    overlay_marker_size: int = 4,
    reference_line: float | None = None,
    reference_label: str | None = None,
) -> None:
    """Box plot with one box per step, plus mean overlay and optional reference line.

    Each box summarises the distribution of *values* across the E_subset
    candidate environments for that step.

    Args:
        values: (T, E_subset) array.
        plots_dir: directory to save to.
        filename: output filename (e.g. ``"score_spread_cv.png"``).
        title: plot title.
        ylabel: y-axis label.
        ylim: (ymin, ymax) tuple.
        facecolor: box fill colour.
        overlay_label: legend label for the mean-overlay line.
        median_color, overlay_color: line colours.
        overlay_marker_size: marker size for the mean overlay.
        reference_line: if not None, draw a horizontal dashed line at this y-value.
        reference_label: legend label for the reference line.
    """
    T = values.shape[0]
    steps = np.arange(T)

    with _figure_saver(plots_dir, filename, figsize=(10, 5)) as (fig, ax):
        box_data = [values[t] for t in range(T)]
        ax.boxplot(
            box_data, positions=steps, widths=0.6,
            patch_artist=True, showfliers=True,
            boxprops={"facecolor": facecolor, "alpha": 0.6},
            medianprops={"color": median_color, "lw": 1.5},
            whiskerprops={"color": "gray"},
            capprops={"color": "gray"},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4,
                        "markeredgewidth": 0},
        )

        # Mean overlay — compute only for steps that have at least one valid
        # value, avoiding "Mean of empty slice" warnings on all-NaN rows
        # (which occur when diagnostics are logged only periodically).
        mean_vals = np.full(T, np.nan)
        row_has_data = np.any(~np.isnan(values), axis=1)
        if row_has_data.any():
            mean_vals[row_has_data] = np.nanmean(values[row_has_data], axis=1)
        valid = ~np.isnan(mean_vals)
        ax.plot(steps[valid], mean_vals[valid], "o-", color=overlay_color, lw=2,
                markersize=overlay_marker_size, label=overlay_label)

        if reference_line is not None:
            ax.axhline(reference_line, color="gray", ls="--", lw=1,
                       label=reference_label)

        ax.set_xlabel("Repository Training Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.legend(loc="lower right")
        _thin_ticks(ax, steps)
