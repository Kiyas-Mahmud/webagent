#!/usr/bin/env python3
"""Generate the publication figures for the completed Task 1 assessment.

The script reads only the frozen Task 1 aggregate CSVs.  It intentionally does
not recompute model predictions or access labels beyond the exported metrics.
Each figure is written as both a high-resolution PNG and a vector PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "task1_qwen25_dual_v1"

ROW_ORDER = [
    "browser_use_base",
    "agent_s2_base",
    "webvoyager_base",
    "browser_use_trained",
    "agent_s2_trained",
    "webvoyager_trained",
    "qwen25_heads",
]
ROW_LABELS = {
    "browser_use_base": "Browser Use + base",
    "agent_s2_base": "Agent S2 + base",
    "webvoyager_base": "WebVoyager + base",
    "browser_use_trained": "Browser Use + trained",
    "agent_s2_trained": "Agent S2 + trained",
    "webvoyager_trained": "WebVoyager + trained",
    "qwen25_heads": "Our Qwen2.5 heads",
}
COLORS = {
    "base": "#5B8DB8",
    "trained": "#D9822B",
    "heads": "#197A50",
}


def load(name: str) -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / name)
    frame["row_id"] = pd.Categorical(frame["row_id"], ROW_ORDER, ordered=True)
    return frame.sort_values("row_id").reset_index(drop=True)


def color_for(row_id: str) -> str:
    if row_id == "qwen25_heads":
        return COLORS["heads"]
    return COLORS["base" if row_id.endswith("_base") else "trained"]


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_axis(ax: plt.Axes, title: str, xmin: float, xmax: float) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(xmin, xmax)
    ax.grid(axis="x", color="#D9DEE5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#87909A")
    ax.spines["bottom"].set_color("#87909A")
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.tick_params(axis="x", labelsize=9)


def core_metrics() -> None:
    interaction = load("interaction_metrics.csv").set_index("row_id").loc[ROW_ORDER]
    recovery = load("recovery_metrics.csv").set_index("row_id").loc[ROW_ORDER]
    # The paper-facing metric panels contain only rows with at least one
    # reportable metric. Rows with zero valid structured outputs remain visible
    # in the coverage and response-validity figures below.
    reportable_rows = [
        row_id
        for row_id in ROW_ORDER
        if not interaction.loc[row_id, ["mcc", "balanced_accuracy", "macro_f1"]].isna().all()
        or not recovery.loc[row_id, ["mcc", "balanced_accuracy", "macro_f1"]].isna().all()
    ]
    interaction = interaction.loc[reportable_rows]
    recovery = recovery.loc[reportable_rows]
    specs = [
        ("mcc", "MCC", -0.35, 0.90),
        ("balanced_accuracy", "Balanced accuracy", 0.0, 1.0),
        ("macro_f1", "Macro-F1", 0.0, 1.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharey="row")
    fig.suptitle("Task 1 assessment performance", fontsize=16, fontweight="bold", y=0.995)
    for row_idx, (phase_name, frame) in enumerate(
        (("Interaction (240 cases)", interaction), ("Recovery (120 cases)", recovery))
    ):
        y = np.arange(len(reportable_rows))
        for col_idx, (column, title, xmin, xmax) in enumerate(specs):
            ax = axes[row_idx, col_idx]
            vals = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            bars = ax.barh(
                y,
                np.nan_to_num(vals, nan=0.0),
                height=0.62,
                color=[color_for(r) for r in reportable_rows],
                alpha=0.92,
            )
            for bar, val, row_id in zip(bars, vals, reportable_rows):
                if row_id == "qwen25_heads" and not np.isnan(val):
                    offset = 0.018 if val >= 0 else -0.018
                    ha = "left" if val >= 0 else "right"
                    ax.text(
                        val + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}",
                        va="center",
                        ha=ha,
                        fontsize=9,
                        fontweight="bold",
                        color=COLORS["heads"],
                    )
            format_axis(ax, title, xmin, xmax)
            if col_idx == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([ROW_LABELS[r] for r in reportable_rows])
            ax.invert_yaxis()
            if row_idx == 1:
                ax.set_xlabel("Score", fontsize=9)
            if col_idx == 0:
                # The phase label lives in the left margin so it cannot collide
                # with the first subplot title or the y-axis labels.
                fig.text(
                    0.018,
                    0.72 if row_idx == 0 else 0.30,
                    phase_name,
                    fontsize=10,
                    color="#4B5560",
                    va="center",
                    rotation=90,
                )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["base"], label="External prompt + base"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["trained"], label="External prompt + trained decoder"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["heads"], label="Our trained heads"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, -0.012),
        fontsize=9,
    )
    fig.subplots_adjust(left=0.28, right=0.985, top=0.92, bottom=0.10, wspace=0.28, hspace=0.36)
    save(fig, "task1_core_metrics")


def coverage_and_accuracy() -> None:
    interaction = load("interaction_metrics.csv").set_index("row_id").loc[ROW_ORDER]
    recovery = load("recovery_metrics.csv").set_index("row_id").loc[ROW_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.4), sharey=True)
    fig.suptitle("Task 1 output coverage and accuracy", fontsize=16, fontweight="bold", y=0.995)
    y = np.arange(len(ROW_ORDER))
    phase_frames = [("Interaction", interaction, 240), ("Recovery", recovery, 120)]

    ax = axes[0]
    for offset, (phase, frame, denominator) in zip((-0.18, 0.18), phase_frames):
        vals = frame["valid_outputs"].to_numpy(dtype=float) / denominator * 100
        ax.barh(y + offset, vals, height=0.30, label=phase, color="#4C78A8" if offset < 0 else "#F58518")
    ax.set_title("Valid output coverage", loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Valid outputs (%)")
    format_axis(ax, "", 0, 105)

    ax = axes[1]
    for offset, (phase, frame, _) in zip((-0.18, 0.18), phase_frames):
        vals = frame["accuracy"].to_numpy(dtype=float) * 100
        ax.barh(y + offset, vals, height=0.30, label=phase, color="#4C78A8" if offset < 0 else "#F58518")
    for offset, (phase, frame, _) in zip((-0.18, 0.18), phase_frames):
        vals = frame["all_case_accuracy"].to_numpy(dtype=float) * 100
        ax.plot(vals, y + offset, "o", color="#253746", markersize=4, label=f"{phase} all-case" if offset < 0 else None)
    ax.set_title("Valid vs all-case accuracy", loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)")
    format_axis(ax, "", 0, 105)

    ax = axes[2]
    vals = interaction["failure_category_macro_f1"].to_numpy(dtype=float)
    bars = ax.barh(y, np.nan_to_num(vals, nan=0.0), height=0.62, color=[color_for(r) for r in ROW_ORDER])
    ax.set_title("Failure-category Macro-F1", loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("Macro-F1")
    format_axis(ax, "", 0, 0.62)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([ROW_LABELS[r] for r in ROW_ORDER])
    for ax in axes:
        ax.invert_yaxis()
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#4C78A8", label="Interaction"),
            plt.Rectangle((0, 0), 1, 1, color="#F58518", label="Recovery"),
            Line2D([0], [0], marker="o", color="#253746", linestyle="None", markersize=5, label="All-case accuracy"),
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.965),
        fontsize=9,
    )
    fig.subplots_adjust(left=0.27, right=0.985, top=0.84, bottom=0.12, wspace=0.30)
    save(fig, "task1_coverage_accuracy")


def response_validity() -> None:
    interaction = load("interaction_metrics.csv").set_index("row_id").loc[ROW_ORDER]
    recovery = load("recovery_metrics.csv").set_index("row_id").loc[ROW_ORDER]
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.6), sharex=True)
    fig.suptitle("Task 1 response validity", fontsize=16, fontweight="bold", y=0.995)
    categories = ["valid_outputs", "parse_error", "abstention"]
    category_labels = ["Valid", "Parse error", "Abstention"]
    category_colors = ["#197A50", "#C44E52", "#E5A33D"]
    for ax, (phase, frame) in zip(axes, (("Interaction (240 cases)", interaction), ("Recovery (120 cases)", recovery))):
        y = np.arange(len(ROW_ORDER))
        left = np.zeros(len(ROW_ORDER))
        for category, label, color in zip(categories, category_labels, category_colors):
            vals = frame[category].to_numpy(dtype=float)
            ax.barh(y, vals, left=left, height=0.62, color=color, label=label)
            left += vals
        ax.set_title(phase, loc="left", fontsize=12, fontweight="bold", pad=8)
        ax.set_yticks(y)
        ax.set_yticklabels([ROW_LABELS[r] for r in ROW_ORDER])
        ax.invert_yaxis()
        ax.set_xlim(0, max(frame["eligible_cases"].max(), 1))
        ax.grid(axis="x", color="#D9DEE5", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#87909A")
        ax.spines["bottom"].set_color("#87909A")
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylabel("System")
    axes[-1].set_xlabel("Number of cases")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=9)
    fig.subplots_adjust(left=0.28, right=0.985, top=0.88, bottom=0.10, hspace=0.36)
    save(fig, "task1_response_validity")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    core_metrics()
    coverage_and_accuracy()
    response_validity()
    print("Generated six Task 1 figure files in", OUT)


if __name__ == "__main__":
    main()
