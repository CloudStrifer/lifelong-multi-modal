from __future__ import annotations

import csv
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# Nature-style typography and editable vector text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["ytick.major.size"] = 3
plt.rcParams["savefig.facecolor"] = "white"


ROOT = Path(__file__).resolve().parent
SOURCE_DATA = ROOT / "source_data_track_c.csv"
DATASETS = ("RGBNT201", "Market-MM", "RGBNT100", "MSVR310", "WMVeID863")
DATASET_LABELS = ("RGBNT201", "Market-MM", "RGBNT100", "MSVR310", "WMVeID863")
QUERIES = ("RNT", "R", "N", "T")
SINGLE_QUERIES = ("R", "N", "T")
METRICS = ("mAP", "R1", "R5", "R10")
COLORS = {
    "RNT": "#A6CEE3",  # RGB(166, 206, 227)
    "R": "#B2DF8A",    # RGB(178, 223, 138)
    "N": "#FB9A99",    # RGB(251, 154, 153)
    "T": "#FDC26F",    # RGB(253, 194, 111)
}


def load_data() -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {dataset: {} for dataset in DATASETS}
    with SOURCE_DATA.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            data[row["dataset"]][row["query"]] = {
                metric: float(row[metric]) for metric in METRICS
            }
    for dataset in DATASETS:
        if set(data[dataset]) != set(QUERIES):
            raise ValueError(f"Incomplete query-modality data for {dataset}.")
    return data


def averages(data: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        query: {
            metric: fmean(data[dataset][query][metric] for dataset in DATASETS)
            for metric in METRICS
        }
        for query in QUERIES
    }


def changes_from_rnt(
    data: dict[str, dict[str, dict[str, float]]], metric: str
) -> dict[str, list[float]]:
    return {
        query: [
            data[dataset][query][metric] - data[dataset]["RNT"][metric]
            for dataset in DATASETS
        ]
        for query in SINGLE_QUERIES
    }


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#272727")
    ax.spines["bottom"].set_color("#272727")
    ax.tick_params(axis="both", labelsize=7.5, color="#272727")


def plot_average_panel(ax: plt.Axes, mean_values: dict[str, dict[str, float]]) -> None:
    x = np.arange(len(METRICS))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(QUERIES))
    for query, offset in zip(QUERIES, offsets):
        values = [mean_values[query][metric] for metric in METRICS]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[query],
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
    ax.set_title("Average performance", fontsize=9.5, pad=7)
    ax.set_ylabel("Performance (%)", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.58, len(METRICS) - 0.42)
    style_axis(ax)


def plot_delta_panel(
    ax: plt.Axes,
    changes: dict[str, list[float]],
    metric: str,
) -> None:
    x = np.arange(len(DATASETS))
    width = 0.25
    offsets = (-width, 0.0, width)
    for query, offset in zip(SINGLE_QUERIES, offsets):
        values = changes[query]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[query],
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
        for bar, value in zip(bars, values):
            if value > 0 or abs(value) < 1.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (1.4 if value >= 0 else -1.8),
                    f"{value:+.1f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=5.5,
                    color="#272727",
                )
    ax.axhline(0, color="#767676", linewidth=0.8, linestyle=(0, (2.2, 2.2)), zorder=1)
    ax.set_title(f"{metric} change vs RNT", fontsize=9.5, pad=7)
    ax.set_ylabel("Performance change (pp)", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(-65, 10)
    ax.set_yticks([-60, -40, -20, 0])
    ax.set_xlim(-0.62, len(DATASETS) - 0.38)
    style_axis(ax)


def save_figure(fig: plt.Figure, filename: str) -> list[Path]:
    base = ROOT / filename
    outputs: list[Path] = []
    for extension, dpi in (("svg", 600), ("pdf", 600), ("png", 600), ("tiff", 600)):
        path = base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def make_average_figure(mean_values: dict[str, dict[str, float]]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(89 / 25.4, 66 / 25.4))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.78)
    plot_average_panel(ax, mean_values)
    legend_handles = [Patch(facecolor=COLORS[q], edgecolor="none", label=q) for q in QUERIES]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.975),
        ncol=4,
        columnspacing=1.1,
        handlelength=1.2,
        handleheight=0.75,
        fontsize=7.5,
    )
    return save_figure(fig, "track_c_average_performance")


def make_delta_figure(changes: dict[str, list[float]], metric: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(89 / 25.4, 66 / 25.4))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.27, top=0.78)
    plot_delta_panel(ax, changes, metric)
    legend_handles = [Patch(facecolor=COLORS[q], edgecolor="none", label=q) for q in SINGLE_QUERIES]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.975),
        ncol=3,
        columnspacing=1.4,
        handlelength=1.2,
        handleheight=0.75,
        fontsize=7.5,
    )
    return save_figure(fig, f"track_c_{metric.lower()}_change_vs_rnt")


def make_figures(data: dict[str, dict[str, dict[str, float]]]) -> list[Path]:
    mean_values = averages(data)
    outputs = make_average_figure(mean_values)
    outputs.extend(make_delta_figure(changes_from_rnt(data, "mAP"), "mAP"))
    outputs.extend(make_delta_figure(changes_from_rnt(data, "R1"), "R1"))

    print("Five-task averages:")
    for query in QUERIES:
        print(query, ", ".join(f"{metric}={mean_values[query][metric]:.2f}" for metric in METRICS))
    return outputs


def main() -> None:
    data = load_data()
    outputs = make_figures(data)
    for output in outputs:
        print(output.name)


if __name__ == "__main__":
    main()
