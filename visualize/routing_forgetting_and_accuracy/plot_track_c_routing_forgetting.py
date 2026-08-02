from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


# Nature-style typography and editable vector text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.5
plt.rcParams["axes.labelsize"] = 8.5
plt.rcParams["axes.titlesize"] = 8.5
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["legend.fontsize"] = 7.5
plt.rcParams["legend.frameon"] = False


SCENARIOS = ["RNT", "R", "N", "T"]
COLORS = {
    "RNT": "#A6CEE3",  # RGB(166, 206, 227)
    "R": "#B2DF8A",    # RGB(178, 223, 138)
    "N": "#FB9A99",    # RGB(251, 154, 153)
    "T": "#FDC26F",    # RGB(253, 194, 111)
}
TASK_LABELS = {
    "rgbnt201": "RGBNT201",
    "market_mm": "Market-MM",
    "rgbnt100": "RGBNT100",
    "msvr310": "MSVR310",
    "wmveid863": "WMVeID863",
}
NEUTRAL = "#6F6F6F"
EDGE = "#4D4D4D"
SINGLE_FIG_WIDTH_IN = 89 / 25.4
SINGLE_FIG_HEIGHT_IN = 70 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Track C routing-accuracy and forgetting figures."
    )
    parser.add_argument(
        "--stage-metrics",
        type=Path,
        default=None,
        help="Raw stage_metrics.csv. Optional after source-data CSVs are exported.",
    )
    parser.add_argument(
        "--continual-summary",
        type=Path,
        default=None,
        help="Raw continual_summary.csv. Optional after source-data CSVs are exported.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for figures and source-data tables.",
    )
    return parser.parse_args()


def validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def prepare_source_data(
    stage_metrics_path: Path,
    continual_summary_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_metrics = pd.read_csv(stage_metrics_path)
    continual_summary = pd.read_csv(continual_summary_path)

    validate_columns(
        stage_metrics,
        {"stage", "trained_task", "eval_task", "scenario", "routing_accuracy"},
        "stage_metrics.csv",
    )
    validate_columns(
        continual_summary,
        {
            "stage",
            "trained_task",
            "scenario",
            "forgetting_mAP",
            "forgetting_R1",
            "forgetting_R5",
            "forgetting_R10",
        },
        "continual_summary.csv",
    )

    unknown_scenarios = set(stage_metrics["scenario"]).difference(SCENARIOS)
    if unknown_scenarios:
        raise ValueError(f"Unexpected routing scenarios: {sorted(unknown_scenarios)}")

    stage_order = (
        stage_metrics[["stage", "trained_task"]]
        .drop_duplicates()
        .sort_values("stage")
        .reset_index(drop=True)
    )
    expected_stages = list(range(1, len(stage_order) + 1))
    if stage_order["stage"].tolist() != expected_stages:
        raise ValueError("Training stages must be consecutive and start at 1.")

    # Macro-average over all evaluation tasks seen at a given training stage.
    routing_mean = (
        stage_metrics.groupby(["stage", "scenario"], as_index=False)["routing_accuracy"]
        .mean()
        .rename(columns={"routing_accuracy": "mean_routing_accuracy"})
        .merge(stage_order, on="stage", how="left")
    )
    routing_mean["chance_accuracy"] = 100.0 / routing_mean["stage"]
    routing_mean["scenario"] = pd.Categorical(
        routing_mean["scenario"], categories=SCENARIOS, ordered=True
    )
    routing_mean = routing_mean.sort_values(["stage", "scenario"]).reset_index(drop=True)

    final_stage = int(stage_order["stage"].max())
    final_routing = stage_metrics.loc[
        stage_metrics["stage"].eq(final_stage),
        ["stage", "trained_task", "eval_task", "scenario", "routing_accuracy"],
    ].copy()
    final_routing["scenario"] = pd.Categorical(
        final_routing["scenario"], categories=SCENARIOS, ordered=True
    )
    final_routing["eval_task"] = pd.Categorical(
        final_routing["eval_task"],
        categories=stage_order["trained_task"].tolist(),
        ordered=True,
    )
    final_routing = final_routing.sort_values(["eval_task", "scenario"]).reset_index(drop=True)

    forgetting_all = continual_summary[
        [
            "stage",
            "trained_task",
            "scenario",
            "forgetting_mAP",
            "forgetting_R1",
            "forgetting_R5",
            "forgetting_R10",
        ]
    ].copy()
    forgetting_all["scenario"] = pd.Categorical(
        forgetting_all["scenario"], categories=SCENARIOS, ordered=True
    )
    forgetting_all = forgetting_all.sort_values(["stage", "scenario"]).reset_index(drop=True)
    forgetting_plotted = forgetting_all.loc[forgetting_all["stage"].gt(1)].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    routing_mean.to_csv(output_dir / "source_data_routing_mean.csv", index=False)
    final_routing.to_csv(output_dir / "source_data_routing_final_stage.csv", index=False)
    forgetting_all.to_csv(output_dir / "source_data_forgetting_all_metrics.csv", index=False)
    stage_order.to_csv(output_dir / "source_data_stage_order.csv", index=False)
    return routing_mean, final_routing, forgetting_plotted, stage_order


def load_exported_source_data(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    routing_mean = pd.read_csv(output_dir / "source_data_routing_mean.csv")
    final_routing = pd.read_csv(output_dir / "source_data_routing_final_stage.csv")
    forgetting_all = pd.read_csv(output_dir / "source_data_forgetting_all_metrics.csv")
    stage_order = pd.read_csv(output_dir / "source_data_stage_order.csv")
    forgetting_plotted = forgetting_all.loc[forgetting_all["stage"].gt(1)].copy()
    return routing_mean, final_routing, forgetting_plotted, stage_order


def stage_tick_labels(stage_order: pd.DataFrame, start_stage: int = 1) -> list[str]:
    labels: list[str] = []
    for row in stage_order.loc[stage_order["stage"].ge(start_stage)].itertuples():
        task = TASK_LABELS.get(row.trained_task, row.trained_task)
        labels.append(f"{int(row.stage)}\n{task}")
    return labels


def plot_scenario_lines(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_column: str,
    stages: np.ndarray,
) -> None:
    for scenario in SCENARIOS:
        values = (
            data.loc[data["scenario"].eq(scenario)]
            .sort_values("stage")[y_column]
            .to_numpy(dtype=float)
        )
        if len(values) != len(stages):
            raise ValueError(f"Incomplete {y_column} series for {scenario}.")
        ax.plot(
            stages,
            values,
            color=COLORS[scenario],
            linewidth=1.9,
            marker="o",
            markersize=4.6,
            markeredgecolor=EDGE,
            markeredgewidth=0.55,
            label=scenario,
            zorder=3,
        )


def blend_with_white(color: str, strength: float) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color))
    strength = float(np.clip(strength, 0.0, 1.0))
    return tuple(1.0 - strength * (1.0 - rgb))


def draw_final_stage_heatmap(ax: plt.Axes, final_routing: pd.DataFrame) -> None:
    task_order = final_routing["eval_task"].drop_duplicates().tolist()
    matrix = (
        final_routing.pivot(index="eval_task", columns="scenario", values="routing_accuracy")
        .reindex(index=task_order, columns=SCENARIOS)
        .to_numpy(dtype=float)
    )
    if np.isnan(matrix).any():
        raise ValueError("Final-stage routing heatmap contains missing values.")

    rows, cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            value = matrix[row, col]
            strength = 0.10 + 0.90 * value / 100.0
            facecolor = blend_with_white(COLORS[SCENARIOS[col]], strength)
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="white",
                    linewidth=1.2,
                )
            )
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6.8,
                color="#222222",
            )

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels(SCENARIOS)
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels([TASK_LABELS.get(task, task) for task in task_order])
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def add_scenario_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        handlelength=1.7,
        columnspacing=1.0,
        handletextpad=0.45,
    )


def create_routing_figures(
    routing_mean: pd.DataFrame,
    final_routing: pd.DataFrame,
    stage_order: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax_line = plt.subplots(
        figsize=(SINGLE_FIG_WIDTH_IN, SINGLE_FIG_HEIGHT_IN)
    )
    stages = stage_order["stage"].to_numpy(dtype=int)

    plot_scenario_lines(ax_line, routing_mean, "mean_routing_accuracy", stages)
    chance = 100.0 / stages
    ax_line.plot(
        stages,
        chance,
        color=NEUTRAL,
        linewidth=1.25,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    ax_line.text(
        stages[-1] - 0.02,
        chance[-1] + 3.2,
        "Chance",
        ha="right",
        va="bottom",
        fontsize=6.6,
        color=NEUTRAL,
    )
    ax_line.set_xticks(stages)
    ax_line.set_xticklabels(stage_tick_labels(stage_order))
    ax_line.set_xlim(stages.min() - 0.15, stages.max() + 0.15)
    ax_line.set_ylim(0, 104)
    ax_line.set_yticks(np.arange(0, 101, 20))
    ax_line.set_xlabel("Training stage")
    ax_line.set_ylabel("Mean routing accuracy (%)")
    ax_line.set_title("Routing across stages", pad=7)
    ax_line.tick_params(axis="x", pad=3, labelsize=6.4)
    add_scenario_legend(fig, ax_line)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.23, top=0.73)
    save_figure(fig, output_dir / "track_c_routing_accuracy_by_stage")

    fig, ax_heat = plt.subplots(
        figsize=(SINGLE_FIG_WIDTH_IN, SINGLE_FIG_HEIGHT_IN)
    )
    draw_final_stage_heatmap(ax_heat, final_routing)
    ax_heat.set_xlabel("Query setting", labelpad=6)
    ax_heat.set_title("Final-stage routing accuracy (%)", pad=7)
    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.18, top=0.88)
    save_figure(fig, output_dir / "track_c_routing_accuracy_final_stage")


def create_forgetting_figures(
    forgetting: pd.DataFrame,
    stage_order: pd.DataFrame,
    output_dir: Path,
) -> None:
    stages = np.sort(forgetting["stage"].unique()).astype(int)
    tick_labels = stage_tick_labels(stage_order, start_stage=2)

    figures = [
        ("forgetting_mAP", "mAP forgetting", "track_c_forgetting_map"),
        ("forgetting_R1", "R1 forgetting", "track_c_forgetting_r1"),
    ]
    for column, title, filename in figures:
        fig, ax = plt.subplots(
            figsize=(SINGLE_FIG_WIDTH_IN, SINGLE_FIG_HEIGHT_IN)
        )
        plot_scenario_lines(ax, forgetting, column, stages)
        ax.axhline(0, color=NEUTRAL, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
        ax.set_xticks(stages)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(stages.min() - 0.14, stages.max() + 0.14)
        ax.set_ylim(-10, 20)
        ax.set_yticks(np.arange(-10, 21, 5))
        ax.set_xlabel("Training stage")
        ax.set_ylabel("Forgetting (percentage points)")
        ax.set_title(title, pad=7)
        ax.tick_params(axis="x", pad=3, labelsize=6.5)
        add_scenario_legend(fig, ax)
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.23, top=0.73)
        save_figure(fig, output_dir / filename)


def write_captions(output_dir: Path) -> None:
    captions = (
        "Routing accuracy across stages\n"
        "Fig. X | Routing accuracy across sequential training on Track C. "
        "The figure shows macro-averaged routing accuracy across all tasks observed at each training "
        "stage for RNT, R, N and T query settings. The grey dashed curve denotes "
        "chance-level routing accuracy (100/k for k candidate task adapters). Stage 1 "
        "contains a single candidate route and is therefore trivially 100%. Results are "
        "single-run point estimates. Source data are provided with the figure.\n\n"
        "Final-stage routing accuracy\n"
        "Fig. X | Dataset-specific routing accuracy at the final training stage on Track C. "
        "Rows denote the five evaluation datasets, and columns denote the RNT, R, N and T "
        "query settings. Cell values are percentages, and darker shading within each column indicates "
        "higher routing accuracy. Results are single-run point estimates. Source data are provided "
        "with the figure.\n\n"
        "mAP forgetting\n"
        "Fig. X | mAP forgetting under automatic routing on Track C. "
        "The figure shows average mAP forgetting after training stages 2-5 for RNT, R, N and T "
        "query settings. Forgetting is computed over previously learned tasks as each task's "
        "best earlier mAP minus its current mAP, with the newly learned task excluded. Positive "
        "values indicate forgetting, whereas negative values indicate backward transfer or performance "
        "recovery. Because evaluation uses automatic routing, the reported values include both "
        "representation-related and routing-induced effects. Results are single-run point estimates. "
        "Source data are provided with the figure.\n\n"
        "R1 forgetting\n"
        "Fig. X | R1 forgetting under automatic routing on Track C. "
        "The figure shows average R1 forgetting after training stages 2-5 for RNT, R, N and T "
        "query settings. Forgetting is computed over previously learned tasks as each task's "
        "best earlier R1 minus its current R1, with the newly learned task excluded. Positive "
        "values indicate forgetting, whereas negative values indicate backward transfer or performance "
        "recovery. Because evaluation uses automatic routing, the reported values include both "
        "representation-related and routing-induced effects. Results are single-run point estimates. "
        "Source data are provided with the figure.\n"
    )
    (output_dir / "captions_en.txt").write_text(captions, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if (args.stage_metrics is None) != (args.continual_summary is None):
        raise ValueError(
            "Provide both --stage-metrics and --continual-summary, or provide neither."
        )

    if args.stage_metrics is not None:
        routing_mean, final_routing, forgetting, stage_order = prepare_source_data(
            args.stage_metrics.resolve(),
            args.continual_summary.resolve(),
            output_dir,
        )
    else:
        routing_mean, final_routing, forgetting, stage_order = load_exported_source_data(
            output_dir
        )

    create_routing_figures(routing_mean, final_routing, stage_order, output_dir)
    create_forgetting_figures(forgetting, stage_order, output_dir)
    write_captions(output_dir)
    print(f"Figures and source data written to: {output_dir}")


if __name__ == "__main__":
    main()
