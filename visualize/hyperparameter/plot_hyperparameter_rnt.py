from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
SOURCE_DATA = ROOT / "source_data_rnt.csv"

METRICS = ("mAP", "R1", "R5", "R10")
COLORS = {
    "mAP": "#0F4D92",
    "R1": "#42949E",
    "R5": "#9A4D8E",
    "R10": "#B64342",
}
MARKERS = {"mAP": "o", "R1": "s", "R5": "^", "R10": "D"}

FIGURE_SPECS = {
    "adapter_rank": {
        "filename": "adapter_rank_rnt",
        "xlabel": "Adapter rank, r",
        "scale": "log2",
        "xlim": (13.5, 176.0),
        "label_x": 141.0,
    },
    "keys_per_modality": {
        "filename": "keys_per_modality_rnt",
        "xlabel": "Keys per modality, K",
        "scale": "log2",
        "xlim": (0.84, 10.8),
        "label_x": 8.75,
    },
    "lse_temperature": {
        "filename": "lse_temperature_rnt",
        "xlabel": "LSE temperature, τ",
        "scale": "linear",
        "xlim": (0.014, 0.116),
        "label_x": 0.1035,
    },
}


def load_data() -> dict[str, list[dict[str, float | str | bool]]]:
    grouped: dict[str, list[dict[str, float | str | bool]]] = {}
    with SOURCE_DATA.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | str | bool] = {
                "x_label": raw["x_label"],
                "x_value": float(raw["x_value"]),
                "is_default": raw["is_default"] == "1",
            }
            for metric in METRICS:
                row[metric] = float(raw[metric])
            grouped.setdefault(raw["experiment"], []).append(row)

    for rows in grouped.values():
        rows.sort(key=lambda item: float(item["x_value"]))
    return grouped


def validate_data(data: dict[str, list[dict[str, float | str | bool]]]) -> None:
    if set(data) != set(FIGURE_SPECS):
        raise ValueError("The source data do not match the expected three experiments.")
    controls = []
    for name, rows in data.items():
        if len(rows) != 4:
            raise ValueError(f"{name} must contain exactly four hyperparameter settings.")
        default_rows = [row for row in rows if bool(row["is_default"])]
        if len(default_rows) != 1:
            raise ValueError(f"{name} must contain exactly one default setting.")
        controls.append(tuple(round(float(default_rows[0][m]), 6) for m in METRICS))
    if len(set(controls)) != 1:
        raise ValueError("Default-control metrics are inconsistent across experiments.")


def make_figure(name: str, rows: list[dict[str, float | str | bool]]) -> list[Path]:
    spec = FIGURE_SPECS[name]
    x = [float(row["x_value"]) for row in rows]
    xlabels = [str(row["x_label"]) for row in rows]
    default_x = next(float(row["x_value"]) for row in rows if bool(row["is_default"]))

    # 89 x 66 mm: compact single-column manuscript figure.
    fig, ax = plt.subplots(figsize=(89 / 25.4, 66 / 25.4))
    fig.subplots_adjust(left=0.175, right=0.965, bottom=0.205, top=0.955)

    ax.axvline(
        default_x,
        color="#B8B8B8",
        linewidth=0.8,
        linestyle=(0, (2.2, 2.2)),
        zorder=0,
    )
    ax.text(
        default_x,
        89.0,
        "default",
        ha="center",
        va="top",
        color="#767676",
        fontsize=6.5,
    )

    for metric in METRICS:
        y = [float(row[metric]) for row in rows]
        ax.plot(
            x,
            y,
            color=COLORS[metric],
            marker=MARKERS[metric],
            markersize=4.1,
            markeredgewidth=0.6,
            linewidth=1.35,
            solid_capstyle="round",
            zorder=2,
        )
        ax.text(
            float(spec["label_x"]),
            y[-1],
            metric,
            color=COLORS[metric],
            fontsize=7.2,
            fontweight="bold",
            ha="left",
            va="center",
        )

    if spec["scale"] == "log2":
        ax.set_xscale("log", base=2)
    ax.set_xlim(*spec["xlim"])
    ax.set_ylim(54.0, 89.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_yticks([55, 65, 75, 85])
    ax.set_xlabel(str(spec["xlabel"]), fontsize=8.5, labelpad=5)
    ax.set_ylabel("RNT performance (%)", fontsize=8.5, labelpad=5)
    ax.tick_params(axis="both", which="major", labelsize=7.5, color="#272727")
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.spines["left"].set_color("#272727")
    ax.spines["bottom"].set_color("#272727")

    base = ROOT / str(spec["filename"])
    outputs: list[Path] = []
    for extension, dpi in (("svg", 600), ("pdf", 600), ("png", 600), ("tiff", 600)):
        path = base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    data = load_data()
    validate_data(data)
    outputs: list[Path] = []
    for name in ("adapter_rank", "keys_per_modality", "lse_temperature"):
        outputs.extend(make_figure(name, data[name]))
    for path in outputs:
        print(path.name)


if __name__ == "__main__":
    main()
