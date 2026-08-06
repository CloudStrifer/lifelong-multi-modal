#!/usr/bin/env python3
"""Configurable R/N/T x ID t-SNE visualization.

The script can either generate synthetic high-dimensional multimodal features or
load real features from an NPZ file. Color represents identity; marker shape
represents modality.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Keep text editable in SVG/PDF exports.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["legend.frameon"] = False


REFERENCE_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#56B4E9",  # sky blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#000000",  # black
]
DEFAULT_MARKERS = {"R": "o", "N": "^", "T": "s"}
EXTRA_MARKERS = ["D", "P", "X", "v", "<", ">", "*"]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def natural_key(value: object) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def unit_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, np.finfo(float).eps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw a configurable t-SNE plot: ID=color, modality=marker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("synthetic high-dimensional data")
    data.add_argument("--num-ids", type=positive_int, default=8, help="number of identities")
    data.add_argument("--modalities", nargs="+", default=["R", "N", "T"], help="modality names")
    data.add_argument("--points-per-group", type=positive_int, default=8, help="samples per ID x modality")
    data.add_argument("--feature-dim", type=positive_int, default=64, help="feature vector dimension")
    data.add_argument(
        "--id-separation",
        type=positive_float,
        default=4.0,
        help="distance scale between identity centers; larger usually separates IDs",
    )
    data.add_argument(
        "--modality-gap",
        type=nonnegative_float,
        default=1.5,
        help="R/N/T offset within each ID; larger separates modalities",
    )
    data.add_argument(
        "--cluster-std",
        type=nonnegative_float,
        default=10.0,
        help="high-dimensional within-group noise; smaller gives genuinely tighter inputs",
    )
    data.add_argument("--data-seed", type=int, default=42, help="seed for centers, offsets and samples")
    data.add_argument(
        "--input-npz",
        type=Path,
        default=None,
        help="optional NPZ with features (N,D), ids (N,), modalities (N,); overrides synthetic settings",
    )

    tsne = parser.add_argument_group("t-SNE")
    tsne.add_argument("--tsne-seed", type=int, default=42, help="seed for random t-SNE initialization")
    tsne.add_argument("--perplexity", type=positive_float, default=25.0, help="local-neighborhood scale")
    tsne.add_argument("--early-exaggeration", type=positive_float, default=12.0)
    tsne.add_argument(
        "--learning-rate",
        default="auto",
        help="'auto' or a positive number",
    )
    tsne.add_argument("--max-iter", type=positive_int, default=1200, help="optimization iterations")
    tsne.add_argument(
        "--pca-dim",
        type=nonnegative_int,
        default=30,
        help="PCA dimensions before t-SNE; 0 disables PCA",
    )

    display = parser.add_argument_group("display")
    display.add_argument(
        "--display-scale",
        type=nonnegative_float,
        default=1.0,
        help="post-t-SNE within-group scale: <1 tighter, >1 looser, 1 untouched",
    )
    display.add_argument("--point-size", type=positive_float, default=55.0)
    display.add_argument("--alpha", type=float, default=0.88)
    display.add_argument("--title", default="", help="optional figure title")
    display.add_argument("--figure-width", type=positive_float, default=11.5, help="inches")
    display.add_argument("--figure-height", type=positive_float, default=9.0, help="inches")
    display.add_argument("--dpi", type=positive_int, default=300)
    display.add_argument(
        "--output",
        type=Path,
        default=Path("tsne_rnt"),
        help="output prefix without extension",
    )
    display.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "svg", "pdf"],
        default=["png", "svg", "pdf"],
    )
    display.add_argument(
        "--save-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save source features, coordinates and metadata",
    )
    return parser


def generate_synthetic_features(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.data_seed)
    n_ids = args.num_ids
    modalities = list(dict.fromkeys(args.modalities))
    if not modalities:
        raise ValueError("At least one modality is required.")

    id_centers = unit_rows(rng.normal(size=(n_ids, args.feature_dim))) * args.id_separation
    mode_directions = unit_rows(rng.normal(size=(n_ids * len(modalities), args.feature_dim)))
    mode_offsets = mode_directions.reshape(n_ids, len(modalities), args.feature_dim) * args.modality_gap

    features: list[np.ndarray] = []
    ids: list[str] = []
    modes: list[str] = []
    noise_scale = args.cluster_std / np.sqrt(args.feature_dim)

    for id_index in range(n_ids):
        for mode_index, modality in enumerate(modalities):
            center = id_centers[id_index] + mode_offsets[id_index, mode_index]
            noise = rng.normal(
                loc=0.0,
                scale=noise_scale,
                size=(args.points_per_group, args.feature_dim),
            )
            features.append(center + noise)
            ids.extend([f"ID {id_index + 1}"] * args.points_per_group)
            modes.extend([str(modality)] * args.points_per_group)

    return np.vstack(features), np.asarray(ids), np.asarray(modes)


def load_npz_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        missing = {"features", "ids", "modalities"} - set(bundle.files)
        if missing:
            raise ValueError(f"NPZ is missing arrays: {', '.join(sorted(missing))}")
        features = np.asarray(bundle["features"], dtype=float)
        ids = np.asarray(bundle["ids"]).astype(str)
        modes = np.asarray(bundle["modalities"]).astype(str)

    if features.ndim != 2:
        raise ValueError("features must have shape (n_samples, n_features)")
    if ids.ndim != 1 or modes.ndim != 1:
        raise ValueError("ids and modalities must be one-dimensional")
    if len(features) != len(ids) or len(features) != len(modes):
        raise ValueError("features, ids and modalities must contain the same number of samples")
    if not np.isfinite(features).all():
        raise ValueError("features contains NaN or infinite values")
    return features, ids, modes


def parse_learning_rate(value: str) -> str | float:
    if str(value).lower() == "auto":
        return "auto"
    parsed = float(value)
    if parsed <= 0:
        raise ValueError("--learning-rate must be 'auto' or > 0")
    return parsed


def compute_tsne(features: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    n_samples, n_features = features.shape
    if n_samples < 3:
        raise ValueError("t-SNE needs at least 3 samples")

    pca_requested = int(args.pca_dim)
    pca_used = min(pca_requested, n_features, n_samples - 1)
    reduced = features
    if 0 < pca_used < n_features:
        reduced = PCA(n_components=pca_used, svd_solver="full").fit_transform(features)
    else:
        pca_used = 0

    perplexity_used = min(float(args.perplexity), float(n_samples - 1))
    if perplexity_used != float(args.perplexity):
        print(
            f"[warning] perplexity was capped from {args.perplexity:g} "
            f"to {perplexity_used:g} because n_samples={n_samples}."
        )

    kwargs: dict[str, object] = {
        "n_components": 2,
        "perplexity": perplexity_used,
        "early_exaggeration": args.early_exaggeration,
        "learning_rate": parse_learning_rate(args.learning_rate),
        "init": "random",
        "random_state": args.tsne_seed,
        "method": "barnes_hut",
        "angle": 0.5,
    }
    # scikit-learn renamed n_iter to max_iter in v1.5.
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = args.max_iter
    else:
        kwargs["n_iter"] = args.max_iter

    embedding = TSNE(**kwargs).fit_transform(reduced)
    metadata = {
        "n_samples": int(n_samples),
        "input_feature_dim": int(n_features),
        "pca_dim_used": int(pca_used),
        "perplexity_requested": float(args.perplexity),
        "perplexity_used": float(perplexity_used),
        "scikit_learn_version": sklearn_version,
    }
    return embedding, metadata


def scale_within_groups(
    embedding: np.ndarray,
    ids: np.ndarray,
    modes: np.ndarray,
    scale: float,
) -> np.ndarray:
    if scale == 1.0:
        return embedding.copy()
    adjusted = embedding.copy()
    for identity in np.unique(ids):
        for modality in np.unique(modes):
            mask = (ids == identity) & (modes == modality)
            if not np.any(mask):
                continue
            centroid = embedding[mask].mean(axis=0, keepdims=True)
            adjusted[mask] = centroid + scale * (embedding[mask] - centroid)
    return adjusted


def make_color_map(identity_labels: list[str]) -> dict[str, object]:
    n = len(identity_labels)
    if n <= len(REFERENCE_COLORS):
        colors: list[object] = REFERENCE_COLORS[:n]
    elif n <= 20:
        colors = list(matplotlib.colormaps["tab20"].colors[:n])
    else:
        colors = list(matplotlib.colormaps["turbo"](np.linspace(0.04, 0.96, n)))
    return dict(zip(identity_labels, colors))


def make_marker_map(modality_labels: list[str]) -> dict[str, str]:
    markers: dict[str, str] = {}
    extras = iter(EXTRA_MARKERS)
    for modality in modality_labels:
        if modality in DEFAULT_MARKERS:
            markers[modality] = DEFAULT_MARKERS[modality]
        else:
            markers[modality] = next(extras, "o")
    return markers


def reorder_handles_row_major(handles: list[Line2D], n_columns: int) -> list[Line2D]:
    """Compensate for Matplotlib's column-major multi-row legend layout."""
    if len(handles) <= n_columns:
        return handles
    n_rows = int(np.ceil(len(handles) / n_columns))
    full_columns = len(handles) - n_columns * (n_rows - 1)
    ordered: list[Line2D] = []
    for column in range(n_columns):
        column_length = n_rows if column < full_columns else n_rows - 1
        for row in range(column_length):
            index = row * n_columns + column
            ordered.append(handles[index])
    return ordered


def plot_embedding(
    embedding: np.ndarray,
    ids: np.ndarray,
    modes: np.ndarray,
    args: argparse.Namespace,
) -> tuple[plt.Figure, dict[str, object], dict[str, str]]:
    identity_labels = sorted(np.unique(ids).tolist(), key=natural_key)
    modality_labels = sorted(
        np.unique(modes).tolist(),
        key=lambda value: (args.modalities.index(value) if value in args.modalities else len(args.modalities), value),
    )
    color_map = make_color_map(identity_labels)
    marker_map = make_marker_map(modality_labels)

    fig, ax = plt.subplots(figsize=(args.figure_width, args.figure_height))
    for identity in identity_labels:
        for modality in modality_labels:
            mask = (ids == identity) & (modes == modality)
            if not np.any(mask):
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=args.point_size,
                c=[color_map[identity]],
                marker=marker_map[modality],
                alpha=args.alpha,
                edgecolors="white",
                linewidths=0.75,
                rasterized=False,
            )

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if args.title:
        ax.set_title(args.title, fontsize=15, pad=10)

    id_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map[label],
            markeredgecolor="none",
            markersize=10,
            label=label,
        )
        for label in identity_labels
    ]
    modality_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[label],
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="none",
            markersize=10,
            label=label,
        )
        for label in modality_labels
    ]

    id_columns = min(8, len(identity_labels))
    id_handles = reorder_handles_row_major(id_handles, id_columns)
    fig.legend(
        handles=id_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.082),
        ncol=id_columns,
        fontsize=11.5,
        handletextpad=0.45,
        columnspacing=1.45,
    )
    fig.legend(
        handles=modality_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=len(modality_handles),
        fontsize=11.5,
        handletextpad=0.45,
        columnspacing=2.0,
    )
    id_rows = int(np.ceil(len(identity_labels) / id_columns))
    bottom = 0.18 + 0.045 * id_rows
    fig.subplots_adjust(left=0.025, right=0.975, top=0.97, bottom=bottom)
    return fig, color_map, marker_map


def normalized_output_prefix(output: Path) -> Path:
    if output.suffix.lower() in {".png", ".svg", ".pdf"}:
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_figure(fig: plt.Figure, prefix: Path, formats: list[str], dpi: int) -> list[Path]:
    saved: list[Path] = []
    for file_format in formats:
        path = prefix.with_suffix(f".{file_format}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        saved.append(path)
    plt.close(fig)
    return saved


def save_source_data(
    prefix: Path,
    features: np.ndarray,
    ids: np.ndarray,
    modes: np.ndarray,
    raw_embedding: np.ndarray,
    display_embedding: np.ndarray,
    metadata: dict[str, object],
    args: argparse.Namespace,
) -> list[Path]:
    npz_path = prefix.parent / f"{prefix.name}_source.npz"
    csv_path = prefix.parent / f"{prefix.name}_coordinates.csv"
    json_path = prefix.parent / f"{prefix.name}_metadata.json"

    np.savez_compressed(npz_path, features=features, ids=ids, modalities=modes)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_index", "id", "modality", "tsne_x_raw", "tsne_y_raw", "x_display", "y_display"]
        )
        for index, (identity, modality, raw, shown) in enumerate(
            zip(ids, modes, raw_embedding, display_embedding)
        ):
            writer.writerow([index, identity, modality, *raw.tolist(), *shown.tolist()])

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    payload = {
        "arguments": serializable_args,
        "run": metadata,
        "integrity_note": (
            "display_scale changes only within-(ID, modality) scatter after t-SNE; "
            "raw coordinates are preserved in the CSV."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return [npz_path, csv_path, json_path]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0 < args.alpha <= 1:
        parser.error("--alpha must be in (0, 1]")

    if args.input_npz is None:
        features, ids, modes = generate_synthetic_features(args)
        source = "synthetic"
    else:
        features, ids, modes = load_npz_features(args.input_npz)
        source = str(args.input_npz)

    raw_embedding, metadata = compute_tsne(features, args)
    shown_embedding = scale_within_groups(raw_embedding, ids, modes, args.display_scale)
    metadata["source"] = source
    metadata["display_scale"] = float(args.display_scale)
    metadata["identity_count"] = int(len(np.unique(ids)))
    metadata["modality_count"] = int(len(np.unique(modes)))

    if args.display_scale != 1.0:
        print(
            "[integrity note] --display-scale is a visual post-processing step. "
            "Use 1.0 for an untouched t-SNE embedding."
        )

    fig, _, _ = plot_embedding(shown_embedding, ids, modes, args)
    prefix = normalized_output_prefix(args.output)
    saved = save_figure(fig, prefix, args.formats, args.dpi)
    if args.save_data:
        saved.extend(
            save_source_data(prefix, features, ids, modes, raw_embedding, shown_embedding, metadata, args)
        )

    print(f"Generated {len(features)} samples: {metadata['identity_count']} IDs x {metadata['modality_count']} modalities")
    print("Saved:")
    for path in saved:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
