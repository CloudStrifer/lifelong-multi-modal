"""Extract and visualize lifelong multi-modal ReID features with t-SNE.

The default paper figure is a 2 x 3 grid:

    RGBNT201 (person):  TMDA-only | CC-MTKR-only | Full model
    RGBNT100 (vehicle): TMDA-only | CC-MTKR-only | Full model

Each checkpoint is evaluated with automatic routing.  The same identities and
the same synchronized R/N/T records are reused across all three variants.
Identity is encoded by color and modality by marker shape.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg as default_cfg  # noqa: E402
from data.lifelong_datasets import (  # noqa: E402
    MODALITIES,
    ReIDRecord,
    SynchronizedTriModalTransform,
    TriModalImageDataset,
    canonical_dataset_name,
    dataset_object_category,
    dataset_target_size,
    lifelong_collate,
    load_protocol,
)
from modeling.lifelong_model import LifelongMDReID  # noqa: E402


LOGGER = logging.getLogger("MDReID.tsne")
VARIANTS = (
    ("tmda_only", "TMDA-only", "tmda_checkpoint"),
    ("cc_mtkr_only", "CC-MTKR-only", "cc_mtkr_checkpoint"),
    ("full_model", "Full model", "full_checkpoint"),
)
DEFAULT_DATASETS = ("RGBNT201", "RGBNT100")
DATASET_LABELS = {
    "RGBNT201": "RGBNT201\n(Person)",
    "RGBNT100": "RGBNT100\n(Vehicle)",
}
MODALITY_MARKERS = {"R": "o", "N": "^", "T": "s"}
# Okabe-Ito-derived categorical colors.  Eight identities are the default so
# that every color remains distinguishable under common color-vision deficits.
IDENTITY_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
)
CHECKPOINT_PATTERN = re.compile(r"stage_(\d+)(?:_|\.)", re.IGNORECASE)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


@dataclass(frozen=True)
class SelectedRecord:
    record: ReIDRecord
    identity_label: int
    sample_index: int

    @property
    def pid(self) -> int:
        return int(self.record[1])

    @property
    def camid(self) -> int:
        return int(self.record[2])

    @property
    def sceneid(self) -> int:
        return int(self.record[3])


@dataclass
class FeaturePanel:
    dataset: str
    variant_key: str
    variant_label: str
    checkpoint: str
    stage: int
    features: np.ndarray
    pids: np.ndarray
    identity_labels: np.ndarray
    modalities: np.ndarray
    sample_indices: np.ndarray
    paths: np.ndarray
    selected_tasks: np.ndarray
    route_correct: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2x3 t-SNE comparison for TMDA and CC-MTKR ablations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("all", "extract", "plot"),
        default="all",
        help="Extract+plot, extract only, or redraw cached features only.",
    )
    parser.add_argument(
        "--config_file",
        default=str(REPO_ROOT / "configs/lifelong/MDReID_TMDA_CSCR.yml"),
        help="Fallback config; the config embedded in each checkpoint is authoritative.",
    )
    parser.add_argument("--tmda_checkpoint", type=Path)
    parser.add_argument("--cc_mtkr_checkpoint", type=Path)
    parser.add_argument("--full_checkpoint", type=Path)
    parser.add_argument(
        "--stage",
        type=int,
        default=5,
        help="Stage selected when a checkpoint directory is supplied.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names; the paper layout expects two datasets.",
    )
    parser.add_argument("--num_ids", type=int, default=8)
    parser.add_argument("--samples_per_id", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1555)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. cuda, cuda:0, or cpu. 'cuda' uses the visible GPU 0.",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iterations", type=int, default=1500)
    parser.add_argument(
        "--embedding_mode",
        choices=("joint_dataset", "independent"),
        default="joint_dataset",
        help=(
            "Use one shared embedding per dataset for valid cross-model comparison, "
            "or reproduce the original independent panel embeddings."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--allow_variant_mismatch",
        action="store_true",
        help="Allow labels that disagree with checkpoint TMDA/router flags.",
    )
    parser.add_argument("--hide_metrics", action="store_true")
    parser.add_argument("--no_individual_panels", action="store_true")
    parser.add_argument(
        "--no_connect_modalities",
        action="store_true",
        help="Do not connect synchronized R/N/T points from the same record.",
    )
    parser.add_argument(
        "--hide_route_status",
        action="store_true",
        help="Draw routing errors as filled points instead of hollow points.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Runtime YACS overrides, e.g. DATASETS.ROOT_DIR ./dataset.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_datasets(value: str) -> Tuple[str, ...]:
    datasets = tuple(
        canonical_dataset_name(item)
        for item in value.split(",")
        if item.strip()
    )
    if len(datasets) != 2:
        raise ValueError(
            "The paper grid requires exactly two datasets; got {}.".format(datasets)
        )
    if len(set(datasets)) != len(datasets):
        raise ValueError("Dataset names must be unique.")
    return datasets


def resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Use --device cpu only for a "
                "small debugging run."
            )
        if normalized == "cuda":
            normalized = "cuda:0"
    return torch.device(normalized)


def checkpoint_stage(path: Path) -> Optional[int]:
    match = CHECKPOINT_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def resolve_checkpoint(path: Path, requested_stage: int) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(str(resolved))

    candidates = sorted(
        {
            candidate.resolve()
            for candidate in resolved.rglob("stage_*.pth")
            if candidate.is_file() and checkpoint_stage(candidate) is not None
        }
    )
    stage_candidates = [
        candidate
        for candidate in candidates
        if checkpoint_stage(candidate) == requested_stage
    ]
    if not stage_candidates:
        available = sorted(
            {stage for stage in map(checkpoint_stage, candidates) if stage is not None}
        )
        raise FileNotFoundError(
            "No stage_{:02d} checkpoint below {}. Available stages: {}".format(
                requested_stage, resolved, available or "none"
            )
        )
    if len(stage_candidates) > 1:
        raise RuntimeError(
            "More than one stage_{:02d} checkpoint was found below {}:\n{}\n"
            "Pass the exact .pth file or a narrower experiment directory.".format(
                requested_stage,
                resolved,
                "\n".join("  - {}".format(item) for item in stage_candidates),
            )
        )
    return stage_candidates[0]


def torch_load_checkpoint(path: Path) -> MutableMapping[str, object]:
    LOGGER.info("Loading checkpoint payload: %s", path)
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    required = {"stage", "task_specs", "state_dict"}
    missing = required.difference(payload)
    if missing:
        raise KeyError("Checkpoint {} is missing keys: {}".format(path, sorted(missing)))
    return payload


def build_checkpoint_cfg(
    payload: Mapping[str, object], args: argparse.Namespace
) -> CN:
    cfg = default_cfg.clone()
    config_file = Path(args.config_file).expanduser().resolve()
    if config_file.is_file():
        cfg.merge_from_file(str(config_file))
    elif "config" not in payload:
        raise FileNotFoundError(str(config_file))

    embedded = payload.get("config")
    if embedded:
        embedded_cfg = CN.load_cfg(str(embedded))
        # YACS decodes every replacement string with ``literal_eval`` during a
        # merge.  Thus the dumped string ``DEVICE_ID: '0'`` would become the
        # integer 0 and conflict with the legacy default's string type.  An
        # extra representation layer preserves the intended runtime-neutral
        # string; feature extraction uses --device rather than this field.
        if hasattr(embedded_cfg.MODEL, "DEVICE_ID"):
            embedded_cfg.MODEL.DEVICE_ID = repr(
                str(embedded_cfg.MODEL.DEVICE_ID).strip("'\"")
            )
        cfg.merge_from_other_cfg(embedded_cfg)
    if args.opts:
        cfg.merge_from_list(list(args.opts))
    cfg.TEST.IMS_PER_BATCH = int(args.batch_size)
    if args.num_workers is not None:
        cfg.DATALOADER.NUM_WORKERS = int(args.num_workers)
    cfg.freeze()
    return cfg


def variant_audit(variant_key: str, cfg: CN) -> Dict[str, object]:
    return {
        "variant": variant_key,
        "modality_decoupled": bool(cfg.LIFELONG.MODALITY_DECOUPLED),
        "router_method": str(cfg.LIFELONG.ROUTER.METHOD).lower(),
        "category_aware": bool(cfg.LIFELONG.ROUTER.CATEGORY_AWARE),
        "task_key_calibration": bool(
            cfg.LIFELONG.ROUTER.TASK_KEY_CALIBRATION
        ),
        "task_key_feature_initialization": bool(
            cfg.LIFELONG.ROUTER.TASK_KEY_FEATURE_INITIALIZATION
        ),
        "adapter_rank": int(cfg.LIFELONG.ADAPTER_RANK),
    }


def validate_variant(variant_key: str, cfg: CN) -> Dict[str, object]:
    audit = variant_audit(variant_key, cfg)
    decoupled = bool(audit["modality_decoupled"])
    router_method = str(audit["router_method"])
    task_key = router_method == "task_key"
    calibrated = bool(audit["task_key_calibration"])
    category_aware = bool(audit["category_aware"])

    if variant_key == "tmda_only":
        valid = decoupled and router_method == "legacy"
        expectation = "MODALITY_DECOUPLED=True and ROUTER.METHOD=legacy"
    elif variant_key == "cc_mtkr_only":
        valid = (not decoupled) and task_key and calibrated and category_aware
        expectation = (
            "MODALITY_DECOUPLED=False, ROUTER.METHOD=task_key, "
            "CATEGORY_AWARE=True, and TASK_KEY_CALIBRATION=True"
        )
    elif variant_key == "full_model":
        valid = decoupled and task_key and calibrated and category_aware
        expectation = (
            "MODALITY_DECOUPLED=True, ROUTER.METHOD=task_key, "
            "CATEGORY_AWARE=True, and TASK_KEY_CALIBRATION=True"
        )
    else:
        raise KeyError(variant_key)

    audit["valid"] = valid
    audit["expectation"] = expectation
    return audit


def build_model(
    cfg: CN,
    payload: Mapping[str, object],
    device: torch.device,
) -> LifelongMDReID:
    model = LifelongMDReID(cfg)
    for task_spec in payload["task_specs"]:
        dataset = canonical_dataset_name(str(task_spec["dataset"]))
        model.register_task(
            task_key=str(task_spec["task_key"]),
            num_classes=int(task_spec["num_classes"]),
            object_category=str(
                task_spec.get("object_category", dataset_object_category(dataset))
            ),
            initialize_from_history=False,
        )
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The checkpoint cannot be reconstructed from its saved config. "
            "Check MODEL.PRETRAIN_PATH_T and do not override architecture flags."
        ) from error
    model.to(device)
    model.eval()
    return model


def record_key(record: ReIDRecord) -> str:
    paths = record[0]
    path_key = paths if isinstance(paths, str) else "||".join(paths)
    return "{}||{}||{}||{}".format(path_key, record[1], record[2], record[3])


def modality_path(record: ReIDRecord, modality: str) -> str:
    paths = record[0]
    if isinstance(paths, str):
        return "{}#{}".format(paths, modality)
    return str(paths[MODALITIES.index(modality)])


def unique_eval_records(protocol) -> List[ReIDRecord]:
    records: Dict[str, ReIDRecord] = {}
    for record in list(protocol.query) + list(protocol.gallery):
        records.setdefault(record_key(record), record)
    return [records[key] for key in sorted(records)]


def select_records(
    dataset: str,
    records: Sequence[ReIDRecord],
    num_ids: int,
    samples_per_id: int,
    seed: int,
) -> List[SelectedRecord]:
    by_pid: Dict[int, List[ReIDRecord]] = {}
    for record in records:
        by_pid.setdefault(int(record[1]), []).append(record)
    eligible = sorted(
        pid for pid, items in by_pid.items() if len(items) >= samples_per_id
    )
    if len(eligible) < num_ids:
        largest = sorted(
            ((pid, len(items)) for pid, items in by_pid.items()),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        raise ValueError(
            "{} has only {} identities with at least {} unique test records; "
            "{} were requested. Largest identity counts: {}".format(
                dataset, len(eligible), samples_per_id, num_ids, largest
            )
        )

    dataset_seed = seed + int(zlib.crc32(dataset.encode("utf-8")))
    rng = np.random.default_rng(dataset_seed)
    chosen_pids = sorted(
        int(value)
        for value in rng.choice(eligible, size=num_ids, replace=False).tolist()
    )
    selected: List[SelectedRecord] = []
    sample_index = 0
    for identity_label, pid in enumerate(chosen_pids):
        pid_records = sorted(by_pid[pid], key=record_key)
        indices = sorted(
            int(value)
            for value in rng.choice(
                len(pid_records), size=samples_per_id, replace=False
            ).tolist()
        )
        for index in indices:
            selected.append(
                SelectedRecord(
                    record=pid_records[index],
                    identity_label=identity_label,
                    sample_index=sample_index,
                )
            )
            sample_index += 1
    return selected


def write_sample_manifest(
    output_dir: Path,
    selected_by_dataset: Mapping[str, Sequence[SelectedRecord]],
) -> Path:
    path = output_dir / "source_data_samples.csv"
    fieldnames = (
        "dataset",
        "identity_label",
        "pid",
        "sample_index",
        "camid",
        "sceneid",
        "R_path",
        "N_path",
        "T_path",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, selections in selected_by_dataset.items():
            for selected in selections:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "identity_label": selected.identity_label + 1,
                        "pid": selected.pid,
                        "sample_index": selected.sample_index,
                        "camid": selected.camid,
                        "sceneid": selected.sceneid,
                        "R_path": modality_path(selected.record, "R"),
                        "N_path": modality_path(selected.record, "N"),
                        "T_path": modality_path(selected.record, "T"),
                    }
                )
    return path


def find_task_spec(
    payload: Mapping[str, object], dataset: str
) -> Mapping[str, object]:
    matches = [
        spec
        for spec in payload["task_specs"]
        if canonical_dataset_name(str(spec["dataset"])) == dataset
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Checkpoint contains {} task specs for {} (expected one).".format(
                len(matches), dataset
            )
        )
    return matches[0]


@torch.inference_mode()
def extract_panel(
    model: LifelongMDReID,
    cfg: CN,
    payload: Mapping[str, object],
    checkpoint_path: Path,
    dataset: str,
    selections: Sequence[SelectedRecord],
    variant_key: str,
    variant_label: str,
    device: torch.device,
    amp: bool,
) -> FeaturePanel:
    transform = SynchronizedTriModalTransform(
        size=dataset_target_size(dataset),
        training=False,
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD,
    )
    dataset_object = TriModalImageDataset(
        [selected.record for selected in selections], transform
    )
    loader = DataLoader(
        dataset_object,
        batch_size=int(cfg.TEST.IMS_PER_BATCH),
        shuffle=False,
        num_workers=int(cfg.DATALOADER.NUM_WORKERS),
        collate_fn=lifelong_collate,
        pin_memory=device.type == "cuda",
    )

    task_spec = find_task_spec(payload, dataset)
    target_task = str(task_spec["task_key"])
    object_category = str(
        task_spec.get("object_category", dataset_object_category(dataset))
    )
    candidates = [str(spec["task_key"]) for spec in payload["task_specs"]]
    target_index = candidates.index(target_task)

    feature_chunks: Dict[str, List[torch.Tensor]] = {
        modality: [] for modality in MODALITIES
    }
    route_chunks: Dict[str, List[torch.Tensor]] = {
        modality: [] for modality in MODALITIES
    }
    observed_pids: List[torch.Tensor] = []
    LOGGER.info(
        "Extracting %s | %s | %d triplets | auto routing",
        variant_label,
        dataset,
        len(selections),
    )
    for batch_index, (images, pids, _camids, _sceneids, _paths) in enumerate(loader):
        images = {
            modality: tensor.to(device, non_blocking=True)
            for modality, tensor in images.items()
        }
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            outputs = model.extract_scenarios(
                images=images,
                scenarios=MODALITIES,
                routing="auto",
                candidate_tasks=candidates,
                object_category=object_category,
            )
        observed_pids.append(pids.cpu())
        for modality in MODALITIES:
            descriptor, selected_routes, _scores = outputs[modality]
            feature_chunks[modality].append(
                F.normalize(descriptor.float(), dim=1).cpu()
            )
            route_chunks[modality].append(selected_routes.cpu())
        if batch_index == 0 or (batch_index + 1) % 5 == 0:
            LOGGER.info("  completed batch %d/%d", batch_index + 1, len(loader))

    expected_pids = np.asarray([selected.pid for selected in selections], dtype=np.int64)
    actual_pids = torch.cat(observed_pids).numpy().astype(np.int64)
    if not np.array_equal(expected_pids, actual_pids):
        raise RuntimeError("Dataloader order changed; cross-variant samples are no longer aligned.")

    base_identity_labels = np.asarray(
        [selected.identity_label for selected in selections], dtype=np.int64
    )
    base_sample_indices = np.asarray(
        [selected.sample_index for selected in selections], dtype=np.int64
    )
    features = []
    selected_tasks: List[str] = []
    route_correct = []
    pids = []
    identity_labels = []
    modalities = []
    sample_indices = []
    paths = []
    for modality in MODALITIES:
        modality_features = torch.cat(feature_chunks[modality]).numpy()
        modality_routes = torch.cat(route_chunks[modality]).numpy().astype(np.int64)
        features.append(modality_features)
        selected_tasks.extend(candidates[index] for index in modality_routes)
        route_correct.append(modality_routes == target_index)
        pids.append(expected_pids)
        identity_labels.append(base_identity_labels)
        modalities.extend([modality] * len(selections))
        sample_indices.append(base_sample_indices)
        paths.extend(modality_path(selected.record, modality) for selected in selections)

    panel = FeaturePanel(
        dataset=dataset,
        variant_key=variant_key,
        variant_label=variant_label,
        checkpoint=str(checkpoint_path),
        stage=int(payload["stage"]),
        features=np.concatenate(features, axis=0).astype(np.float32),
        pids=np.concatenate(pids, axis=0),
        identity_labels=np.concatenate(identity_labels, axis=0),
        modalities=np.asarray(modalities),
        sample_indices=np.concatenate(sample_indices, axis=0),
        paths=np.asarray(paths),
        selected_tasks=np.asarray(selected_tasks),
        route_correct=np.concatenate(route_correct, axis=0),
    )
    validate_panel(panel)
    return panel


def validate_panel(panel: FeaturePanel) -> None:
    count = panel.features.shape[0]
    if panel.features.ndim != 2 or count == 0:
        raise ValueError("Feature matrix must be non-empty and two-dimensional.")
    arrays = (
        panel.pids,
        panel.identity_labels,
        panel.modalities,
        panel.sample_indices,
        panel.paths,
        panel.selected_tasks,
        panel.route_correct,
    )
    if any(len(array) != count for array in arrays):
        raise ValueError("Feature metadata arrays have inconsistent lengths.")
    if not np.isfinite(panel.features).all():
        raise ValueError("Feature matrix contains NaN or infinity.")
    if set(panel.modalities.tolist()) != set(MODALITIES):
        raise ValueError("All R/N/T modalities must be present in every panel.")


def panel_path(output_dir: Path, dataset: str, variant_key: str) -> Path:
    return output_dir / "features" / "{}__{}.npz".format(
        dataset.lower().replace("-", "_"), variant_key
    )


def save_panel(path: Path, panel: FeaturePanel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        dataset=np.asarray(panel.dataset),
        variant_key=np.asarray(panel.variant_key),
        variant_label=np.asarray(panel.variant_label),
        checkpoint=np.asarray(panel.checkpoint),
        stage=np.asarray(panel.stage, dtype=np.int64),
        features=panel.features,
        pids=panel.pids,
        identity_labels=panel.identity_labels,
        modalities=panel.modalities,
        sample_indices=panel.sample_indices,
        paths=panel.paths,
        selected_tasks=panel.selected_tasks,
        route_correct=panel.route_correct,
    )


def load_panel(path: Path) -> FeaturePanel:
    if not path.is_file():
        raise FileNotFoundError(
            "Cached feature file not found: {}. Run --mode extract or --mode all first.".format(
                path
            )
        )
    with np.load(str(path), allow_pickle=False) as data:
        panel = FeaturePanel(
            dataset=str(data["dataset"].item()),
            variant_key=str(data["variant_key"].item()),
            variant_label=str(data["variant_label"].item()),
            checkpoint=str(data["checkpoint"].item()),
            stage=int(data["stage"].item()),
            features=data["features"].copy(),
            pids=data["pids"].copy(),
            identity_labels=data["identity_labels"].copy(),
            modalities=data["modalities"].copy(),
            sample_indices=data["sample_indices"].copy(),
            paths=data["paths"].copy(),
            selected_tasks=data["selected_tasks"].copy(),
            route_correct=data["route_correct"].copy(),
        )
    validate_panel(panel)
    return panel


def extract_all(
    args: argparse.Namespace,
    datasets: Sequence[str],
    output_dir: Path,
) -> None:
    checkpoint_arguments = {
        "tmda_checkpoint": args.tmda_checkpoint,
        "cc_mtkr_checkpoint": args.cc_mtkr_checkpoint,
        "full_checkpoint": args.full_checkpoint,
    }
    missing = [name for name, value in checkpoint_arguments.items() if value is None]
    if missing:
        raise ValueError(
            "Checkpoint arguments required for extraction: {}".format(
                ", ".join("--{}".format(name) for name in missing)
            )
        )
    if args.num_ids < 2 or args.samples_per_id < 2:
        raise ValueError("Use at least two identities and two records per identity.")

    device = resolve_device(args.device)
    LOGGER.info("Feature extraction device: %s", device)
    selected_by_dataset: Optional[Dict[str, List[SelectedRecord]]] = None
    extraction_metadata: Dict[str, object] = {
        "routing": "auto",
        "datasets": list(datasets),
        "num_ids": int(args.num_ids),
        "samples_per_id": int(args.samples_per_id),
        "seed": int(args.seed),
        "variants": {},
    }

    for variant_key, variant_label, argument_name in VARIANTS:
        checkpoint = resolve_checkpoint(
            checkpoint_arguments[argument_name], requested_stage=int(args.stage)
        )
        payload = torch_load_checkpoint(checkpoint)
        cfg = build_checkpoint_cfg(payload, args)
        audit = validate_variant(variant_key, cfg)
        audit.update(
            {
                "checkpoint": str(checkpoint),
                "stage": int(payload["stage"]),
                "track": payload.get("track"),
                "order": payload.get("order"),
            }
        )
        extraction_metadata["variants"][variant_key] = audit
        if not audit["valid"]:
            message = (
                "{} checkpoint flags do not match its paper label. Expected {}; got {}"
            ).format(variant_label, audit["expectation"], variant_audit(variant_key, cfg))
            if args.allow_variant_mismatch:
                LOGGER.warning(message)
            else:
                raise ValueError(message + ". Use --allow_variant_mismatch only intentionally.")

        if selected_by_dataset is None:
            selected_by_dataset = {}
            for dataset in datasets:
                protocol = load_protocol(
                    dataset,
                    cfg.DATASETS.ROOT_DIR,
                    load_train=False,
                )
                selected_by_dataset[dataset] = select_records(
                    dataset=dataset,
                    records=unique_eval_records(protocol),
                    num_ids=int(args.num_ids),
                    samples_per_id=int(args.samples_per_id),
                    seed=int(args.seed),
                )
            manifest = write_sample_manifest(output_dir, selected_by_dataset)
            LOGGER.info("Fixed cross-variant sample manifest: %s", manifest)

        model = build_model(cfg, payload, device)
        for dataset in datasets:
            panel = extract_panel(
                model=model,
                cfg=cfg,
                payload=payload,
                checkpoint_path=checkpoint,
                dataset=dataset,
                selections=selected_by_dataset[dataset],
                variant_key=variant_key,
                variant_label=variant_label,
                device=device,
                amp=not args.no_amp,
            )
            destination = panel_path(output_dir, dataset, variant_key)
            save_panel(destination, panel)
            LOGGER.info("Cached features: %s", destination)

        del model
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata_path = output_dir / "extraction_metadata.json"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(extraction_metadata, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Extraction metadata: %s", metadata_path)


def require_sklearn():
    try:
        import sklearn
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires scikit-learn. Install it in the training environment "
            "with: pip install scikit-learn"
        ) from error
    return sklearn, PCA, TSNE, silhouette_score


def run_tsne(
    features: np.ndarray,
    seed: int,
    perplexity: float,
    iterations: int,
) -> Tuple[np.ndarray, float, Dict[str, object]]:
    sklearn, PCA, TSNE, _silhouette_score = require_sklearn()
    sample_count, feature_dim = features.shape
    if sample_count < 4:
        raise ValueError("t-SNE requires at least four feature vectors.")
    effective_perplexity = min(float(perplexity), (sample_count - 1) / 3.0)
    if effective_perplexity < 2:
        raise ValueError("Effective t-SNE perplexity is below 2.")
    pca_components = min(50, feature_dim, sample_count - 1)
    reduced = PCA(
        n_components=pca_components,
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(features)

    kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "metric": "euclidean",
        "random_state": seed,
        "verbose": 0,
    }
    signature = inspect.signature(TSNE.__init__)
    if "max_iter" in signature.parameters:
        kwargs["max_iter"] = int(iterations)
    else:
        kwargs["n_iter"] = int(iterations)
    estimator = TSNE(**kwargs)
    embedding = estimator.fit_transform(reduced).astype(np.float32)
    embedding -= embedding.mean(axis=0, keepdims=True)
    metadata = {
        "sklearn_version": sklearn.__version__,
        "pca_components": int(pca_components),
        "perplexity": float(effective_perplexity),
        "iterations": int(iterations),
        "seed": int(seed),
        "kl_divergence": float(estimator.kl_divergence_),
    }
    return embedding, float(estimator.kl_divergence_), metadata


def validate_cross_variant_alignment(
    dataset: str,
    ordered_panels: Sequence[FeaturePanel],
) -> None:
    if len(ordered_panels) != len(VARIANTS):
        raise ValueError("Joint t-SNE requires all three model variants.")
    reference = ordered_panels[0]
    for panel in ordered_panels[1:]:
        if panel.features.shape != reference.features.shape:
            raise ValueError(
                "{} feature shapes differ across variants: {} vs {}.".format(
                    dataset, reference.features.shape, panel.features.shape
                )
            )
        for field in (
            "pids",
            "identity_labels",
            "modalities",
            "sample_indices",
            "paths",
        ):
            if not np.array_equal(getattr(reference, field), getattr(panel, field)):
                raise ValueError(
                    "{} panels are not aligned on '{}'. Re-run feature extraction "
                    "once with all three checkpoints and the same seed.".format(
                        dataset, field
                    )
                )


def run_joint_dataset_tsne(
    dataset: str,
    panels: Mapping[Tuple[str, str], FeaturePanel],
    seed: int,
    perplexity: float,
    iterations: int,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[str, object]]:
    ordered_keys = [(dataset, variant_key) for variant_key, _label, _arg in VARIANTS]
    ordered_panels = [panels[key] for key in ordered_keys]
    validate_cross_variant_alignment(dataset, ordered_panels)
    joint_features = np.concatenate(
        [panel.features for panel in ordered_panels], axis=0
    )
    LOGGER.info(
        "Running one joint t-SNE: %s | %d variants | %d points",
        dataset,
        len(ordered_panels),
        len(joint_features),
    )
    joint_embedding, _kl, metadata = run_tsne(
        joint_features,
        seed=seed,
        perplexity=perplexity,
        iterations=iterations,
    )
    embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    offset = 0
    for key, panel in zip(ordered_keys, ordered_panels):
        next_offset = offset + len(panel.features)
        embeddings[key] = joint_embedding[offset:next_offset].copy()
        offset = next_offset
    if offset != len(joint_embedding):
        raise RuntimeError("Joint t-SNE split did not consume every coordinate.")
    metadata.update(
        {
            "embedding_mode": "joint_dataset",
            "dataset": dataset,
            "joint_num_points": int(len(joint_features)),
            "variants": [label for _key, label, _arg in VARIANTS],
        }
    )
    return embeddings, metadata


def shared_axis_limits(
    datasets: Sequence[str],
    embeddings: Mapping[Tuple[str, str], np.ndarray],
) -> Dict[str, Tuple[float, float, float, float]]:
    limits: Dict[str, Tuple[float, float, float, float]] = {}
    for dataset in datasets:
        coordinates = np.concatenate(
            [
                embeddings[(dataset, variant_key)]
                for variant_key, _label, _arg in VARIANTS
            ],
            axis=0,
        )
        if not np.isfinite(coordinates).all():
            raise ValueError("Joint t-SNE coordinates contain NaN or infinity.")
        x_min, y_min = coordinates.min(axis=0)
        x_max, y_max = coordinates.max(axis=0)
        center_x = 0.5 * float(x_min + x_max)
        center_y = 0.5 * float(y_min + y_max)
        half_extent = 0.53 * max(float(x_max - x_min), float(y_max - y_min))
        if half_extent <= 0:
            raise ValueError("Joint t-SNE coordinates have zero spatial extent.")
        limits[dataset] = (
            center_x - half_extent,
            center_x + half_extent,
            center_y - half_extent,
            center_y + half_extent,
        )
    return limits


def panel_metrics(panel: FeaturePanel) -> Dict[str, float]:
    _sklearn, _PCA, _TSNE, silhouette_score = require_sklearn()
    silhouette = float(
        silhouette_score(panel.features, panel.identity_labels, metric="cosine")
    )

    by_modality = {
        modality: panel.features[panel.modalities == modality]
        for modality in MODALITIES
    }
    sample_orders = {
        modality: panel.sample_indices[panel.modalities == modality]
        for modality in MODALITIES
    }
    for modality in MODALITIES:
        order = np.argsort(sample_orders[modality])
        by_modality[modality] = by_modality[modality][order]
        sample_orders[modality] = sample_orders[modality][order]
    if not all(
        np.array_equal(sample_orders["R"], sample_orders[modality])
        for modality in ("N", "T")
    ):
        raise ValueError("R/N/T features do not describe the same sampled records.")
    cross_modal_distances = []
    for left, right in (("R", "N"), ("R", "T"), ("N", "T")):
        cosine_similarity = np.sum(by_modality[left] * by_modality[right], axis=1)
        cross_modal_distances.append(1.0 - cosine_similarity)
    cross_modal_distance = float(np.mean(np.concatenate(cross_modal_distances)))

    values: Dict[str, float] = {
        "silhouette_cosine": silhouette,
        "cross_modal_positive_distance": cross_modal_distance,
        "routing_accuracy": float(panel.route_correct.mean()),
    }
    for modality in MODALITIES:
        values["routing_accuracy_{}".format(modality)] = float(
            panel.route_correct[panel.modalities == modality].mean()
        )
    return values


def identity_palette(identity_count: int) -> List[object]:
    if identity_count <= len(IDENTITY_COLORS):
        return list(IDENTITY_COLORS[:identity_count])
    LOGGER.warning(
        "%d identities exceed the eight-color accessible palette; using tab20.",
        identity_count,
    )
    cmap = plt.get_cmap("tab20", identity_count)
    return [cmap(index) for index in range(identity_count)]


def draw_panel(
    ax: plt.Axes,
    panel: FeaturePanel,
    embedding: np.ndarray,
    metrics: Mapping[str, float],
    palette: Sequence[object],
    letter: Optional[str] = None,
    show_metrics: bool = True,
    connect_modalities: bool = False,
    show_route_status: bool = False,
    axis_limits: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    if connect_modalities:
        for sample_index in sorted(set(panel.sample_indices.tolist())):
            indices = []
            for modality in MODALITIES:
                matches = np.flatnonzero(
                    (panel.sample_indices == sample_index)
                    & (panel.modalities == modality)
                )
                if len(matches) != 1:
                    raise ValueError(
                        "Each synchronized record must contain exactly one R/N/T point."
                    )
                indices.append(int(matches[0]))
            ax.plot(
                embedding[indices, 0],
                embedding[indices, 1],
                color="#A8A8A8",
                linewidth=0.30,
                alpha=0.24,
                solid_capstyle="round",
                zorder=1,
            )

    for identity_label in sorted(set(panel.identity_labels.tolist())):
        color = palette[int(identity_label)]
        for modality in MODALITIES:
            base_mask = (panel.identity_labels == identity_label) & (
                panel.modalities == modality
            )
            correct_mask = base_mask & (
                panel.route_correct if show_route_status else np.ones_like(
                    panel.route_correct, dtype=bool
                )
            )
            if bool(correct_mask.any()):
                ax.scatter(
                    embedding[correct_mask, 0],
                    embedding[correct_mask, 1],
                    s=14,
                    marker=MODALITY_MARKERS[modality],
                    c=[color],
                    alpha=0.82,
                    linewidths=0.35,
                    edgecolors="white",
                    rasterized=False,
                    zorder=3,
                )
            if show_route_status:
                incorrect_mask = base_mask & ~panel.route_correct
                if bool(incorrect_mask.any()):
                    ax.scatter(
                        embedding[incorrect_mask, 0],
                        embedding[incorrect_mask, 1],
                        s=16,
                        marker=MODALITY_MARKERS[modality],
                        facecolors="none",
                        edgecolors=[color],
                        alpha=0.95,
                        linewidths=0.85,
                        rasterized=False,
                        zorder=4,
                    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if axis_limits is None:
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.06)
    else:
        x_min, x_max, y_min, y_max = axis_limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
    if letter:
        ax.text(
            0.01,
            0.99,
            "({})".format(letter),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
        )
    if show_metrics:
        ax.text(
            0.99,
            0.01,
            "Sil.={:.3f}  Route={:.1f}%".format(
                metrics["silhouette_cosine"],
                100.0 * metrics["routing_accuracy"],
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=3,
        )


def legend_handles(
    identity_count: int,
    palette: Sequence[object],
    connect_modalities: bool,
    show_route_status: bool,
):
    identity_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=palette[index],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=5.2,
            label="ID {}".format(index + 1),
        )
        for index in range(identity_count)
    ]
    modality_handles = [
        Line2D(
            [0],
            [0],
            marker=MODALITY_MARKERS[modality],
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=5.2,
            label=modality,
        )
        for modality in MODALITIES
    ]
    structure_handles: List[Line2D] = []
    if show_route_status:
        structure_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="#777777",
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    markersize=5.2,
                    label="Correct route",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="none",
                    markeredgecolor="#777777",
                    markeredgewidth=0.85,
                    markersize=5.2,
                    label="Wrong route",
                ),
            ]
        )
    if connect_modalities:
        structure_handles.append(
            Line2D(
                [0, 1],
                [0, 0],
                color="#A8A8A8",
                linewidth=0.6,
                label="Same R/N/T record",
            )
        )
    return identity_handles, modality_handles, structure_handles


def save_figure(fig: plt.Figure, base: Path) -> List[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("svg", "pdf", "png", "tiff"):
        path = base.with_suffix(".{}".format(extension))
        save_kwargs = {"dpi": 600, "facecolor": "white"}
        if extension == "tiff":
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(str(path), **save_kwargs)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Figure export failed: {}".format(path))
        outputs.append(path)
    return outputs


def write_plot_source_data(
    output_dir: Path,
    panels: Mapping[Tuple[str, str], FeaturePanel],
    embeddings: Mapping[Tuple[str, str], np.ndarray],
    embedding_mode: str,
) -> Path:
    suffix = "_joint" if embedding_mode == "joint_dataset" else ""
    path = output_dir / "source_data_tsne_coordinates{}.csv".format(suffix)
    fields = (
        "dataset",
        "variant",
        "embedding_mode",
        "stage",
        "identity_label",
        "pid",
        "modality",
        "sample_index",
        "path",
        "selected_task",
        "route_correct",
        "tsne_x",
        "tsne_y",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, panel in panels.items():
            embedding = embeddings[key]
            for index in range(len(panel.features)):
                writer.writerow(
                    {
                        "dataset": panel.dataset,
                        "variant": panel.variant_label,
                        "embedding_mode": embedding_mode,
                        "stage": panel.stage,
                        "identity_label": int(panel.identity_labels[index]) + 1,
                        "pid": int(panel.pids[index]),
                        "modality": str(panel.modalities[index]),
                        "sample_index": int(panel.sample_indices[index]),
                        "path": str(panel.paths[index]),
                        "selected_task": str(panel.selected_tasks[index]),
                        "route_correct": int(panel.route_correct[index]),
                        "tsne_x": "{:.8f}".format(float(embedding[index, 0])),
                        "tsne_y": "{:.8f}".format(float(embedding[index, 1])),
                    }
                )
    return path


def write_metrics(
    output_dir: Path,
    panels: Mapping[Tuple[str, str], FeaturePanel],
    metrics: Mapping[Tuple[str, str], Mapping[str, float]],
    tsne_metadata: Mapping[Tuple[str, str], Mapping[str, object]],
    embedding_mode: str,
) -> Path:
    suffix = "_joint" if embedding_mode == "joint_dataset" else ""
    path = output_dir / "source_data_panel_metrics{}.csv".format(suffix)
    fields = (
        "dataset",
        "variant",
        "embedding_mode",
        "stage",
        "num_points",
        "silhouette_cosine",
        "cross_modal_positive_distance",
        "routing_accuracy",
        "routing_accuracy_R",
        "routing_accuracy_N",
        "routing_accuracy_T",
        "tsne_kl_divergence",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, panel in panels.items():
            values = metrics[key]
            writer.writerow(
                {
                    "dataset": panel.dataset,
                    "variant": panel.variant_label,
                    "embedding_mode": embedding_mode,
                    "stage": panel.stage,
                    "num_points": len(panel.features),
                    "silhouette_cosine": "{:.8f}".format(
                        values["silhouette_cosine"]
                    ),
                    "cross_modal_positive_distance": "{:.8f}".format(
                        values["cross_modal_positive_distance"]
                    ),
                    "routing_accuracy": "{:.8f}".format(
                        values["routing_accuracy"]
                    ),
                    "routing_accuracy_R": "{:.8f}".format(
                        values["routing_accuracy_R"]
                    ),
                    "routing_accuracy_N": "{:.8f}".format(
                        values["routing_accuracy_N"]
                    ),
                    "routing_accuracy_T": "{:.8f}".format(
                        values["routing_accuracy_T"]
                    ),
                    "tsne_kl_divergence": "{:.8f}".format(
                        tsne_metadata[key]["kl_divergence"]
                    ),
                }
            )
    return path


def create_grid_figure(
    output_dir: Path,
    datasets: Sequence[str],
    panels: Mapping[Tuple[str, str], FeaturePanel],
    embeddings: Mapping[Tuple[str, str], np.ndarray],
    metrics: Mapping[Tuple[str, str], Mapping[str, float]],
    identity_count: int,
    hide_metrics: bool,
    embedding_mode: str,
    connect_modalities: bool,
    show_route_status: bool,
    dataset_limits: Optional[
        Mapping[str, Tuple[float, float, float, float]]
    ] = None,
) -> List[Path]:
    palette = identity_palette(identity_count)
    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=len(VARIANTS),
        figsize=(183 / 25.4, 119 / 25.4),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.995,
        bottom=0.165,
        top=0.925,
        wspace=0.055,
        hspace=0.12,
    )
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for row, dataset in enumerate(datasets):
        for column, (variant_key, variant_label, _argument) in enumerate(VARIANTS):
            key = (dataset, variant_key)
            ax = axes[row, column]
            draw_panel(
                ax=ax,
                panel=panels[key],
                embedding=embeddings[key],
                metrics=metrics[key],
                palette=palette,
                letter=next(letters),
                show_metrics=False,
                connect_modalities=connect_modalities,
                show_route_status=show_route_status,
                axis_limits=(
                    dataset_limits[dataset]
                    if dataset_limits is not None
                    else None
                ),
            )
            # if row == 0:
            #     ax.set_title(variant_label, fontsize=9, pad=4, fontweight="semibold")
            if column == 0:
                ax.text(
                    -0.17,
                    0.50,
                    DATASET_LABELS.get(dataset, dataset),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=8.2,
                    fontweight="semibold",
                )

    identity_handles, modality_handles, structure_handles = legend_handles(
        identity_count,
        palette,
        connect_modalities=connect_modalities,
        show_route_status=show_route_status,
    )
    fig.legend(
        handles=identity_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.064),
        ncol=identity_count,
        frameon=False,
        columnspacing=0.75,
        handletextpad=0.25,
        fontsize=7,
    )
    fig.legend(
        handles=modality_handles + structure_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.012),
        ncol=len(modality_handles + structure_handles),
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.35,
        fontsize=7,
        title="Encoding",
        title_fontsize=7,
    )
    suffix = "_joint" if embedding_mode == "joint_dataset" else ""
    outputs = save_figure(
        fig, output_dir / "lifelong_tsne_comparison{}".format(suffix)
    )
    plt.close(fig)
    return outputs


def create_individual_figures(
    output_dir: Path,
    datasets: Sequence[str],
    panels: Mapping[Tuple[str, str], FeaturePanel],
    embeddings: Mapping[Tuple[str, str], np.ndarray],
    metrics: Mapping[Tuple[str, str], Mapping[str, float]],
    identity_count: int,
    hide_metrics: bool,
    embedding_mode: str,
    connect_modalities: bool,
    show_route_status: bool,
    dataset_limits: Optional[
        Mapping[str, Tuple[float, float, float, float]]
    ] = None,
) -> List[Path]:
    palette = identity_palette(identity_count)
    identity_handles, modality_handles, structure_handles = legend_handles(
        identity_count,
        palette,
        connect_modalities=connect_modalities,
        show_route_status=show_route_status,
    )
    outputs: List[Path] = []
    for dataset in datasets:
        for variant_key, variant_label, _argument in VARIANTS:
            key = (dataset, variant_key)
            fig, ax = plt.subplots(figsize=(89 / 25.4, 72 / 25.4))
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.20, top=0.88)
            draw_panel(
                ax=ax,
                panel=panels[key],
                embedding=embeddings[key],
                metrics=metrics[key],
                palette=palette,
                show_metrics=False,
                connect_modalities=connect_modalities,
                show_route_status=show_route_status,
                axis_limits=(
                    dataset_limits[dataset]
                    if dataset_limits is not None
                    else None
                ),
            )
            # ax.set_title(
            #     "{} — {}".format(dataset, variant_label),
            #     fontsize=9,
            #     pad=4,
            #     fontweight="semibold",
            # )
            fig.legend(
                handles=identity_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.075),
                ncol=min(identity_count, 8),
                frameon=False,
                columnspacing=0.65,
                handletextpad=0.20,
                fontsize=6.5,
            )
            fig.legend(
                handles=modality_handles + structure_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.005),
                ncol=min(len(modality_handles + structure_handles), 3),
                frameon=False,
                columnspacing=1.1,
                handletextpad=0.25,
                fontsize=6.5,
            )
            outputs.extend(
                save_figure(
                    fig,
                    output_dir
                    / (
                        "individual_joint"
                        if embedding_mode == "joint_dataset"
                        else "individual"
                    )
                    / "{}__{}".format(dataset.lower(), variant_key),
                )
            )
            plt.close(fig)
    return outputs


def plot_all(
    args: argparse.Namespace,
    datasets: Sequence[str],
    output_dir: Path,
) -> None:
    panels: Dict[Tuple[str, str], FeaturePanel] = {}
    embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    tsne_metadata: Dict[Tuple[str, str], Dict[str, object]] = {}
    identity_count: Optional[int] = None

    for dataset in datasets:
        for variant_key, _variant_label, _argument in VARIANTS:
            key = (dataset, variant_key)
            panel = load_panel(panel_path(output_dir, dataset, variant_key))
            panel_identity_count = len(set(panel.identity_labels.tolist()))
            if identity_count is None:
                identity_count = panel_identity_count
            elif identity_count != panel_identity_count:
                raise ValueError("All panels must contain the same number of identities.")
            panels[key] = panel
            metrics[key] = panel_metrics(panel)

    if args.embedding_mode == "joint_dataset":
        for dataset in datasets:
            dataset_embeddings, metadata = run_joint_dataset_tsne(
                dataset=dataset,
                panels=panels,
                seed=int(args.seed),
                perplexity=float(args.perplexity),
                iterations=int(args.tsne_iterations),
            )
            embeddings.update(dataset_embeddings)
            for variant_key, _variant_label, _argument in VARIANTS:
                key = (dataset, variant_key)
                tsne_metadata[key] = dict(metadata)
        dataset_limits: Optional[
            Dict[str, Tuple[float, float, float, float]]
        ] = shared_axis_limits(datasets, embeddings)
    else:
        for dataset in datasets:
            for variant_key, _variant_label, _argument in VARIANTS:
                key = (dataset, variant_key)
                panel = panels[key]
                LOGGER.info(
                    "Running independent t-SNE: %s | %s",
                    panel.variant_label,
                    dataset,
                )
                embedding, _kl, metadata = run_tsne(
                    panel.features,
                    seed=int(args.seed),
                    perplexity=float(args.perplexity),
                    iterations=int(args.tsne_iterations),
                )
                metadata["embedding_mode"] = "independent"
                embeddings[key] = embedding
                tsne_metadata[key] = metadata
        dataset_limits = None

    connect_modalities = (
        args.embedding_mode == "joint_dataset"
        and not bool(args.no_connect_modalities)
    )
    show_route_status = (
        args.embedding_mode == "joint_dataset"
        and not bool(args.hide_route_status)
    )
    source_path = write_plot_source_data(
        output_dir,
        panels,
        embeddings,
        embedding_mode=str(args.embedding_mode),
    )
    metrics_path = write_metrics(
        output_dir,
        panels,
        metrics,
        tsne_metadata,
        embedding_mode=str(args.embedding_mode),
    )
    outputs = create_grid_figure(
        output_dir=output_dir,
        datasets=datasets,
        panels=panels,
        embeddings=embeddings,
        metrics=metrics,
        identity_count=int(identity_count),
        hide_metrics=bool(args.hide_metrics),
        embedding_mode=str(args.embedding_mode),
        connect_modalities=connect_modalities,
        show_route_status=show_route_status,
        dataset_limits=dataset_limits,
    )
    if not args.no_individual_panels:
        outputs.extend(
            create_individual_figures(
                output_dir=output_dir,
                datasets=datasets,
                panels=panels,
                embeddings=embeddings,
                metrics=metrics,
                identity_count=int(identity_count),
                hide_metrics=bool(args.hide_metrics),
                embedding_mode=str(args.embedding_mode),
                connect_modalities=connect_modalities,
                show_route_status=show_route_status,
                dataset_limits=dataset_limits,
            )
        )

    metadata_suffix = "_joint" if args.embedding_mode == "joint_dataset" else ""
    metadata_path = output_dir / "plot_metadata{}.json".format(metadata_suffix)
    serializable_metadata = {
        "figure": "2x3 t-SNE comparison",
        "embedding_mode": str(args.embedding_mode),
        "datasets": list(datasets),
        "variants": [item[1] for item in VARIANTS],
        "routing": "auto",
        "identity_encoding": "color",
        "modality_encoding": "marker shape",
        "routing_encoding": (
            "filled=correct, hollow=wrong"
            if show_route_status
            else "not displayed"
        ),
        "synchronized_record_encoding": (
            "R/N/T points connected by gray line"
            if connect_modalities
            else "not displayed"
        ),
        "metrics_note": (
            "Silhouette and cross-modal positive distance are computed in the "
            "original L2-normalized descriptor space, not in t-SNE space."
        ),
        "panels": {
            "{}__{}".format(dataset, variant): tsne_metadata[(dataset, variant)]
            for dataset, variant in tsne_metadata
        },
    }
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(serializable_metadata, handle, ensure_ascii=False, indent=2)

    LOGGER.info("t-SNE coordinates: %s", source_path)
    LOGGER.info("Panel metrics: %s", metrics_path)
    LOGGER.info("Plot metadata: %s", metadata_path)
    LOGGER.info("Exported %d figure files.", len(outputs))
    for output in outputs:
        LOGGER.info("  %s", output)


def main() -> None:
    args = parse_args()
    setup_logging()
    datasets = parse_datasets(args.datasets)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage < 1:
        raise ValueError("--stage must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be positive.")
    if args.tsne_iterations < 250:
        raise ValueError("--tsne_iterations must be at least 250.")

    if args.mode in ("all", "extract"):
        extract_all(args, datasets, output_dir)
    if args.mode in ("all", "plot"):
        plot_all(args, datasets, output_dir)


if __name__ == "__main__":
    main()
