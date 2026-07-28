"""Data protocols used by the exemplar-free lifelong multi-modal ReID tracks.

This module is intentionally independent from the original ``make_dataloader`` so
that the published single-dataset MDReID pipeline remains reproducible.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .datasets.sampler import RandomIdentitySampler


ImageFile.LOAD_TRUNCATED_IMAGES = True

MODALITIES = ("R", "N", "T")
WM_PROTOCOL_VERSION = "local-clean-v1"
WM_TRAIN_TEST_OVERLAP = {"0325", "0331", "0397", "0406", "0451", "0471", "0862"}

RecordPath = Union[str, Tuple[str, str, str]]
ReIDRecord = Tuple[RecordPath, int, int, int]

_CANONICAL_NAMES = {
    "rgbnt201": "RGBNT201",
    "market-mm": "Market-MM",
    "market_mm": "Market-MM",
    "marketmm": "Market-MM",
    "market1501_mm": "Market-MM",
    "rgbnt100": "RGBNT100",
    "msvr310": "MSVR310",
    "wmveid863": "WMVeID863",
    "msvwild863": "WMVeID863",
}

_TARGET_SIZES = {
    "RGBNT201": (256, 128),
    "Market-MM": (256, 128),
    "RGBNT100": (128, 256),
    "MSVR310": (128, 256),
    "WMVeID863": (128, 256),
}

_OBJECT_CATEGORIES = {
    "RGBNT201": "person",
    "Market-MM": "person",
    "RGBNT100": "vehicle",
    "MSVR310": "vehicle",
    "WMVeID863": "vehicle",
}


def canonical_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in _CANONICAL_NAMES:
        raise KeyError("Unsupported lifelong dataset: {}".format(name))
    return _CANONICAL_NAMES[key]


def dataset_task_key(name: str) -> str:
    return canonical_dataset_name(name).lower().replace("-", "_")


def dataset_target_size(name: str) -> Tuple[int, int]:
    return _TARGET_SIZES[canonical_dataset_name(name)]


def dataset_object_category(name: str) -> str:
    """Return the coarse detector category used by category-aware routing."""
    return _OBJECT_CATEGORIES[canonical_dataset_name(name)]


def _find_dataset_dir(root: Union[str, Path], candidates: Sequence[str]) -> Path:
    root = Path(root).expanduser().resolve()
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "Dataset directory not found. Tried: {}".format(
            ", ".join(str(root / candidate) for candidate in candidates)
        )
    )


def _image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _relabel(records: Sequence[ReIDRecord]) -> List[ReIDRecord]:
    identities = sorted({record[1] for record in records})
    pid_to_label = {pid: label for label, pid in enumerate(identities)}
    return [(paths, pid_to_label[pid], camid, sceneid) for paths, pid, camid, sceneid in records]


def _require_triplet(rgb_path: Path, nir_dir: Path, tir_dir: Path) -> Optional[Tuple[str, str, str]]:
    nir_path = nir_dir / rgb_path.name
    tir_path = tir_dir / rgb_path.name
    if not nir_path.is_file() or not tir_path.is_file():
        return None
    return str(rgb_path), str(nir_path), str(tir_path)


def _load_flat_paired_split(
    split_dir: Path,
    pid_cam_parser,
    modality_dirs: Tuple[str, str, str] = ("RGB", "NI", "TI"),
    drop_junk: bool = True,
) -> Tuple[List[ReIDRecord], int]:
    rgb_dir, nir_dir, tir_dir = (split_dir / name for name in modality_dirs)
    records: List[ReIDRecord] = []
    dropped = 0
    for rgb_path in _image_files(rgb_dir):
        parsed = pid_cam_parser(rgb_path.name)
        if parsed is None:
            dropped += 1
            continue
        pid, camid, sceneid = parsed
        if drop_junk and pid == -1:
            continue
        triplet = _require_triplet(rgb_path, nir_dir, tir_dir)
        if triplet is None:
            dropped += 1
            continue
        records.append((triplet, pid, camid, sceneid))
    return records, dropped


def _parse_rgbnt201(filename: str) -> Optional[Tuple[int, int, int]]:
    match = re.match(r"(?P<pid>\d+)_cam(?P<cam>\d+)", filename)
    if not match:
        return None
    return int(match.group("pid")), int(match.group("cam")) - 1, -1


def _parse_market(filename: str) -> Optional[Tuple[int, int, int]]:
    match = re.match(r"(?P<pid>-?\d+)_c(?P<cam>\d+)", filename)
    if not match:
        return None
    return int(match.group("pid")), int(match.group("cam")) - 1, -1


def _load_rgbnt201(root: Union[str, Path], load_train: bool) -> "ProtocolData":
    dataset_dir = _find_dataset_dir(root, ("RGBNT201",))
    train, dropped_train = ([], 0)
    if load_train:
        train, dropped_train = _load_flat_paired_split(dataset_dir / "train_171", _parse_rgbnt201)
        train = _relabel(train)
    query, dropped_query = _load_flat_paired_split(dataset_dir / "test", _parse_rgbnt201)
    gallery = list(query)
    return ProtocolData(
        name="RGBNT201",
        dataset_dir=dataset_dir,
        train=train,
        query=query,
        gallery=gallery,
        target_size=_TARGET_SIZES["RGBNT201"],
        protocol="official-local",
        audit={
            "dropped_train": dropped_train,
            "dropped_query": dropped_query,
            "note": "test is used as both query and gallery; same-ID same-camera items are filtered by evaluation.",
        },
    )


def _load_market_mm(root: Union[str, Path], load_train: bool) -> "ProtocolData":
    dataset_dir = _find_dataset_dir(root, ("market1501_mm", "Market-MM", "Market_MM"))
    train, dropped_train = ([], 0)
    if load_train:
        train, dropped_train = _load_flat_paired_split(dataset_dir / "train", _parse_market)
        train = _relabel(train)
    query, dropped_query = _load_flat_paired_split(dataset_dir / "query", _parse_market)
    gallery, dropped_gallery = _load_flat_paired_split(dataset_dir / "gallery", _parse_market)
    return ProtocolData(
        name="Market-MM",
        dataset_dir=dataset_dir,
        train=train,
        query=query,
        gallery=gallery,
        target_size=_TARGET_SIZES["Market-MM"],
        protocol="official-local",
        audit={
            "dropped_train": dropped_train,
            "dropped_query": dropped_query,
            "dropped_gallery": dropped_gallery,
        },
    )


def _load_rgbnt100(root: Union[str, Path], load_train: bool) -> "ProtocolData":
    dataset_dir = _find_dataset_dir(root, ("RGBNT100",)) / "rgbir"
    pattern = re.compile(r"(?P<pid>-?\d+)_c(?P<cam>\d+)")

    def load_split(directory: Path) -> List[ReIDRecord]:
        records: List[ReIDRecord] = []
        for path in _image_files(directory):
            match = pattern.search(path.name)
            if match is None:
                continue
            pid = int(match.group("pid"))
            if pid == -1:
                continue
            camid = int(match.group("cam")) - 1
            records.append((str(path), pid, camid, -1))
        return records

    train = _relabel(load_split(dataset_dir / "bounding_box_train")) if load_train else []
    query = load_split(dataset_dir / "query")
    gallery = load_split(dataset_dir / "bounding_box_test")
    return ProtocolData(
        name="RGBNT100",
        dataset_dir=dataset_dir,
        train=train,
        query=query,
        gallery=gallery,
        target_size=_TARGET_SIZES["RGBNT100"],
        protocol="official-local-composite",
        audit={"note": "Each 768x128 composite is split into R/N/T crops of 256x128."},
    )


def _parse_vehicle_filename(filename: str) -> Tuple[int, int]:
    scene_match = re.search(r"_s(?P<scene>\d+)", filename)
    view_match = re.search(r"(?:^|_)v(?P<view>\d+)", filename)
    sceneid = int(scene_match.group("scene")) if scene_match else -1
    camid = int(view_match.group("view")) if view_match else -1
    return camid, sceneid


def _load_nested_vehicle_split(
    split_dir: Path,
    excluded_ids: Optional[Iterable[str]] = None,
) -> Tuple[List[ReIDRecord], Dict[str, int]]:
    excluded = set(excluded_ids or ())
    records: List[ReIDRecord] = []
    dropped = 0
    excluded_count = 0
    if not split_dir.is_dir():
        raise FileNotFoundError(str(split_dir))
    for pid_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        if pid_dir.name in excluded:
            excluded_count += 1
            continue
        try:
            pid = int(pid_dir.name)
        except ValueError:
            dropped += 1
            continue
        rgb_dir, nir_dir, tir_dir = pid_dir / "vis", pid_dir / "ni", pid_dir / "th"
        for rgb_path in _image_files(rgb_dir):
            triplet = _require_triplet(rgb_path, nir_dir, tir_dir)
            if triplet is None:
                dropped += 1
                continue
            camid, sceneid = _parse_vehicle_filename(rgb_path.name)
            records.append((triplet, pid, camid, sceneid))
    return records, {
        "dropped_unreliable_samples": dropped,
        "excluded_identity_directories": excluded_count,
    }


def _load_msvr310(root: Union[str, Path], load_train: bool) -> "ProtocolData":
    dataset_dir = _find_dataset_dir(root, ("MSVR310",))
    train, train_audit = ([], {})
    if load_train:
        train, train_audit = _load_nested_vehicle_split(dataset_dir / "bounding_box_train")
        train = _relabel(train)
    query_dir = dataset_dir / "query3"
    if not query_dir.is_dir():
        query_dir = dataset_dir / "query"
    query, query_audit = _load_nested_vehicle_split(query_dir)
    gallery, gallery_audit = _load_nested_vehicle_split(dataset_dir / "bounding_box_test")
    return ProtocolData(
        name="MSVR310",
        dataset_dir=dataset_dir,
        train=train,
        query=query,
        gallery=gallery,
        target_size=_TARGET_SIZES["MSVR310"],
        protocol="official-local-scene-filter",
        audit={"train": train_audit, "query": query_audit, "gallery": gallery_audit},
    )


def _relative_record(record: ReIDRecord, dataset_dir: Path) -> Dict[str, object]:
    paths, pid, camid, sceneid = record
    if isinstance(paths, str):
        rel_paths = [Path(paths).resolve().relative_to(dataset_dir.resolve()).as_posix()]
    else:
        rel_paths = [Path(path).resolve().relative_to(dataset_dir.resolve()).as_posix() for path in paths]
    return {"paths": rel_paths, "pid": pid, "camid": camid, "sceneid": sceneid}


def _absolute_record(item: Dict[str, object], dataset_dir: Path) -> ReIDRecord:
    paths = tuple(str(dataset_dir / str(path)) for path in item["paths"])
    if len(paths) == 1:
        record_paths: RecordPath = paths[0]
    elif len(paths) == 3:
        record_paths = paths
    else:
        raise ValueError("A tri-modal manifest record must contain one composite path or three paths.")
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError("Manifest path does not exist: {}".format(path))
    return record_paths, int(item["pid"]), int(item["camid"]), int(item["sceneid"])


def _build_wm_manifest(dataset_dir: Path, include_train: bool = True) -> Dict[str, object]:
    if include_train:
        train, train_audit = _load_nested_vehicle_split(
            dataset_dir / "train", excluded_ids=WM_TRAIN_TEST_OVERLAP
        )
    else:
        train, train_audit = [], {"not_scanned": "evaluation-only manifest"}
    query, query_audit = _load_nested_vehicle_split(dataset_dir / "query")
    gallery, gallery_audit = _load_nested_vehicle_split(dataset_dir / "test")
    return {
        "protocol": WM_PROTOCOL_VERSION,
        "source": "local directories",
        "rules": {
            "anchor_modality": "vis",
            "pairing": "exact filename must exist in vis/ni/th",
            "excluded_train_ids": sorted(WM_TRAIN_TEST_OVERLAP),
            "unmatched_or_orphan_files": "drop",
        },
        "splits": {
            "train": [_relative_record(record, dataset_dir) for record in train],
            "query": [_relative_record(record, dataset_dir) for record in query],
            "gallery": [_relative_record(record, dataset_dir) for record in gallery],
        },
        "audit": {"train": train_audit, "query": query_audit, "gallery": gallery_audit},
    }


def _load_wmveid863(
    root: Union[str, Path],
    load_train: bool,
    manifest_path: Optional[Union[str, Path]],
) -> "ProtocolData":
    dataset_dir = _find_dataset_dir(root, ("WMVEID863", "WMVeID863", "MSVWild863"))
    manifest_file = Path(manifest_path).expanduser().resolve() if manifest_path else None
    if manifest_file is not None and manifest_file.is_file():
        with manifest_file.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("protocol") != WM_PROTOCOL_VERSION:
            raise ValueError(
                "Expected WMVeID863 protocol {}, got {}".format(
                    WM_PROTOCOL_VERSION, manifest.get("protocol")
                )
            )
    else:
        manifest = _build_wm_manifest(dataset_dir, include_train=load_train)
        if manifest_file is not None:
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            with manifest_file.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)

    splits = manifest["splits"]
    train = [_absolute_record(item, dataset_dir) for item in splits["train"]] if load_train else []
    if load_train:
        train = _relabel(train)
    query = [_absolute_record(item, dataset_dir) for item in splits["query"]]
    gallery = [_absolute_record(item, dataset_dir) for item in splits["gallery"]]
    audit = dict(manifest.get("audit", {}))
    audit["manifest"] = str(manifest_file) if manifest_file is not None else "in-memory"
    audit["excluded_train_ids"] = sorted(WM_TRAIN_TEST_OVERLAP)
    return ProtocolData(
        name="WMVeID863",
        dataset_dir=dataset_dir,
        train=train,
        query=query,
        gallery=gallery,
        target_size=_TARGET_SIZES["WMVeID863"],
        protocol=WM_PROTOCOL_VERSION,
        audit=audit,
    )


@dataclass
class ProtocolData:
    name: str
    dataset_dir: Path
    train: List[ReIDRecord]
    query: List[ReIDRecord]
    gallery: List[ReIDRecord]
    target_size: Tuple[int, int]
    protocol: str
    audit: Dict[str, object]

    @property
    def num_train_pids(self) -> int:
        return len({record[1] for record in self.train})

    @property
    def num_train_cams(self) -> int:
        return len({record[2] for record in self.train})

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "target_size": list(self.target_size),
            "train_pids": self.num_train_pids,
            "train_samples": len(self.train),
            "query_pids": len({record[1] for record in self.query}),
            "query_samples": len(self.query),
            "gallery_pids": len({record[1] for record in self.gallery}),
            "gallery_samples": len(self.gallery),
            "audit": self.audit,
        }


def load_protocol(
    name: str,
    root: Union[str, Path],
    load_train: bool = True,
    wm_manifest_path: Optional[Union[str, Path]] = None,
) -> ProtocolData:
    canonical = canonical_dataset_name(name)
    if canonical == "RGBNT201":
        return _load_rgbnt201(root, load_train)
    if canonical == "Market-MM":
        return _load_market_mm(root, load_train)
    if canonical == "RGBNT100":
        return _load_rgbnt100(root, load_train)
    if canonical == "MSVR310":
        return _load_msvr310(root, load_train)
    if canonical == "WMVeID863":
        return _load_wmveid863(root, load_train, wm_manifest_path)
    raise AssertionError("Unreachable dataset branch")


def _read_modalities(paths: RecordPath) -> List[Image.Image]:
    if isinstance(paths, str):
        with Image.open(paths) as image:
            image = image.convert("RGB")
            if image.width < 768 or image.height < 128:
                raise ValueError("RGBNT100 composite has invalid size {}: {}".format(image.size, paths))
            return [
                image.crop((0, 0, 256, 128)),
                image.crop((256, 0, 512, 128)),
                image.crop((512, 0, 768, 128)),
            ]
    images: List[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    if len(images) != 3:
        raise ValueError("Expected three modalities, got {}".format(len(images)))
    return images


class SynchronizedTriModalTransform:
    """Resize to a native task aspect ratio and synchronize all geometric noise."""

    def __init__(
        self,
        size: Tuple[int, int],
        training: bool,
        mean: Sequence[float],
        std: Sequence[float],
        flip_probability: float = 0.5,
        padding: int = 10,
        erasing_probability: float = 0.5,
    ):
        self.size = tuple(int(value) for value in size)
        self.training = training
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.flip_probability = flip_probability
        self.padding = int(padding)
        self.erasing_probability = erasing_probability

    def _erase(self, tensors: List[torch.Tensor]) -> None:
        if random.random() >= self.erasing_probability:
            return
        height, width = tensors[0].shape[-2:]
        area = height * width
        for _ in range(10):
            target_area = random.uniform(0.02, 0.33) * area
            aspect = math.exp(random.uniform(math.log(0.3), math.log(1.0 / 0.3)))
            erase_h = int(round(math.sqrt(target_area * aspect)))
            erase_w = int(round(math.sqrt(target_area / aspect)))
            if 0 < erase_h < height and 0 < erase_w < width:
                top = random.randint(0, height - erase_h)
                left = random.randint(0, width - erase_w)
                for tensor in tensors:
                    tensor[:, top : top + erase_h, left : left + erase_w] = 0
                return

    def __call__(self, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        height, width = self.size
        resized = [image.resize((width, height), Image.BICUBIC) for image in images]
        if self.training:
            if random.random() < self.flip_probability:
                transpose = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT
                resized = [image.transpose(transpose) for image in resized]
            if self.padding > 0:
                resized = [ImageOps.expand(image, border=self.padding, fill=0) for image in resized]
                max_top = 2 * self.padding
                max_left = 2 * self.padding
                top = random.randint(0, max_top)
                left = random.randint(0, max_left)
                resized = [image.crop((left, top, left + width, top + height)) for image in resized]
        tensors = [TF.normalize(TF.to_tensor(image), self.mean, self.std) for image in resized]
        if self.training:
            self._erase(tensors)
        return dict(zip(MODALITIES, tensors))


class TriModalImageDataset(Dataset):
    def __init__(self, records: Sequence[ReIDRecord], transform: SynchronizedTriModalTransform):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        paths, pid, camid, sceneid = self.records[index]
        images = self.transform(_read_modalities(paths))
        display_path = paths if isinstance(paths, str) else paths[0]
        return images, pid, camid, sceneid, display_path


def lifelong_collate(batch):
    images, pids, camids, sceneids, paths = zip(*batch)
    modal_batch = {
        modality: torch.stack([sample[modality] for sample in images], dim=0)
        for modality in MODALITIES
    }
    return (
        modal_batch,
        torch.tensor(pids, dtype=torch.long),
        torch.tensor(camids, dtype=torch.long),
        torch.tensor(sceneids, dtype=torch.long),
        tuple(paths),
    )


@dataclass
class TaskDataLoaders:
    protocol: ProtocolData
    train_loader: Optional[DataLoader]
    fingerprint_loader: Optional[DataLoader]
    eval_loader: DataLoader
    num_query: int


def build_task_dataloaders(
    name: str,
    root: Union[str, Path],
    train_batch_size: int,
    test_batch_size: int,
    num_instances: int,
    num_workers: int,
    mean: Sequence[float],
    std: Sequence[float],
    flip_probability: float,
    padding: int,
    erasing_probability: float,
    wm_manifest_path: Optional[Union[str, Path]] = None,
    load_train: bool = True,
) -> TaskDataLoaders:
    protocol = load_protocol(
        name=name,
        root=root,
        load_train=load_train,
        wm_manifest_path=wm_manifest_path,
    )
    eval_transform = SynchronizedTriModalTransform(
        size=protocol.target_size,
        training=False,
        mean=mean,
        std=std,
    )
    eval_dataset = TriModalImageDataset(protocol.query + protocol.gallery, eval_transform)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lifelong_collate,
        pin_memory=True,
    )

    train_loader: Optional[DataLoader] = None
    fingerprint_loader: Optional[DataLoader] = None
    if load_train:
        if not protocol.train:
            raise RuntimeError("{} contains no reliable training triplets.".format(protocol.name))
        if train_batch_size % num_instances != 0:
            raise ValueError("train_batch_size must be divisible by num_instances.")
        train_transform = SynchronizedTriModalTransform(
            size=protocol.target_size,
            training=True,
            mean=mean,
            std=std,
            flip_probability=flip_probability,
            padding=padding,
            erasing_probability=erasing_probability,
        )
        train_dataset = TriModalImageDataset(protocol.train, train_transform)
        sampler = RandomIdentitySampler(protocol.train, train_batch_size, num_instances)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=lifelong_collate,
            pin_memory=True,
        )
        # The Gaussian domain fingerprint must reflect the dataset rather than
        # stochastic augmentation. This loader reads only the current task's
        # training split with the deterministic evaluation transform.
        fingerprint_dataset = TriModalImageDataset(
            protocol.train, eval_transform
        )
        fingerprint_loader = DataLoader(
            fingerprint_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lifelong_collate,
            pin_memory=True,
        )
    return TaskDataLoaders(
        protocol=protocol,
        train_loader=train_loader,
        fingerprint_loader=fingerprint_loader,
        eval_loader=eval_loader,
        num_query=len(protocol.query),
    )
