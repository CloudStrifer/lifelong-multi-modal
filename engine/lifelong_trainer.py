"""Training, evaluation, and reporting for exemplar-free lifelong MDReID."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data.lifelong_datasets import (
    MODALITIES,
    TaskDataLoaders,
    build_task_dataloaders,
    canonical_dataset_name,
    dataset_object_category,
    dataset_task_key,
)
from layers.softmax_loss import LabelSmoothingCrossEntropy
from layers.triplet_loss import TripletLoss
from modeling.lifelong_model import LifelongMDReID


TRACKS = {
    "A": {
        "default": ["RGBNT201", "Market-MM"],
    },
    "B": {
        "default": ["RGBNT100", "MSVR310", "WMVeID863"],
    },
    "C": {
        "grouped": ["RGBNT201", "Market-MM", "RGBNT100", "MSVR310", "WMVeID863"],
        "interleaved": ["RGBNT201", "RGBNT100", "Market-MM", "MSVR310", "WMVeID863"],
        "alternate": ["RGBNT100", "RGBNT201", "MSVR310", "Market-MM", "WMVeID863"],
    },
}


def resolve_track(track: str, order: str) -> List[str]:
    track = track.upper()
    if track not in TRACKS:
        raise KeyError("Unknown track: {}".format(track))
    orders = TRACKS[track]
    key = order.lower() if track == "C" else "default"
    if key not in orders:
        raise KeyError("Track {} does not define order {}.".format(track, order))
    return list(orders[key])


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(cfg) -> torch.device:
    if str(cfg.MODEL.DEVICE).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:{}".format(cfg.MODEL.DEVICE_ID))
    return torch.device("cpu")


def _move_images(images: Dict[str, torch.Tensor], device: torch.device):
    return {
        modality: tensor.to(device, non_blocking=True)
        for modality, tensor in images.items()
    }


def _build_optimizer(cfg, model: LifelongMDReID):
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError("No trainable parameters for the current task.")
    router_parameters = [
        parameter
        for name, parameter in named_parameters
        if name.startswith("task_router_keys.")
    ]
    retrieval_parameters = [
        parameter
        for name, parameter in named_parameters
        if not name.startswith("task_router_keys.")
    ]
    parameter_groups = []
    if retrieval_parameters:
        parameter_groups.append({
            "params": retrieval_parameters,
            "lr": float(cfg.SOLVER.BASE_LR),
        })
    if router_parameters:
        parameter_groups.append({
            "params": router_parameters,
            "lr": (
                float(cfg.SOLVER.BASE_LR)
                * float(cfg.LIFELONG.ROUTER.KEY_LR_MULTIPLIER)
            ),
        })
    name = str(cfg.SOLVER.OPTIMIZER_NAME).lower()
    if name == "adamw":
        return torch.optim.AdamW(
            parameter_groups,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    if name == "adam":
        return torch.optim.Adam(
            parameter_groups,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            momentum=cfg.SOLVER.MOMENTUM,
        )
    raise ValueError("Unsupported lifelong optimizer: {}".format(cfg.SOLVER.OPTIMIZER_NAME))


def _cross_spectral_consistency(modal_features: Sequence[torch.Tensor]) -> torch.Tensor:
    normalized = [F.normalize(feature.float(), dim=1) for feature in modal_features]
    losses = []
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            losses.append(1.0 - (left * right).sum(dim=1).mean())
    return torch.stack(losses).mean()


def train_one_task(
    cfg,
    model: LifelongMDReID,
    task_key: str,
    train_loader,
    device: torch.device,
    logger: logging.Logger,
    periodic_evaluator: Optional[Callable[[int], Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Train only the current task bank for a fixed number of epochs."""
    model.set_trainable_task(task_key)
    model.to(device)
    optimizer = _build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(cfg.SOLVER.MAX_EPOCHS))
    )
    identity_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    triplet_loss_fn = TripletLoss(
        None if bool(cfg.MODEL.NO_MARGIN) else float(cfg.SOLVER.MARGIN)
    )
    amp_enabled = bool(cfg.LIFELONG.AMP) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    final_stats: Dict[str, object] = {}
    periodic_evaluations: List[Dict[str, object]] = []
    max_epochs = int(cfg.SOLVER.MAX_EPOCHS)
    periodic_eval_period = int(cfg.LIFELONG.PERIODIC_EVAL_PERIOD)

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = {
            "loss": 0.0,
            "id": 0.0,
            "triplet": 0.0,
            "consistency": 0.0,
            "router": 0.0,
            "router_positive": 0.0,
            "router_margin": 0.0,
            "router_separation": 0.0,
            "router_diversity": 0.0,
            "router_accuracy": 0.0,
            "router_accuracy_R": 0.0,
            "router_accuracy_N": 0.0,
            "router_accuracy_T": 0.0,
        }
        samples = 0
        for iteration, (images, labels, _, _, _) in enumerate(train_loader, start=1):
            images = _move_images(images, device)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model.forward_train(images, task_key)
                modal_outputs = output["modalities"]
                identity_loss = torch.stack([
                    identity_loss_fn(modal_outputs[modality]["logits"], labels)
                    for modality in MODALITIES
                ]).mean()
                modal_triplets = [
                    triplet_loss_fn(modal_outputs[modality]["descriptor"], labels)[0]
                    for modality in MODALITIES
                ]
                fused_triplet = triplet_loss_fn(output["fused"], labels)[0]
                triplet_loss = (torch.stack(modal_triplets).mean() + fused_triplet) / 2.0
                consistency_loss = _cross_spectral_consistency([
                    modal_outputs[modality]["descriptor"] for modality in MODALITIES
                ])
                router_losses = output.get("router_losses")
                if router_losses is None:
                    zero = output["fused"].sum() * 0.0
                    router_losses = {
                        "total": zero,
                        "positive": zero,
                        "margin": zero,
                        "separation": zero,
                        "diversity": zero,
                        "accuracy": zero,
                        "accuracy_R": zero,
                        "accuracy_N": zero,
                        "accuracy_T": zero,
                    }
                loss = (
                    float(cfg.MODEL.ID_LOSS_WEIGHT) * identity_loss
                    + float(cfg.MODEL.TRIPLET_LOSS_WEIGHT) * triplet_loss
                    + float(cfg.LIFELONG.CONSISTENCY_LOSS_WEIGHT) * consistency_loss
                    + float(cfg.LIFELONG.ROUTER.LOSS_WEIGHT)
                    * router_losses["total"]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if float(cfg.LIFELONG.GRAD_CLIP) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.trainable_parameters()), float(cfg.LIFELONG.GRAD_CLIP)
                )
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.shape[0]
            samples += batch_size
            running["loss"] += loss.item() * batch_size
            running["id"] += identity_loss.item() * batch_size
            running["triplet"] += triplet_loss.item() * batch_size
            running["consistency"] += consistency_loss.item() * batch_size
            running["router"] += router_losses["total"].item() * batch_size
            running["router_positive"] += (
                router_losses["positive"].item() * batch_size
            )
            running["router_margin"] += (
                router_losses["margin"].item() * batch_size
            )
            running["router_separation"] += (
                router_losses["separation"].item() * batch_size
            )
            running["router_diversity"] += (
                router_losses["diversity"].item() * batch_size
            )
            running["router_accuracy"] += (
                router_losses["accuracy"].item() * batch_size
            )
            for modality in MODALITIES:
                running["router_accuracy_{}".format(modality)] += (
                    router_losses["accuracy_{}".format(modality)].item()
                    * batch_size
                )
            if iteration % int(cfg.SOLVER.LOG_PERIOD) == 0:
                logger.info(
                    "task=%s epoch=%d/%d iter=%d/%d loss=%.4f",
                    task_key,
                    epoch,
                    cfg.SOLVER.MAX_EPOCHS,
                    iteration,
                    len(train_loader),
                    running["loss"] / max(1, samples),
                )
        scheduler.step()
        final_stats = {key: value / max(1, samples) for key, value in running.items()}
        logger.info(
            "task=%s epoch=%d fixed-schedule train loss=%.4f id=%.4f "
            "tri=%.4f consistency=%.4f router=%.4f "
            "router_pos=%.4f router_margin=%.4f "
            "router_acc_RNT/R/N/T=%.2f/%.2f/%.2f/%.2f",
            task_key,
            epoch,
            final_stats["loss"],
            final_stats["id"],
            final_stats["triplet"],
            final_stats["consistency"],
            final_stats["router"],
            final_stats["router_positive"],
            final_stats["router_margin"],
            final_stats["router_accuracy"] * 100.0,
            final_stats["router_accuracy_R"] * 100.0,
            final_stats["router_accuracy_N"] * 100.0,
            final_stats["router_accuracy_T"] * 100.0,
        )
        # The final epoch is followed immediately by the full stage evaluation.
        if (
            periodic_evaluator is not None
            and periodic_eval_period > 0
            and epoch % periodic_eval_period == 0
            and epoch < max_epochs
        ):
            monitor_result = periodic_evaluator(epoch)
            periodic_evaluations.append(monitor_result)
    final_stats["periodic_evaluations"] = periodic_evaluations
    return final_stats


@torch.no_grad()
def initialize_task_key_bank(
    cfg,
    model: LifelongMDReID,
    task_key: str,
    fingerprint_loader,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, object]:
    """Initialize current Task Keys from deterministic diverse features."""
    if fingerprint_loader is None:
        raise RuntimeError(
            "A deterministic current-task loader is required for "
            "Task-Key feature initialization."
        )
    initialization_batches = int(
        cfg.LIFELONG.ROUTER.TASK_KEY_INITIALIZATION_BATCHES
    )
    if initialization_batches < 1:
        raise ValueError("TASK_KEY_INITIALIZATION_BATCHES must be positive.")
    model.eval()
    feature_chunks = {modality: [] for modality in MODALITIES}
    for batch_index, (images, _, _, _, _) in enumerate(fingerprint_loader):
        images = _move_images(images, device)
        batch_features = model._encode_router_features(images, MODALITIES)
        for modality in MODALITIES:
            feature_chunks[modality].append(
                batch_features[modality].detach()
            )
        if batch_index + 1 >= initialization_batches:
            break
    if not feature_chunks["R"]:
        raise RuntimeError("Cannot initialize Task Keys from an empty loader.")
    router_features = {
        modality: torch.cat(feature_chunks[modality], dim=0)
        for modality in MODALITIES
    }
    bank = model.task_router_keys[task_key]
    bank.initialize_from_features(router_features)

    pair_similarity = {}
    keys = F.normalize(bank.keys.detach().float(), dim=2)
    for modality_index, modality in enumerate(MODALITIES):
        if keys.shape[1] <= 1:
            pair_similarity[modality] = 0.0
            continue
        similarity = keys[modality_index] @ keys[modality_index].t()
        upper = torch.triu_indices(
            keys.shape[1],
            keys.shape[1],
            offset=1,
            device=keys.device,
        )
        pair_similarity[modality] = float(
            similarity[upper[0], upper[1]].mean().cpu()
        )
    summary = {
        "task": task_key,
        "source_samples": int(next(iter(router_features.values())).shape[0]),
        "mean_pair_cosine": pair_similarity,
        "retained_samples": 0,
    }
    logger.info(
        "TASK-KEY-INITIALIZATION task=%s samples=%d pair_cosine_R/N/T="
        "%.4f/%.4f/%.4f retained_samples=0",
        task_key,
        summary["source_samples"],
        *[pair_similarity[modality] for modality in MODALITIES],
    )
    return summary


@torch.no_grad()
def fit_task_key_calibration(
    cfg,
    model: LifelongMDReID,
    task_key: str,
    fingerprint_loader,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, object]:
    """Fit robust per-modality raw-score calibration on the current task."""
    if fingerprint_loader is None:
        raise RuntimeError(
            "A deterministic current-task loader is required for "
            "Task-Key calibration."
        )
    calibration_floor = float(
        cfg.LIFELONG.ROUTER.TASK_KEY_CALIBRATION_FLOOR
    )
    if calibration_floor <= 0.0:
        raise ValueError("TASK_KEY_CALIBRATION_FLOOR must be positive.")
    if task_key not in model.task_router_keys:
        raise KeyError("Unregistered Task-Key bank: {}".format(task_key))

    model.eval()
    score_chunks = {modality: [] for modality in MODALITIES}
    sample_count = 0
    for images, _, _, _, _ in fingerprint_loader:
        images = _move_images(images, device)
        router_features = model._encode_router_features(images, MODALITIES)
        batch_size = next(iter(router_features.values())).shape[0]
        sample_count += int(batch_size)
        for modality in MODALITIES:
            raw_score = model._task_modality_key_score(
                feature=router_features[modality],
                task_key=task_key,
                modality=modality,
                calibrated=False,
            )
            score_chunks[modality].append(raw_score.float().cpu())
    if sample_count < 2:
        raise RuntimeError(
            "{} has only {} reliable samples; Task-Key calibration needs at "
            "least two.".format(task_key, sample_count)
        )

    medians = []
    iqrs = []
    modality_summary = {}
    for modality in MODALITIES:
        scores = torch.cat(score_chunks[modality])
        quartiles = torch.quantile(
            scores, torch.tensor([0.25, 0.5, 0.75])
        )
        median = quartiles[1]
        iqr = (quartiles[2] - quartiles[0]).clamp_min(calibration_floor)
        medians.append(median)
        iqrs.append(iqr)
        modality_summary[modality] = {
            "score_q25": float(quartiles[0]),
            "score_median": float(median),
            "score_q75": float(quartiles[2]),
            "calibration_iqr": float(iqr),
        }
    median_tensor = torch.stack(medians)
    iqr_tensor = torch.stack(iqrs)
    model.task_router_keys[task_key].set_calibration(
        median=median_tensor,
        iqr=iqr_tensor,
        sample_count=sample_count,
    )
    summary = {
        "task": task_key,
        "samples": sample_count,
        "modalities": modality_summary,
        "calibration_floor": calibration_floor,
        "retained_samples": 0,
    }
    logger.info(
        "TASK-KEY-CALIBRATION task=%s samples=%d R/N/T median=%.4f/%.4f/%.4f "
        "IQR=%.4f/%.4f/%.4f retained_samples=0",
        task_key,
        sample_count,
        *[modality_summary[m]["score_median"] for m in MODALITIES],
        *[modality_summary[m]["calibration_iqr"] for m in MODALITIES],
    )
    return summary


@torch.no_grad()
def fit_gaussian_fingerprint(
    cfg,
    model: LifelongMDReID,
    task_key: str,
    fingerprint_loader,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, object]:
    """Fit and freeze one diagonal-Gaussian R/N/T domain fingerprint.

    The implementation performs two deterministic passes over only the current
    training split. The first pass estimates feature moments; the second keeps
    only one scalar distance per sample to estimate robust score calibration.
    No image or per-sample feature is retained after this function returns.
    """
    if fingerprint_loader is None:
        raise RuntimeError(
            "A deterministic current-task loader is required for Gaussian routing."
        )
    shrinkage = float(cfg.LIFELONG.ROUTER.GAUSSIAN_VARIANCE_SHRINKAGE)
    absolute_floor = float(cfg.LIFELONG.ROUTER.GAUSSIAN_VARIANCE_FLOOR)
    relative_floor = float(
        cfg.LIFELONG.ROUTER.GAUSSIAN_RELATIVE_VARIANCE_FLOOR
    )
    calibration_floor = float(
        cfg.LIFELONG.ROUTER.GAUSSIAN_CALIBRATION_FLOOR
    )
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("GAUSSIAN_VARIANCE_SHRINKAGE must be in [0, 1].")
    if absolute_floor <= 0.0:
        raise ValueError("GAUSSIAN_VARIANCE_FLOOR must be positive.")
    if relative_floor < 0.0:
        raise ValueError("GAUSSIAN_RELATIVE_VARIANCE_FLOOR cannot be negative.")
    if calibration_floor <= 0.0:
        raise ValueError("GAUSSIAN_CALIBRATION_FLOOR must be positive.")

    model.eval()
    feature_sums = {
        modality: torch.zeros(model.feature_dim, dtype=torch.float64)
        for modality in MODALITIES
    }
    feature_square_sums = {
        modality: torch.zeros(model.feature_dim, dtype=torch.float64)
        for modality in MODALITIES
    }
    sample_count = 0
    for images, _, _, _, _ in fingerprint_loader:
        images = _move_images(images, device)
        features = model._encode_router_features(images, MODALITIES)
        batch_size = next(iter(features.values())).shape[0]
        sample_count += int(batch_size)
        for modality in MODALITIES:
            feature = features[modality].float()
            if model.gaussian_normalize_features:
                feature = F.normalize(feature, dim=1)
            feature_sums[modality] += feature.sum(dim=0).double().cpu()
            feature_square_sums[modality] += (
                feature.square().sum(dim=0).double().cpu()
            )
    if sample_count < 2:
        raise RuntimeError(
            "{} has only {} reliable samples; Gaussian routing needs at "
            "least two.".format(task_key, sample_count)
        )

    means = []
    variances = []
    for modality in MODALITIES:
        mean = feature_sums[modality] / float(sample_count)
        variance = (
            feature_square_sums[modality] / float(sample_count)
            - mean.square()
        ).clamp_min(0.0)
        spherical_variance = variance.mean()
        variance = (
            (1.0 - shrinkage) * variance
            + shrinkage * spherical_variance
        )
        modality_floor = max(
            absolute_floor,
            float(spherical_variance) * relative_floor,
        )
        means.append(mean.float())
        variances.append(variance.clamp_min(modality_floor).float())
    mean_tensor = torch.stack(means)
    variance_tensor = torch.stack(variances)

    device_means = {
        modality: mean_tensor[index].to(device)
        for index, modality in enumerate(MODALITIES)
    }
    device_variances = {
        modality: variance_tensor[index].to(device)
        for index, modality in enumerate(MODALITIES)
    }
    distance_chunks = {modality: [] for modality in MODALITIES}
    for images, _, _, _, _ in fingerprint_loader:
        images = _move_images(images, device)
        features = model._encode_router_features(images, MODALITIES)
        for modality in MODALITIES:
            feature = features[modality].float()
            if model.gaussian_normalize_features:
                feature = F.normalize(feature, dim=1)
            distance = (
                (feature - device_means[modality]).square()
                / device_variances[modality]
            ).mean(dim=1)
            distance_chunks[modality].append(distance.cpu())

    medians = []
    iqrs = []
    calibration_summary = {}
    for modality in MODALITIES:
        distances = torch.cat(distance_chunks[modality])
        quartiles = torch.quantile(
            distances, torch.tensor([0.25, 0.5, 0.75])
        )
        median = quartiles[1]
        iqr = (quartiles[2] - quartiles[0]).clamp_min(calibration_floor)
        medians.append(median)
        iqrs.append(iqr)
        calibration_summary[modality] = {
            "distance_q25": float(quartiles[0]),
            "distance_median": float(median),
            "distance_q75": float(quartiles[2]),
            "calibration_iqr": float(iqr),
            "mean_variance": float(
                variance_tensor[MODALITIES.index(modality)].mean()
            ),
        }
    median_tensor = torch.stack(medians)
    iqr_tensor = torch.stack(iqrs)
    model.set_gaussian_fingerprint(
        task_key=task_key,
        mean=mean_tensor,
        variance=variance_tensor,
        calibration_median=median_tensor,
        calibration_iqr=iqr_tensor,
        sample_count=sample_count,
    )
    summary = {
        "task": task_key,
        "samples": sample_count,
        "feature_dim": model.feature_dim,
        "modalities": calibration_summary,
        "normalize_features": model.gaussian_normalize_features,
        "variance_shrinkage": shrinkage,
        "variance_floor": absolute_floor,
        "relative_variance_floor": relative_floor,
        "calibration_floor": calibration_floor,
        "retained_samples": 0,
    }
    logger.info(
        "GAUSSIAN-FINGERPRINT task=%s samples=%d R/N/T median=%.4f/%.4f/%.4f "
        "IQR=%.4f/%.4f/%.4f retained_samples=0",
        task_key,
        sample_count,
        *[calibration_summary[m]["distance_median"] for m in MODALITIES],
        *[calibration_summary[m]["calibration_iqr"] for m in MODALITIES],
    )
    return summary


def _evaluate_rank(
    distance: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    query_sceneids: np.ndarray,
    gallery_sceneids: np.ndarray,
    scene_protocol: bool,
    max_rank: int = 50,
) -> Tuple[np.ndarray, float]:
    num_query, num_gallery = distance.shape
    max_rank = min(max_rank, num_gallery)
    indices = np.argsort(distance, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, None]).astype(np.int32)
    all_cmc: List[np.ndarray] = []
    all_ap: List[float] = []

    for query_index in range(num_query):
        order = indices[query_index]
        if scene_protocol:
            remove = (
                (gallery_pids[order] == query_pids[query_index])
                & (gallery_sceneids[order] == query_sceneids[query_index])
            )
        else:
            remove = (
                (gallery_pids[order] == query_pids[query_index])
                & (gallery_camids[order] == query_camids[query_index])
            )
        keep = ~remove
        raw_cmc = matches[query_index][keep]
        if not np.any(raw_cmc):
            continue
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        clipped = cmc[:max_rank]
        if clipped.shape[0] < max_rank:
            clipped = np.pad(clipped, (0, max_rank - clipped.shape[0]), mode="edge")
        all_cmc.append(clipped)
        relevant = raw_cmc.sum()
        precision = raw_cmc.cumsum() / (np.arange(raw_cmc.shape[0]) + 1.0)
        all_ap.append(float((precision * raw_cmc).sum() / relevant))

    if not all_cmc:
        raise RuntimeError("All query identities have no valid gallery match.")
    return np.asarray(all_cmc, dtype=np.float32).mean(axis=0), float(np.mean(all_ap))


@dataclass
class EvalTask:
    dataset_name: str
    task_key: str
    object_category: str
    loader: object
    num_query: int
    protocol_summary: Dict[str, object]


@torch.no_grad()
def evaluate_scenarios(
    model: LifelongMDReID,
    eval_task: EvalTask,
    scenarios: Sequence[str],
    candidate_tasks: Sequence[str],
    routing: str,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    scenarios = [str(scenario).upper() for scenario in scenarios]
    features = {scenario: [] for scenario in scenarios}
    selected_routes = {scenario: [] for scenario in scenarios}
    router_scores = {scenario: [] for scenario in scenarios}
    pids = []
    camids = []
    sceneids = []
    expected_index = candidate_tasks.index(eval_task.task_key)

    for images, batch_pids, batch_camids, batch_sceneids, _ in eval_task.loader:
        images = _move_images(images, device)
        outputs = model.extract_scenarios(
            images=images,
            scenarios=scenarios,
            routing=routing,
            oracle_task=eval_task.task_key if routing == "oracle" else None,
            candidate_tasks=candidate_tasks,
            object_category=eval_task.object_category,
        )
        for scenario, (descriptor, selected, scores) in outputs.items():
            features[scenario].append(descriptor.float().cpu())
            selected_routes[scenario].append(selected.cpu())
            router_scores[scenario].append(scores.float().cpu())
        pids.append(batch_pids.numpy())
        camids.append(batch_camids.numpy())
        sceneids.append(batch_sceneids.numpy())
    pid_array = np.concatenate(pids)
    camid_array = np.concatenate(camids)
    sceneid_array = np.concatenate(sceneids)
    results = {}
    for scenario in scenarios:
        feature_tensor = F.normalize(torch.cat(features[scenario], dim=0), dim=1)
        query_features = feature_tensor[: eval_task.num_query]
        gallery_features = feature_tensor[eval_task.num_query :]
        distance = (1.0 - query_features @ gallery_features.t()).numpy()
        cmc, mean_ap = _evaluate_rank(
            distance=distance,
            query_pids=pid_array[: eval_task.num_query],
            gallery_pids=pid_array[eval_task.num_query :],
            query_camids=camid_array[: eval_task.num_query],
            gallery_camids=camid_array[eval_task.num_query :],
            query_sceneids=sceneid_array[: eval_task.num_query],
            gallery_sceneids=sceneid_array[eval_task.num_query :],
            scene_protocol=eval_task.dataset_name == "MSVR310",
        )
        route_tensor = torch.cat(selected_routes[scenario])
        route_counts = torch.bincount(
            route_tensor, minlength=len(candidate_tasks)
        )
        score_tensor = torch.cat(router_scores[scenario])
        if model.category_aware_routing:
            allowed_indices = [
                index
                for index, task_key in enumerate(candidate_tasks)
                if model.task_categories[task_key] == eval_task.object_category
            ]
        else:
            allowed_indices = list(range(len(candidate_tasks)))
        allowed_scores = score_tensor[:, allowed_indices]
        if allowed_scores.shape[1] >= 2:
            top_scores = torch.topk(allowed_scores, k=2, dim=1).values
            routing_margin_mean = float(
                (top_scores[:, 0] - top_scores[:, 1]).mean()
            )
        else:
            routing_margin_mean = 0.0
        routing_score_mean = {
            task_key: (
                float(score_tensor[:, index].mean())
                if index in allowed_indices
                else None
            )
            for index, task_key in enumerate(candidate_tasks)
        }
        results[scenario] = {
            "mAP": mean_ap * 100.0,
            "R1": float(cmc[0]) * 100.0,
            "R5": float(cmc[min(4, len(cmc) - 1)]) * 100.0,
            "R10": float(cmc[min(9, len(cmc) - 1)]) * 100.0,
            "routing_accuracy": float(
                (route_tensor == expected_index).float().mean()
            ) * 100.0,
            "routing_distribution": {
                task_key: float(route_counts[index]) * 100.0 / len(route_tensor)
                for index, task_key in enumerate(candidate_tasks)
            },
            "routing_score_mean": routing_score_mean,
            "routing_margin_mean": routing_margin_mean,
        }
    return results


@torch.no_grad()
def evaluate_scenario(
    model: LifelongMDReID,
    eval_task: EvalTask,
    scenario: str,
    candidate_tasks: Sequence[str],
    routing: str,
    device: torch.device,
) -> Dict[str, float]:
    return evaluate_scenarios(
        model=model,
        eval_task=eval_task,
        scenarios=[scenario],
        candidate_tasks=candidate_tasks,
        routing=routing,
        device=device,
    )[str(scenario).upper()]


def _continual_summary(
    history: Sequence[Dict[str, object]],
    current_datasets: Dict[str, Dict[str, Dict[str, float]]],
    scenarios: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics = ("mAP", "R1", "R5", "R10")
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scenario in scenarios:
        average_seen = {}
        forgetting = {}
        for metric in metrics:
            current_values = [
                scenario_results[scenario][metric]
                for scenario_results in current_datasets.values()
            ]
            average_seen[metric] = float(np.mean(current_values))
            task_forgetting = []
            for task_key, scenario_results in current_datasets.items():
                previous_values = []
                for stage in history:
                    previous_task_results = stage["datasets"].get(task_key)
                    if previous_task_results is not None:
                        previous_values.append(previous_task_results[scenario][metric])
                if previous_values:
                    task_forgetting.append(
                        max(previous_values) - scenario_results[scenario][metric]
                    )
            forgetting[metric] = float(np.mean(task_forgetting)) if task_forgetting else 0.0
        summary[scenario] = {
            "average_seen": average_seen,
            "forgetting": forgetting,
        }
    return summary


def _reserved_any_to_any() -> Dict[str, None]:
    subsets = ("R", "N", "T", "RN", "RT", "NT", "RNT")
    enabled = {"R->R", "N->N", "T->T", "RNT->RNT"}
    return {
        "{}->{}".format(query, gallery): None
        for query in subsets
        for gallery in subsets
        if "{}->{}".format(query, gallery) not in enabled
    }


def save_checkpoint(
    path: Path,
    model: LifelongMDReID,
    stage: int,
    track: str,
    order: str,
    task_specs: Sequence[Dict[str, object]],
    cfg,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "track": track,
            "order": order,
            "task_specs": list(task_specs),
            "state_dict": model.state_dict(),
            "parameter_accounting": model.parameter_accounting(),
            "config": cfg.dump(),
        },
        str(path),
    )


def load_checkpoint_model(cfg, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = LifelongMDReID(cfg)
    for task_spec in checkpoint["task_specs"]:
        object_category = str(
            task_spec.get(
                "object_category",
                dataset_object_category(str(task_spec["dataset"])),
            )
        )
        model.register_task(
            task_key=str(task_spec["task_key"]),
            num_classes=int(task_spec["num_classes"]),
            object_category=object_category,
            initialize_from_history=False,
        )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    router_state_prefixes = (
        "task_router_keys.",
        "task_gaussian_fingerprints.",
    )
    missing_router_keys = [
        key
        for key in incompatible.missing_keys
        if key.startswith(router_state_prefixes)
    ]
    unexpected_router_keys = [
        key
        for key in incompatible.unexpected_keys
        if key.startswith(router_state_prefixes)
    ]
    other_missing = [
        key for key in incompatible.missing_keys if key not in missing_router_keys
    ]
    other_unexpected = [
        key
        for key in incompatible.unexpected_keys
        if key not in unexpected_router_keys
    ]
    if other_missing or other_unexpected:
        raise RuntimeError("Checkpoint mismatch: {}".format(incompatible))
    if missing_router_keys:
        missing_task_key_parameters = [
            key
            for key in missing_router_keys
            if key.startswith("task_router_keys.") and key.endswith(".keys")
        ]
        missing_task_key_calibration = [
            key
            for key in missing_router_keys
            if (
                key.startswith("task_router_keys.")
                and not key.endswith(".keys")
            )
        ]
        if model.router_method == "task_key" and missing_task_key_parameters:
            model.router_keys_ready = False
        logging.getLogger("MDReID.lifelong.test").warning(
            "The checkpoint does not contain router state required by method "
            "'%s'. Oracle evaluation remains valid, but auto routing requires "
            "a checkpoint trained with the same router method.",
            model.router_method,
        )
        if (
            model.router_method == "task_key"
            and missing_task_key_calibration
            and not missing_task_key_parameters
        ):
            logging.getLogger("MDReID.lifelong.test").warning(
                "Task-Key parameters were loaded, but this older checkpoint "
                "has no calibrated score statistics. Raw-score auto routing "
                "is available only with "
                "LIFELONG.ROUTER.TASK_KEY_CALIBRATION=False."
            )
    model.to(device)
    return model, checkpoint


def _write_reports(output_dir: Path, report: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "continual_results.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    result_rows = []
    summary_rows = []
    for stage in report["stages"]:
        for task_key, task_results in stage["datasets"].items():
            for scenario, metrics in task_results.items():
                result_rows.append({
                    "stage": stage["stage"],
                    "trained_task": stage["trained_task"],
                    "eval_task": task_key,
                    "scenario": scenario,
                    **metrics,
                })
        for scenario, values in stage["continual_summary"].items():
            summary_rows.append({
                "stage": stage["stage"],
                "trained_task": stage["trained_task"],
                "scenario": scenario,
                **{
                    "average_seen_{}".format(metric): value
                    for metric, value in values["average_seen"].items()
                },
                **{
                    "forgetting_{}".format(metric): value
                    for metric, value in values["forgetting"].items()
                },
            })

    if result_rows:
        with (output_dir / "stage_metrics.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
            writer.writeheader()
            writer.writerows(result_rows)
    if summary_rows:
        with (output_dir / "continual_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def _wm_manifest_path(cfg, output_dir: Path) -> str:
    if str(cfg.LIFELONG.WM_MANIFEST):
        return str(Path(cfg.LIFELONG.WM_MANIFEST).expanduser())
    return str(output_dir / "protocols" / "WMVeID863_local-clean-v1.json")


def build_cfg_dataloaders(
    cfg,
    dataset_name: str,
    output_dir: Path,
    load_train: bool,
) -> TaskDataLoaders:
    return build_task_dataloaders(
        name=dataset_name,
        root=cfg.DATASETS.ROOT_DIR,
        train_batch_size=int(cfg.SOLVER.IMS_PER_BATCH),
        test_batch_size=int(cfg.TEST.IMS_PER_BATCH),
        num_instances=int(cfg.DATALOADER.NUM_INSTANCE),
        num_workers=int(cfg.DATALOADER.NUM_WORKERS),
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD,
        flip_probability=float(cfg.INPUT.PROB),
        padding=int(cfg.INPUT.PADDING),
        erasing_probability=float(cfg.INPUT.RE_PROB),
        wm_manifest_path=_wm_manifest_path(cfg, output_dir),
        load_train=load_train,
    )


def run_lifelong_training(cfg, track: str, order: str) -> Dict[str, object]:
    logger = logging.getLogger("MDReID.lifelong")
    sequence = resolve_track(track, order)
    output_dir = Path(cfg.OUTPUT_DIR).expanduser().resolve() / (
        "track_{}_{}".format(track.upper(), order.lower())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(int(cfg.SOLVER.SEED))
    device = resolve_device(cfg)
    logger.info("Device: %s", device)
    logger.info("Sequence: %s", " -> ".join(sequence))

    model = LifelongMDReID(cfg).to(device)
    eval_tasks: Dict[str, EvalTask] = {}
    task_specs: List[Dict[str, object]] = []
    report: Dict[str, object] = {
        "track": track.upper(),
        "order": order.lower(),
        "sequence": sequence,
        "routing": str(cfg.LIFELONG.ROUTING),
        "router_method": str(cfg.LIFELONG.ROUTER.METHOD),
        "category_aware_routing": bool(cfg.LIFELONG.ROUTER.CATEGORY_AWARE),
        "scenarios": list(cfg.LIFELONG.EVAL_SCENARIOS),
        "no_replay": True,
        "retained_old_images": 0,
        "retained_old_sample_features": 0,
        "retained_router_statistics": (
            (
                str(cfg.LIFELONG.ROUTER.METHOD).lower()
                in ("gaussian", "gaussian_fingerprint")
            )
            or (
                str(cfg.LIFELONG.ROUTER.METHOD).lower() == "task_key"
                and bool(cfg.LIFELONG.ROUTER.TASK_KEY_CALIBRATION)
            )
        ),
        "checkpoint_selection": "fixed epoch; test sets are never used for model selection",
        "stages": [],
        "reserved_any_to_any": _reserved_any_to_any(),
    }

    for stage_index, dataset_name in enumerate(sequence, start=1):
        dataset_name = canonical_dataset_name(dataset_name)
        task_key = dataset_task_key(dataset_name)
        logger.info("Preparing stage %d: %s", stage_index, dataset_name)
        bundle = build_cfg_dataloaders(cfg, dataset_name, output_dir, load_train=True)
        protocol_summary = bundle.protocol.summary()
        logger.info("Protocol: %s", json.dumps(protocol_summary, ensure_ascii=False))
        model.register_task(
            task_key=task_key,
            num_classes=bundle.protocol.num_train_pids,
            object_category=dataset_object_category(dataset_name),
            initialize_from_history=True,
        )
        current_eval_task = EvalTask(
            dataset_name=dataset_name,
            task_key=task_key,
            object_category=dataset_object_category(dataset_name),
            loader=bundle.eval_loader,
            num_query=bundle.num_query,
            protocol_summary=protocol_summary,
        )
        task_key_initialization = None
        if (
            model.router_method == "task_key"
            and bool(cfg.LIFELONG.ROUTER.TASK_KEY_FEATURE_INITIALIZATION)
        ):
            task_key_initialization = initialize_task_key_bank(
                cfg=cfg,
                model=model,
                task_key=task_key,
                fingerprint_loader=bundle.fingerprint_loader,
                device=device,
                logger=logger,
            )

        def periodic_evaluator(epoch, eval_task=current_eval_task):
            scenarios = [
                str(value) for value in cfg.LIFELONG.PERIODIC_EVAL_SCENARIOS
            ]
            monitor_results = evaluate_scenarios(
                model=model,
                eval_task=eval_task,
                scenarios=scenarios,
                candidate_tasks=list(model.task_order),
                routing=str(cfg.LIFELONG.PERIODIC_EVAL_ROUTING),
                device=device,
            )
            for monitor_scenario, monitor_metrics in monitor_results.items():
                logger.info(
                    "PERIODIC-MONITOR task=%s epoch=%d %s routing=%s "
                    "mAP=%.2f R1=%.2f R5=%.2f R10=%.2f",
                    task_key,
                    epoch,
                    monitor_scenario,
                    cfg.LIFELONG.PERIODIC_EVAL_ROUTING,
                    monitor_metrics["mAP"],
                    monitor_metrics["R1"],
                    monitor_metrics["R5"],
                    monitor_metrics["R10"],
                )
            return {"epoch": epoch, "results": monitor_results}

        train_stats = train_one_task(
            cfg=cfg,
            model=model,
            task_key=task_key,
            train_loader=bundle.train_loader,
            device=device,
            logger=logger,
            periodic_evaluator=periodic_evaluator,
        )
        if task_key_initialization is not None:
            train_stats["task_key_initialization"] = task_key_initialization
        if (
            model.router_method == "task_key"
            and bool(cfg.LIFELONG.ROUTER.TASK_KEY_CALIBRATION)
        ):
            train_stats["task_key_calibration"] = fit_task_key_calibration(
                cfg=cfg,
                model=model,
                task_key=task_key,
                fingerprint_loader=bundle.fingerprint_loader,
                device=device,
                logger=logger,
            )
        elif model.router_method == "gaussian_fingerprint":
            train_stats["gaussian_fingerprint"] = fit_gaussian_fingerprint(
                cfg=cfg,
                model=model,
                task_key=task_key,
                fingerprint_loader=bundle.fingerprint_loader,
                device=device,
                logger=logger,
            )
        task_specs.append({
            "dataset": dataset_name,
            "task_key": task_key,
            "num_classes": bundle.protocol.num_train_pids,
            "object_category": dataset_object_category(dataset_name),
            "target_size": list(bundle.protocol.target_size),
            "protocol": bundle.protocol.protocol,
        })
        eval_tasks[task_key] = current_eval_task

        # Explicitly release all current/old training records before evaluation.
        bundle.protocol.train.clear()
        bundle.train_loader = None
        bundle.fingerprint_loader = None
        del bundle

        candidate_tasks = [spec["task_key"] for spec in task_specs]
        dataset_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for seen_task_key, eval_task in eval_tasks.items():
            scenario_results = evaluate_scenarios(
                model=model,
                eval_task=eval_task,
                scenarios=[str(value) for value in cfg.LIFELONG.EVAL_SCENARIOS],
                candidate_tasks=candidate_tasks,
                routing=str(cfg.LIFELONG.ROUTING),
                device=device,
            )
            dataset_results[seen_task_key] = scenario_results
            for scenario, metrics in scenario_results.items():
                logger.info(
                    "stage=%d eval=%s %s mAP=%.2f R1=%.2f R5=%.2f R10=%.2f route=%.2f",
                    stage_index,
                    seen_task_key,
                    scenario,
                    metrics["mAP"],
                    metrics["R1"],
                    metrics["R5"],
                    metrics["R10"],
                    metrics["routing_accuracy"],
                )
                if str(cfg.LIFELONG.ROUTING) == "auto":
                    logger.info(
                        "stage=%d eval=%s %s route_distribution=%s "
                        "route_score_mean=%s route_margin_mean=%.4f",
                        stage_index,
                        seen_task_key,
                        scenario,
                        json.dumps(
                            metrics["routing_distribution"],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        json.dumps(
                            metrics["routing_score_mean"],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        metrics["routing_margin_mean"],
                    )

        stage_report = {
            "stage": stage_index,
            "trained_task": task_key,
            "train_stats": train_stats,
            "protocol": protocol_summary,
            "datasets": dataset_results,
            "continual_summary": _continual_summary(
                history=report["stages"],
                current_datasets=dataset_results,
                scenarios=[str(value) for value in cfg.LIFELONG.EVAL_SCENARIOS],
            ),
            "parameter_accounting": model.parameter_accounting(),
        }
        report["stages"].append(stage_report)
        checkpoint_path = (
            output_dir / "checkpoints" / "stage_{:02d}_{}.pth".format(stage_index, task_key)
        )
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            stage=stage_index,
            track=track,
            order=order,
            task_specs=task_specs,
            cfg=cfg,
        )
        stage_report["checkpoint"] = str(checkpoint_path)
        _write_reports(output_dir, report)
    return report


@torch.no_grad()
def run_lifelong_test(
    cfg,
    checkpoint_path: str,
    routing: str,
) -> Dict[str, object]:
    logger = logging.getLogger("MDReID.lifelong.test")
    device = resolve_device(cfg)
    model, checkpoint = load_checkpoint_model(cfg, checkpoint_path, device)
    task_specs = checkpoint["task_specs"]
    output_dir = Path(checkpoint_path).expanduser().resolve().parent.parent
    candidates = [str(spec["task_key"]) for spec in task_specs]
    results = {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "stage": int(checkpoint["stage"]),
        "routing": routing,
        "datasets": {},
        "reserved_any_to_any": _reserved_any_to_any(),
    }
    for task_spec in task_specs:
        dataset_name = str(task_spec["dataset"])
        task_key = str(task_spec["task_key"])
        bundle = build_cfg_dataloaders(cfg, dataset_name, output_dir, load_train=False)
        eval_task = EvalTask(
            dataset_name=dataset_name,
            task_key=task_key,
            object_category=str(
                task_spec.get(
                    "object_category",
                    dataset_object_category(dataset_name),
                )
            ),
            loader=bundle.eval_loader,
            num_query=bundle.num_query,
            protocol_summary=bundle.protocol.summary(),
        )
        results["datasets"][task_key] = {}
        scenario_results = evaluate_scenarios(
            model=model,
            eval_task=eval_task,
            scenarios=[str(value) for value in cfg.LIFELONG.EVAL_SCENARIOS],
            candidate_tasks=candidates,
            routing=routing,
            device=device,
        )
        results["datasets"][task_key] = scenario_results
        for scenario, metrics in scenario_results.items():
            logger.info(
                "%s %s mAP=%.2f R1=%.2f R5=%.2f R10=%.2f route=%.2f",
                task_key,
                scenario,
                metrics["mAP"],
                metrics["R1"],
                metrics["R5"],
                metrics["R10"],
                metrics["routing_accuracy"],
            )
            if routing == "auto":
                logger.info(
                    "%s %s route_distribution=%s route_score_mean=%s "
                    "route_margin_mean=%.4f",
                    task_key,
                    scenario,
                    json.dumps(
                        metrics["routing_distribution"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    json.dumps(
                        metrics["routing_score_mean"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    metrics["routing_margin_mean"],
                )
    result_path = output_dir / "test_{}_stage_{:02d}.json".format(
        routing, int(checkpoint["stage"])
    )
    with result_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    return results
