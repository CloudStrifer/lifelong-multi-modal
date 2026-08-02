"""Task-modality adapter bank and task-agnostic routing for lifelong MDReID."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.meta_arch import build_transformer


MODALITIES = ("R", "N", "T")
SCENARIO_MODALITIES = {
    "RNT": ("R", "N", "T"),
    "R": ("R",),
    "N": ("N",),
    "T": ("T",),
}


class TaskIdentityHeads(nn.Module):
    """Task-private identity classifiers used by the retrieval objective."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.classifiers = nn.ModuleDict({
            modality: nn.Linear(feature_dim, num_classes, bias=False)
            for modality in MODALITIES
        })
        for classifier in self.classifiers.values():
            nn.init.normal_(classifier.weight, std=0.001)

    def forward(self, modality: str, feature: torch.Tensor) -> torch.Tensor:
        return self.classifiers[modality](feature)


class TaskModalityKeyBank(nn.Module):
    """Small learnable domain keys retained as model parameters, not replay data."""

    def __init__(self, feature_dim: int, keys_per_modality: int):
        super().__init__()
        if keys_per_modality < 1:
            raise ValueError("keys_per_modality must be positive.")
        self.feature_dim = int(feature_dim)
        self.keys_per_modality = int(keys_per_modality)
        self.keys = nn.Parameter(
            torch.empty(len(MODALITIES), self.keys_per_modality, self.feature_dim)
        )
        nn.init.normal_(self.keys, std=0.02)
        self.register_buffer(
            "calibration_median",
            torch.zeros(len(MODALITIES), dtype=torch.float32),
        )
        self.register_buffer(
            "calibration_iqr",
            torch.ones(len(MODALITIES), dtype=torch.float32),
        )
        self.register_buffer("calibration_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("calibration_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("feature_initialized", torch.zeros((), dtype=torch.bool))

    def for_modality(self, modality: str) -> torch.Tensor:
        try:
            modality_index = MODALITIES.index(str(modality).upper())
        except ValueError as error:
            raise KeyError("Unknown modality: {}".format(modality)) from error
        return self.keys[modality_index]

    @staticmethod
    def modality_index(modality: str) -> int:
        try:
            return MODALITIES.index(str(modality).upper())
        except ValueError as error:
            raise KeyError("Unknown modality: {}".format(modality)) from error

    @torch.no_grad()
    def initialize_from_features(
        self,
        features: Dict[str, torch.Tensor],
    ) -> None:
        """Seed keys with diverse frozen features from the first task batch."""
        if bool(self.feature_initialized.item()):
            return
        for modality in MODALITIES:
            feature = F.normalize(features[modality].detach().float(), dim=1)
            if feature.shape[0] == 0:
                raise RuntimeError("Cannot initialize Task Keys from an empty batch.")
            mean_direction = F.normalize(feature.mean(dim=0, keepdim=True), dim=1)
            selected_indices = [
                int((feature @ mean_direction.t()).squeeze(1).argmax().item())
            ]
            while len(selected_indices) < self.keys_per_modality:
                selected = feature[selected_indices]
                maximum_similarity = (feature @ selected.t()).max(dim=1).values
                maximum_similarity[selected_indices] = float("inf")
                next_index = int(maximum_similarity.argmin().item())
                if next_index in selected_indices:
                    next_index = len(selected_indices) % feature.shape[0]
                selected_indices.append(next_index)
            modality_index = self.modality_index(modality)
            self.keys[modality_index].copy_(feature[selected_indices])
        self.feature_initialized.fill_(True)

    def set_calibration(
        self,
        median: torch.Tensor,
        iqr: torch.Tensor,
        sample_count: int,
    ) -> None:
        expected_shape = (len(MODALITIES),)
        if tuple(median.shape) != expected_shape:
            raise ValueError(
                "Task-Key calibration median must have shape {}, got {}.".format(
                    expected_shape, tuple(median.shape)
                )
            )
        if tuple(iqr.shape) != expected_shape:
            raise ValueError(
                "Task-Key calibration IQR must have shape {}, got {}.".format(
                    expected_shape, tuple(iqr.shape)
                )
            )
        if int(sample_count) < 2:
            raise ValueError("At least two samples are required for calibration.")
        if not bool(torch.isfinite(median).all()):
            raise ValueError("Task-Key calibration medians must be finite.")
        if not bool(torch.isfinite(iqr).all()) or bool((iqr <= 0).any()):
            raise ValueError("Task-Key calibration IQR values must be positive.")
        with torch.no_grad():
            self.calibration_median.copy_(
                median.to(
                    device=self.calibration_median.device,
                    dtype=self.calibration_median.dtype,
                )
            )
            self.calibration_iqr.copy_(
                iqr.to(
                    device=self.calibration_iqr.device,
                    dtype=self.calibration_iqr.dtype,
                )
            )
            self.calibration_count.fill_(int(sample_count))
            self.calibration_ready.fill_(True)


class TaskGaussianFingerprint(nn.Module):
    """Frozen per-task diagonal-Gaussian statistics for R/N/T routing."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        shape = (len(MODALITIES), self.feature_dim)
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("variance", torch.ones(shape, dtype=torch.float32))
        self.register_buffer(
            "calibration_median",
            torch.zeros(len(MODALITIES), dtype=torch.float32),
        )
        self.register_buffer(
            "calibration_iqr",
            torch.ones(len(MODALITIES), dtype=torch.float32),
        )
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))

    @staticmethod
    def modality_index(modality: str) -> int:
        try:
            return MODALITIES.index(str(modality).upper())
        except ValueError as error:
            raise KeyError("Unknown modality: {}".format(modality)) from error

    def set_statistics(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        calibration_median: torch.Tensor,
        calibration_iqr: torch.Tensor,
        sample_count: int,
    ) -> None:
        expected_shape = (len(MODALITIES), self.feature_dim)
        if tuple(mean.shape) != expected_shape:
            raise ValueError(
                "Gaussian mean shape must be {}, got {}.".format(
                    expected_shape, tuple(mean.shape)
                )
            )
        if tuple(variance.shape) != expected_shape:
            raise ValueError(
                "Gaussian variance shape must be {}, got {}.".format(
                    expected_shape, tuple(variance.shape)
                )
            )
        if tuple(calibration_median.shape) != (len(MODALITIES),):
            raise ValueError("calibration_median must contain one value per modality.")
        if tuple(calibration_iqr.shape) != (len(MODALITIES),):
            raise ValueError("calibration_iqr must contain one value per modality.")
        if int(sample_count) < 2:
            raise ValueError("At least two samples are required for Gaussian routing.")
        if not bool(torch.isfinite(mean).all()):
            raise ValueError("Gaussian means contain non-finite values.")
        if not bool(torch.isfinite(variance).all()) or bool((variance <= 0).any()):
            raise ValueError("Gaussian variances must be finite and positive.")
        if (
            not bool(torch.isfinite(calibration_iqr).all())
            or bool((calibration_iqr <= 0).any())
        ):
            raise ValueError("Gaussian calibration IQR values must be positive.")
        with torch.no_grad():
            self.mean.copy_(mean.to(device=self.mean.device, dtype=self.mean.dtype))
            self.variance.copy_(
                variance.to(device=self.variance.device, dtype=self.variance.dtype)
            )
            self.calibration_median.copy_(
                calibration_median.to(
                    device=self.calibration_median.device,
                    dtype=self.calibration_median.dtype,
                )
            )
            self.calibration_iqr.copy_(
                calibration_iqr.to(
                    device=self.calibration_iqr.device,
                    dtype=self.calibration_iqr.dtype,
                )
            )
            self.sample_count.fill_(int(sample_count))
            self.ready.fill_(True)

    def score(
        self,
        feature: torch.Tensor,
        modality: str,
        normalize_feature: bool,
    ) -> torch.Tensor:
        """Return a calibrated log-like score; larger means a closer domain."""
        if not bool(self.ready.item()):
            raise RuntimeError("Gaussian fingerprint statistics are not fitted.")
        modality_index = self.modality_index(modality)
        feature = feature.float()
        if normalize_feature:
            feature = F.normalize(feature, dim=1)
        mean = self.mean[modality_index].float()
        variance = self.variance[modality_index].float()
        distance = ((feature - mean).square() / variance).mean(dim=1)
        calibrated_distance = (
            distance - self.calibration_median[modality_index].float()
        ) / self.calibration_iqr[modality_index].float()
        return -calibrated_distance


class LifelongMDReID(nn.Module):
    """One frozen CLIP backbone plus a growing task x modality adapter bank."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.MODEL.TRANSFORMER_TYPE != "ViT-B-16":
            raise ValueError(
                "The lifelong adapter bank currently requires MODEL.TRANSFORMER_TYPE='ViT-B-16'."
            )
        if cfg.MODEL.PROMPT or cfg.MODEL.ADAPTER:
            raise ValueError(
                "Disable the legacy PROMPT/ADAPTER flags when using the lifelong adapter bank."
            )
        if cfg.MODEL.SIE_CAMERA:
            raise ValueError(
                "Set MODEL.SIE_CAMERA=False: camera vocabularies differ across lifelong tasks."
            )
        self.cfg = cfg
        self.feature_dim = 512
        self.add_share = bool(cfg.MODEL.ADD_SHARE)
        self.backbone = build_transformer(
            num_classes=1,
            cfg=cfg,
            camera_num=0,
            view_num=0,
            factory={},
            feat_dim=self.feature_dim,
        )
        self.task_heads = nn.ModuleDict()
        self.task_router_keys = nn.ModuleDict()
        self.task_gaussian_fingerprints = nn.ModuleDict()
        self.task_order: List[str] = []
        self.task_num_classes: Dict[str, int] = {}
        self.task_categories: Dict[str, str] = {}
        self.adapter_rank = int(cfg.LIFELONG.ADAPTER_RANK)
        self.adapter_dropout = float(cfg.LIFELONG.ADAPTER_DROPOUT)
        self.modality_decoupled = bool(
            cfg.LIFELONG.MODALITY_DECOUPLED
        )
        self.adapter_init = str(cfg.LIFELONG.ADAPTER_INIT).lower()
        self.router_method = str(cfg.LIFELONG.ROUTER.METHOD).lower()
        if self.router_method == "gaussian":
            self.router_method = "gaussian_fingerprint"
        if self.router_method not in ("task_key", "gaussian_fingerprint", "legacy"):
            raise ValueError(
                "LIFELONG.ROUTER.METHOD must be 'task_key', "
                "'gaussian_fingerprint', or 'legacy', got {}".format(
                    self.router_method
                )
            )
        self.category_aware_routing = bool(cfg.LIFELONG.ROUTER.CATEGORY_AWARE)
        self.consistency_weight = float(cfg.LIFELONG.ROUTER.CONSISTENCY_WEIGHT)
        self.confidence_weight = float(cfg.LIFELONG.ROUTER.CONFIDENCE_WEIGHT)
        self.keys_per_modality = int(cfg.LIFELONG.ROUTER.KEYS_PER_MODALITY)
        self.key_temperature = float(cfg.LIFELONG.ROUTER.KEY_TEMPERATURE)
        if self.key_temperature <= 0:
            raise ValueError("LIFELONG.ROUTER.KEY_TEMPERATURE must be positive.")
        self.key_weight = float(cfg.LIFELONG.ROUTER.KEY_WEIGHT)
        self.task_key_calibration = bool(
            cfg.LIFELONG.ROUTER.TASK_KEY_CALIBRATION
        )
        self.task_key_feature_initialization = bool(
            cfg.LIFELONG.ROUTER.TASK_KEY_FEATURE_INITIALIZATION
        )
        self.aux_consistency_weight = float(
            cfg.LIFELONG.ROUTER.AUX_CONSISTENCY_WEIGHT
        )
        self.aux_confidence_weight = float(
            cfg.LIFELONG.ROUTER.AUX_CONFIDENCE_WEIGHT
        )
        self.router_positive_weight = float(cfg.LIFELONG.ROUTER.POSITIVE_WEIGHT)
        self.router_margin_weight = float(cfg.LIFELONG.ROUTER.MARGIN_WEIGHT)
        self.router_separation_weight = float(
            cfg.LIFELONG.ROUTER.SEPARATION_WEIGHT
        )
        self.router_diversity_weight = float(cfg.LIFELONG.ROUTER.DIVERSITY_WEIGHT)
        self.router_margin = float(cfg.LIFELONG.ROUTER.MARGIN)
        self.router_separation_margin = float(
            cfg.LIFELONG.ROUTER.SEPARATION_MARGIN
        )
        self.router_diversity_margin = float(cfg.LIFELONG.ROUTER.DIVERSITY_MARGIN)
        self.gaussian_normalize_features = bool(
            cfg.LIFELONG.ROUTER.GAUSSIAN_NORMALIZE_FEATURES
        )
        self.router_keys_ready = True
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    @property
    def visual(self):
        return self.backbone.base

    def register_task(
        self,
        task_key: str,
        num_classes: int,
        object_category: str,
        initialize_from_history: bool = True,
    ) -> None:
        object_category = str(object_category).strip().lower()
        if object_category not in ("person", "vehicle"):
            raise ValueError(
                "object_category must be 'person' or 'vehicle', got {}".format(
                    object_category
                )
            )
        if task_key in self.task_heads:
            if self.task_num_classes[task_key] != int(num_classes):
                raise ValueError("Task {} was registered with a different class count.".format(task_key))
            if self.task_categories[task_key] != object_category:
                raise ValueError(
                    "Task {} was registered with a different object category.".format(
                        task_key
                    )
                )
            return
        if "." in task_key:
            raise ValueError("Task keys cannot contain '.': {}".format(task_key))
        source_tasks: List[str] = []
        if initialize_from_history and self.task_order:
            if self.adapter_init == "mean":
                source_tasks = list(self.task_order)
            elif self.adapter_init == "latest":
                source_tasks = [self.task_order[-1]]
            elif self.adapter_init not in ("zero", "random"):
                raise ValueError("Unknown LIFELONG.ADAPTER_INIT: {}".format(self.adapter_init))
        self.visual.add_lifelong_task(
            task_key=task_key,
            bottleneck_dim=self.adapter_rank,
            dropout=self.adapter_dropout,
            init_from=source_tasks,
            modality_decoupled=self.modality_decoupled,
        )
        self.task_heads[task_key] = TaskIdentityHeads(self.feature_dim, num_classes)
        if self.router_method == "task_key":
            self.task_router_keys[task_key] = TaskModalityKeyBank(
                feature_dim=self.feature_dim,
                keys_per_modality=self.keys_per_modality,
            )
        elif self.router_method == "gaussian_fingerprint":
            self.task_gaussian_fingerprints[task_key] = TaskGaussianFingerprint(
                feature_dim=self.feature_dim
            )
        self.task_order.append(task_key)
        self.task_num_classes[task_key] = int(num_classes)
        self.task_categories[task_key] = object_category
        device = next(self.backbone.parameters()).device
        self.to(device)
        self.set_trainable_task(task_key)

    def set_trainable_task(self, task_key: str) -> None:
        if task_key not in self.task_heads:
            raise KeyError("Unregistered task: {}".format(task_key))
        for parameter in self.parameters():
            parameter.requires_grad = False
        adapter_marker = ".lifelong_adapters.{}.".format(task_key)
        for name, parameter in self.named_parameters():
            if (
                adapter_marker in name
                or name.startswith("task_heads.{}.".format(task_key))
                or name.startswith("task_router_keys.{}.".format(task_key))
            ):
                parameter.requires_grad = True

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _encode_task(
        self,
        images: Dict[str, torch.Tensor],
        task_key: str,
        modalities: Sequence[str],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        encoded: Dict[str, Dict[str, torch.Tensor]] = {}
        for modality in modalities:
            tokens, private_feature = self.backbone(
                images[modality],
                cam_label=None,
                view_label=None,
                modality=modality,
                task_key=task_key,
            )
            shared_feature = tokens[:, 0] if self.add_share else private_feature
            descriptor = torch.cat([private_feature, shared_feature], dim=1)
            logits = self.task_heads[task_key](modality, private_feature)
            encoded[modality] = {
                "private": private_feature,
                "shared": shared_feature,
                "descriptor": descriptor,
                "logits": logits,
            }
        return encoded

    @staticmethod
    def _compose_descriptor(
        encoded: Dict[str, Dict[str, torch.Tensor]],
        modalities: Sequence[str],
    ) -> torch.Tensor:
        return torch.cat([encoded[modality]["descriptor"] for modality in modalities], dim=1)

    def forward_train(self, images: Dict[str, torch.Tensor], task_key: str):
        router_features = (
            self._encode_router_features(images, MODALITIES)
            if self.router_method == "task_key"
            else None
        )
        if (
            self.router_method == "task_key"
            and self.task_key_feature_initialization
            and not bool(
                self.task_router_keys[task_key].feature_initialized.item()
            )
        ):
            self.task_router_keys[task_key].initialize_from_features(
                router_features
            )
        encoded = self._encode_task(images, task_key, MODALITIES)
        output = {
            "modalities": encoded,
            "fused": self._compose_descriptor(encoded, MODALITIES),
        }
        if self.router_method == "task_key":
            output["router_losses"] = self._router_training_losses(
                router_features=router_features,
                task_key=task_key,
            )
        return output

    def _encode_router_features(
        self,
        images: Dict[str, torch.Tensor],
        modalities: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        """Extract task-independent frozen CLIP features for domain routing."""
        features: Dict[str, torch.Tensor] = {}
        # The shared backbone is frozen. Detaching here prevents the routing
        # objective from becoming another representation-learning shortcut.
        # Router features must also be deterministic: otherwise train-mode
        # stochastic depth changes the feature distribution that is later
        # calibrated and evaluated in model.eval().
        backbone_was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                for modality in modalities:
                    _, private_feature = self.backbone(
                        images[modality],
                        cam_label=None,
                        view_label=None,
                        modality=modality,
                        task_key=None,
                    )
                    features[modality] = private_feature.detach()
        finally:
            self.backbone.train(backbone_was_training)
        return features

    def set_gaussian_fingerprint(
        self,
        task_key: str,
        mean: torch.Tensor,
        variance: torch.Tensor,
        calibration_median: torch.Tensor,
        calibration_iqr: torch.Tensor,
        sample_count: int,
    ) -> None:
        if self.router_method != "gaussian_fingerprint":
            raise RuntimeError(
                "Gaussian fingerprints require "
                "LIFELONG.ROUTER.METHOD='gaussian_fingerprint'."
            )
        if task_key not in self.task_gaussian_fingerprints:
            raise KeyError("Unregistered Gaussian task: {}".format(task_key))
        self.task_gaussian_fingerprints[task_key].set_statistics(
            mean=mean,
            variance=variance,
            calibration_median=calibration_median,
            calibration_iqr=calibration_iqr,
            sample_count=sample_count,
        )

    def _gaussian_score_matrix(
        self,
        router_features: Dict[str, torch.Tensor],
        modalities: Sequence[str],
        candidates: Sequence[str],
    ) -> torch.Tensor:
        missing = [
            task_key
            for task_key in candidates
            if (
                task_key not in self.task_gaussian_fingerprints
                or not bool(
                    self.task_gaussian_fingerprints[task_key].ready.item()
                )
            )
        ]
        if missing:
            raise RuntimeError(
                "Gaussian fingerprint statistics are missing for tasks {}. "
                "Train the sequence with the Gaussian router or use oracle "
                "routing for an older checkpoint.".format(missing)
            )
        modality_scores = []
        for modality in modalities:
            modality_scores.append(torch.stack([
                self.task_gaussian_fingerprints[task_key].score(
                    feature=router_features[modality],
                    modality=modality,
                    normalize_feature=self.gaussian_normalize_features,
                )
                for task_key in candidates
            ], dim=1))
        return torch.stack(modality_scores, dim=2).mean(dim=2)

    def _task_modality_key_score(
        self,
        feature: torch.Tensor,
        task_key: str,
        modality: str,
        calibrated: bool = False,
    ) -> torch.Tensor:
        feature = F.normalize(feature.float(), dim=1)
        keys = F.normalize(
            self.task_router_keys[task_key].for_modality(modality).float(),
            dim=1,
        )
        similarities = feature @ keys.t()
        if similarities.shape[1] == 1:
            score = similarities[:, 0]
        else:
            # Smooth max with log(K) correction keeps scores comparable when
            # the number of keys changes and lets every key receive a gradient.
            score = self.key_temperature * (
                torch.logsumexp(similarities / self.key_temperature, dim=1)
                - math.log(similarities.shape[1])
            )
        if calibrated and self.task_key_calibration:
            bank = self.task_router_keys[task_key]
            if not bool(bank.calibration_ready.item()):
                raise RuntimeError(
                    "Task-Key score calibration is missing for task '{}'. "
                    "Retrain with the calibrated Task-Key router, disable "
                    "LIFELONG.ROUTER.TASK_KEY_CALIBRATION for a raw-score "
                    "ablation, or use oracle routing.".format(task_key)
                )
            modality_index = bank.modality_index(modality)
            score = (
                score - bank.calibration_median[modality_index].float()
            ) / bank.calibration_iqr[modality_index].float()
        return score

    def _task_key_modality_score_matrices(
        self,
        router_features: Dict[str, torch.Tensor],
        modalities: Sequence[str],
        candidates: Sequence[str],
        calibrated: bool,
    ) -> Dict[str, torch.Tensor]:
        if not self.router_keys_ready:
            raise RuntimeError(
                "This checkpoint has no trained Task-Key router. Retrain with "
                "LIFELONG.ROUTER.METHOD='task_key', or evaluate the legacy "
                "checkpoint with LIFELONG.ROUTER.METHOD='legacy'."
            )
        return {
            modality: torch.stack([
                self._task_modality_key_score(
                    router_features[modality],
                    task_key,
                    modality,
                    calibrated=calibrated,
                )
                for task_key in candidates
            ], dim=1)
            for modality in modalities
        }

    def _task_key_score_matrix(
        self,
        router_features: Dict[str, torch.Tensor],
        modalities: Sequence[str],
        candidates: Sequence[str],
    ) -> torch.Tensor:
        modality_matrices = self._task_key_modality_score_matrices(
            router_features=router_features,
            modalities=modalities,
            candidates=candidates,
            calibrated=True,
        )
        return torch.stack(
            [modality_matrices[modality] for modality in modalities],
            dim=2,
        ).mean(dim=2)

    def _category_candidate_indices(
        self,
        candidates: Sequence[str],
        object_category: Optional[str],
    ) -> List[int]:
        if not self.category_aware_routing:
            return list(range(len(candidates)))
        candidate_categories = {
            self.task_categories[task_key] for task_key in candidates
        }
        if object_category is None:
            if len(candidate_categories) == 1:
                return list(range(len(candidates)))
            raise ValueError(
                "object_category is required for category-aware routing when "
                "person and vehicle task banks are both candidates."
            )
        object_category = str(object_category).strip().lower()
        allowed = [
            index
            for index, task_key in enumerate(candidates)
            if self.task_categories[task_key] == object_category
        ]
        if not allowed:
            raise RuntimeError(
                "No {} task bank is available among candidates {}.".format(
                    object_category, list(candidates)
                )
            )
        return allowed

    def _mask_scores_by_category(
        self,
        scores: torch.Tensor,
        candidates: Sequence[str],
        object_category: Optional[str],
    ) -> torch.Tensor:
        allowed = self._category_candidate_indices(candidates, object_category)
        if len(allowed) == len(candidates):
            return scores
        masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
        masked[:, allowed] = scores[:, allowed]
        return masked

    @staticmethod
    def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _key_diversity_loss(self, task_key: str) -> torch.Tensor:
        keys = F.normalize(self.task_router_keys[task_key].keys.float(), dim=2)
        if keys.shape[1] <= 1:
            return self._zero_loss(keys)
        losses = []
        upper = torch.triu_indices(
            keys.shape[1], keys.shape[1], offset=1, device=keys.device
        )
        for modality_index in range(keys.shape[0]):
            similarity = keys[modality_index] @ keys[modality_index].t()
            pair_similarity = similarity[upper[0], upper[1]]
            losses.append(
                F.relu(pair_similarity - self.router_diversity_margin).mean()
            )
        return torch.stack(losses).mean()

    def _key_separation_loss(
        self,
        task_key: str,
        old_tasks: Sequence[str],
    ) -> torch.Tensor:
        current_keys = F.normalize(
            self.task_router_keys[task_key].keys.float(), dim=2
        )
        if not old_tasks:
            return self._zero_loss(current_keys)
        losses = []
        for old_task in old_tasks:
            old_keys = F.normalize(
                self.task_router_keys[old_task].keys.detach().float(), dim=2
            )
            for modality_index in range(len(MODALITIES)):
                cross_similarity = (
                    current_keys[modality_index]
                    @ old_keys[modality_index].t()
                )
                losses.append(
                    F.relu(
                        cross_similarity - self.router_separation_margin
                    ).mean()
                )
        return torch.stack(losses).mean()

    def _router_training_losses(
        self,
        router_features: Dict[str, torch.Tensor],
        task_key: str,
    ) -> Dict[str, torch.Tensor]:
        candidates = list(self.task_order)
        modality_score_matrices = self._task_key_modality_score_matrices(
            router_features=router_features,
            modalities=MODALITIES,
            candidates=candidates,
            calibrated=False,
        )
        scores = torch.stack(
            [
                modality_score_matrices[modality]
                for modality in MODALITIES
            ],
            dim=2,
        ).mean(dim=2)
        current_index = candidates.index(task_key)
        current_score = scores[:, current_index]
        positive_loss = torch.stack([
            (1.0 - modality_score_matrices[modality][:, current_index]).mean()
            for modality in MODALITIES
        ]).mean()

        current_category = self.task_categories[task_key]
        old_tasks = [
            candidate
            for candidate in candidates[:current_index]
            if (
                not self.category_aware_routing
                or self.task_categories[candidate] == current_category
            )
        ]
        if old_tasks:
            old_indices = [candidates.index(candidate) for candidate in old_tasks]
            margin_views = [
                modality_score_matrices[modality]
                for modality in MODALITIES
            ] + [scores]
            margin_loss = torch.stack([
                F.relu(
                    self.router_margin
                    - view_scores[:, current_index]
                    + view_scores[:, old_indices].max(dim=1).values
                ).mean()
                for view_scores in margin_views
            ]).mean()
        else:
            margin_loss = self._zero_loss(current_score)

        separation_loss = self._key_separation_loss(task_key, old_tasks)
        diversity_loss = self._key_diversity_loss(task_key)
        routed_scores = self._mask_scores_by_category(
            scores=scores,
            candidates=candidates,
            object_category=current_category,
        )
        routing_accuracy = (
            routed_scores.argmax(dim=1) == current_index
        ).float().mean()
        modality_accuracies = {}
        for modality in MODALITIES:
            routed_modality_scores = self._mask_scores_by_category(
                scores=modality_score_matrices[modality],
                candidates=candidates,
                object_category=current_category,
            )
            modality_accuracies[modality] = (
                routed_modality_scores.argmax(dim=1) == current_index
            ).float().mean()
        total = (
            self.router_positive_weight * positive_loss
            + self.router_margin_weight * margin_loss
            + self.router_separation_weight * separation_loss
            + self.router_diversity_weight * diversity_loss
        )
        return {
            "total": total,
            "positive": positive_loss,
            "margin": margin_loss,
            "separation": separation_loss,
            "diversity": diversity_loss,
            "accuracy": routing_accuracy,
            **{
                "accuracy_{}".format(modality): accuracy
                for modality, accuracy in modality_accuracies.items()
            },
        }

    @staticmethod
    def _normalized_classifier_confidence(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] <= 1:
            return torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
        probabilities = F.softmax(logits.float(), dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        return (1.0 - entropy / math.log(logits.shape[1])).to(dtype=logits.dtype)

    def _legacy_routing_components(
        self,
        encoded: Dict[str, Dict[str, torch.Tensor]],
        modalities: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        confidence = torch.stack([
            self._normalized_classifier_confidence(encoded[modality]["logits"])
            for modality in modalities
        ]).mean(dim=0).float()
        if len(modalities) < 2:
            return torch.zeros_like(confidence), confidence
        pair_scores = []
        for left, right in combinations(modalities, 2):
            left_feature = F.normalize(encoded[left]["descriptor"].float(), dim=1)
            right_feature = F.normalize(encoded[right]["descriptor"].float(), dim=1)
            pair_scores.append((left_feature * right_feature).sum(dim=1))
        consistency = torch.stack(pair_scores).mean(dim=0)
        return consistency, confidence

    def _auto_score_matrix(
        self,
        encoded_candidates: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        router_features: Optional[Dict[str, torch.Tensor]],
        modalities: Sequence[str],
        candidates: Sequence[str],
        object_category: Optional[str],
    ) -> torch.Tensor:
        if self.router_method == "gaussian_fingerprint":
            if router_features is None:
                raise RuntimeError(
                    "Gaussian fingerprint routing requires adapter-free features."
                )
            scores = self._gaussian_score_matrix(
                router_features=router_features,
                modalities=modalities,
                candidates=candidates,
            )
            return self._mask_scores_by_category(
                scores=scores,
                candidates=candidates,
                object_category=object_category,
            )

        consistency_columns = []
        confidence_columns = []
        for task_key in candidates:
            consistency, confidence = self._legacy_routing_components(
                encoded_candidates[task_key], modalities
            )
            consistency_columns.append(consistency)
            confidence_columns.append(confidence)
        consistency_scores = torch.stack(consistency_columns, dim=1)
        confidence_scores = torch.stack(confidence_columns, dim=1)

        if self.router_method == "legacy":
            if len(modalities) < 2:
                scores = confidence_scores
            else:
                scores = (
                    self.consistency_weight * consistency_scores
                    + self.confidence_weight * confidence_scores
                )
        else:
            if router_features is None:
                raise RuntimeError("Task-Key routing requires router features.")
            key_scores = self._task_key_score_matrix(
                router_features=router_features,
                modalities=modalities,
                candidates=candidates,
            )
            scores = (
                self.key_weight * key_scores
                + self.aux_consistency_weight * consistency_scores
                + self.aux_confidence_weight * confidence_scores
            )
        return self._mask_scores_by_category(
            scores=scores,
            candidates=candidates,
            object_category=object_category,
        )

    def extract(
        self,
        images: Dict[str, torch.Tensor],
        scenario: str,
        routing: str = "auto",
        oracle_task: Optional[str] = None,
        candidate_tasks: Optional[Sequence[str]] = None,
        object_category: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a descriptor and return selected bank indices and router scores."""
        scenario = scenario.upper()
        return self.extract_scenarios(
            images=images,
            scenarios=[scenario],
            routing=routing,
            oracle_task=oracle_task,
            candidate_tasks=candidate_tasks,
            object_category=object_category,
        )[scenario]

    def extract_scenarios(
        self,
        images: Dict[str, torch.Tensor],
        scenarios: Sequence[str],
        routing: str = "auto",
        oracle_task: Optional[str] = None,
        candidate_tasks: Optional[Sequence[str]] = None,
        object_category: Optional[str] = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Extract several scenarios while encoding each task/modality only once."""
        normalized_scenarios = [str(scenario).upper() for scenario in scenarios]
        for scenario in normalized_scenarios:
            if scenario not in SCENARIO_MODALITIES:
                raise KeyError("Scenario {} is not enabled.".format(scenario))
        required_modalities = tuple(
            modality
            for modality in MODALITIES
            if any(modality in SCENARIO_MODALITIES[scenario] for scenario in normalized_scenarios)
        )
        candidates = list(candidate_tasks or self.task_order)
        if not candidates:
            raise RuntimeError("No task adapters have been registered.")
        unknown = [task for task in candidates if task not in self.task_heads]
        if unknown:
            raise KeyError("Unknown candidate tasks: {}".format(unknown))
        if routing == "oracle":
            if oracle_task is None:
                raise ValueError("oracle_task is required for oracle routing.")
            encoded = self._encode_task(images, oracle_task, required_modalities)
            oracle_index = candidates.index(oracle_task)
            outputs = {}
            for scenario in normalized_scenarios:
                modalities = SCENARIO_MODALITIES[scenario]
                descriptor = self._compose_descriptor(encoded, modalities)
                selected = torch.full(
                    (descriptor.shape[0],),
                    oracle_index,
                    dtype=torch.long,
                    device=descriptor.device,
                )
                scores = torch.zeros(
                    descriptor.shape[0], len(candidates), device=descriptor.device
                )
                scores[:, oracle_index] = 1.0
                outputs[scenario] = descriptor, selected, scores
            return outputs
        if routing != "auto":
            raise ValueError("routing must be 'auto' or 'oracle', got {}".format(routing))

        encoded_candidates = {
            task_key: self._encode_task(images, task_key, required_modalities)
            for task_key in candidates
        }
        router_features = (
            self._encode_router_features(images, required_modalities)
            if self.router_method in ("task_key", "gaussian_fingerprint")
            else None
        )
        outputs = {}
        for scenario in normalized_scenarios:
            modalities = SCENARIO_MODALITIES[scenario]
            descriptor_stack = torch.stack([
                self._compose_descriptor(encoded_candidates[task_key], modalities)
                for task_key in candidates
            ])
            scenario_router_features = (
                {
                    modality: router_features[modality]
                    for modality in modalities
                }
                if router_features is not None
                else None
            )
            score_stack = self._auto_score_matrix(
                encoded_candidates=encoded_candidates,
                router_features=scenario_router_features,
                modalities=modalities,
                candidates=candidates,
                object_category=object_category,
            )
            selected = score_stack.argmax(dim=1)
            batch_indices = torch.arange(score_stack.shape[0], device=score_stack.device)
            outputs[scenario] = (
                descriptor_stack[selected, batch_indices],
                selected,
                score_stack,
            )
        return outputs

    def parameter_accounting(self) -> Dict[str, object]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        adapter_counts = {}
        head_counts = {}
        router_key_counts = {}
        router_key_buffer_counts = {}
        gaussian_buffer_counts = {}
        for task_key in self.task_order:
            marker = ".lifelong_adapters.{}.".format(task_key)
            adapter_counts[task_key] = sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if marker in name
            )
            head_counts[task_key] = sum(
                parameter.numel() for parameter in self.task_heads[task_key].parameters()
            )
            router_key_counts[task_key] = (
                sum(
                    parameter.numel()
                    for parameter in self.task_router_keys[task_key].parameters()
                )
                if task_key in self.task_router_keys
                else 0
            )
            router_key_buffer_counts[task_key] = (
                sum(
                    buffer.numel()
                    for buffer in self.task_router_keys[task_key].buffers()
                )
                if task_key in self.task_router_keys
                else 0
            )
            gaussian_buffer_counts[task_key] = (
                sum(
                    buffer.numel()
                    for buffer in self.task_gaussian_fingerprints[
                        task_key
                    ].buffers()
                )
                if task_key in self.task_gaussian_fingerprints
                else 0
            )
        adapter_total = sum(adapter_counts.values())
        head_total = sum(head_counts.values())
        router_key_total = sum(router_key_counts.values())
        router_key_buffer_total = sum(router_key_buffer_counts.values())
        gaussian_buffer_total = sum(gaussian_buffer_counts.values())
        return {
            "modality_decoupled": self.modality_decoupled,
            "total": total,
            "trainable_current": trainable,
            "frozen_shared_and_other": total - trainable,
            "adapter_total": adapter_total,
            "head_total": head_total,
            "router_key_total": router_key_total,
            "router_key_buffer_total": router_key_buffer_total,
            "gaussian_fingerprint_buffer_total": gaussian_buffer_total,
            "shared_backbone_and_tokens": (
                total - adapter_total - head_total - router_key_total
            ),
            "adapter_per_task": adapter_counts,
            "head_per_task": head_counts,
            "router_key_per_task": router_key_counts,
            "router_key_buffer_per_task": router_key_buffer_counts,
            "router_key_buffer_fp32_megabytes": (
                router_key_buffer_total * 4 / (1024 ** 2)
            ),
            "gaussian_fingerprint_buffer_per_task": gaussian_buffer_counts,
            "gaussian_fingerprint_fp32_megabytes": (
                gaussian_buffer_total * 4 / (1024 ** 2)
            ),
            "fp32_megabytes": total * 4 / (1024 ** 2),
            "fp16_megabytes": total * 2 / (1024 ** 2),
        }
