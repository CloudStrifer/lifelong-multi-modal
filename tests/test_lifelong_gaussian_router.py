import logging
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from config import cfg
from engine.lifelong_trainer import fit_gaussian_fingerprint
from modeling.lifelong_model import (
    MODALITIES,
    LifelongMDReID,
    TaskGaussianFingerprint,
)


class _FakeAdapter(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, feature):
        return feature + self.offset


class _FakeVisual(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.lifelong_adapters = nn.ModuleDict()
        self.feature_dim = feature_dim

    def add_lifelong_task(
        self, task_key, bottleneck_dim, dropout=0.0, init_from=None
    ):
        del bottleneck_dim, dropout, init_from
        self.lifelong_adapters[task_key] = nn.ModuleDict({
            modality: _FakeAdapter(self.feature_dim)
            for modality in MODALITIES
        })


class _FakeBackbone(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.base = _FakeVisual(feature_dim)
        self.projection = nn.Linear(3, feature_dim, bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.weight[:3, :3].copy_(torch.eye(3))

    def forward(
        self,
        image,
        cam_label=None,
        view_label=None,
        modality=None,
        task_key=None,
    ):
        del cam_label, view_label
        feature = self.projection(image.mean(dim=(2, 3)))
        if task_key is not None:
            feature = self.base.lifelong_adapters[task_key][modality](feature)
        return torch.stack([feature, feature], dim=1), feature


def _fingerprint_batches(rgb_vectors):
    batches = []
    for vectors in rgb_vectors:
        image = vectors[:, :, None, None].expand(-1, -1, 4, 4).contiguous()
        images = {modality: image.clone() for modality in MODALITIES}
        count = image.shape[0]
        batches.append((
            images,
            torch.zeros(count, dtype=torch.long),
            torch.zeros(count, dtype=torch.long),
            torch.zeros(count, dtype=torch.long),
            tuple("sample" for _ in range(count)),
        ))
    return batches


class GaussianFingerprintRouterUnitTest(unittest.TestCase):
    def _build_model(self):
        test_cfg = cfg.clone()
        test_cfg.merge_from_file(
            "configs/lifelong/MDReID_GaussianFingerprint.yml"
        )
        test_cfg.MODEL.DEVICE = "cpu"
        with patch(
            "modeling.lifelong_model.build_transformer",
            return_value=_FakeBackbone(),
        ):
            model = LifelongMDReID(test_cfg)
        return test_cfg, model

    def test_bank_statistics_are_buffers_and_round_trip(self):
        bank = TaskGaussianFingerprint(feature_dim=4)
        mean = torch.randn(3, 4)
        variance = torch.rand(3, 4) + 0.1
        median = torch.rand(3)
        iqr = torch.rand(3) + 0.1
        bank.set_statistics(mean, variance, median, iqr, sample_count=12)
        self.assertTrue(bool(bank.ready.item()))
        self.assertEqual(int(bank.sample_count.item()), 12)
        self.assertEqual(sum(p.numel() for p in bank.parameters()), 0)

        restored = TaskGaussianFingerprint(feature_dim=4)
        restored.load_state_dict(bank.state_dict())
        self.assertTrue(torch.equal(restored.mean, mean))
        self.assertTrue(torch.equal(restored.variance, variance))
        self.assertTrue(bool(restored.ready.item()))

    def test_fit_freezes_old_statistics_and_routes_all_scenarios(self):
        test_cfg, model = self._build_model()
        model.register_task("person_a", 2, "person")
        model.register_task("person_b", 2, "person")
        model.register_task("vehicle_a", 2, "vehicle")
        logger = logging.getLogger("gaussian-router-test")

        domains = {
            "person_a": torch.tensor([
                [1.0, 0.01, 0.00],
                [1.0, 0.00, 0.01],
                [1.0, 0.02, 0.00],
                [1.0, 0.00, 0.02],
            ]),
            "person_b": torch.tensor([
                [0.01, 1.0, 0.00],
                [0.00, 1.0, 0.01],
                [0.02, 1.0, 0.00],
                [0.00, 1.0, 0.02],
            ]),
            "vehicle_a": torch.tensor([
                [0.01, 0.00, 1.0],
                [0.00, 0.01, 1.0],
                [0.02, 0.00, 1.0],
                [0.00, 0.02, 1.0],
            ]),
        }
        first_summary = fit_gaussian_fingerprint(
            cfg=test_cfg,
            model=model,
            task_key="person_a",
            fingerprint_loader=_fingerprint_batches([domains["person_a"]]),
            device=torch.device("cpu"),
            logger=logger,
        )
        self.assertEqual(first_summary["retained_samples"], 0)
        old_mean = model.task_gaussian_fingerprints["person_a"].mean.clone()
        for task_key in ("person_b", "vehicle_a"):
            fit_gaussian_fingerprint(
                cfg=test_cfg,
                model=model,
                task_key=task_key,
                fingerprint_loader=_fingerprint_batches([domains[task_key]]),
                device=torch.device("cpu"),
                logger=logger,
            )
        self.assertTrue(torch.equal(
            old_mean, model.task_gaussian_fingerprints["person_a"].mean
        ))

        query = domains["person_a"][:2, :, None, None].expand(
            -1, -1, 4, 4
        ).contiguous()
        images = {modality: query.clone() for modality in MODALITIES}
        outputs = model.extract_scenarios(
            images=images,
            scenarios=("RNT", "R", "N", "T"),
            routing="auto",
            candidate_tasks=model.task_order,
            object_category="person",
        )
        expected = model.task_order.index("person_a")
        vehicle_index = model.task_order.index("vehicle_a")
        for _, selected, scores in outputs.values():
            self.assertTrue(bool((selected == expected).all()))
            self.assertTrue(bool(
                (
                    scores[:, vehicle_index]
                    == torch.finfo(scores.dtype).min
                ).all()
            ))


if __name__ == "__main__":
    unittest.main()
