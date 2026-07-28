import logging
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from config import cfg
from data.lifelong_datasets import dataset_object_category
from engine.lifelong_trainer import fit_task_key_calibration
from modeling.lifelong_model import (
    MODALITIES,
    LifelongMDReID,
    TaskModalityKeyBank,
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
        self.feature_dim = feature_dim
        self.lifelong_adapters = nn.ModuleDict()

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

    def forward(
        self,
        image,
        cam_label=None,
        view_label=None,
        modality=None,
        task_key=None,
    ):
        del cam_label, view_label
        pooled = image.mean(dim=(2, 3))
        feature = self.projection(pooled)
        if task_key is not None:
            feature = self.base.lifelong_adapters[task_key][modality](feature)
        tokens = torch.stack([feature, feature], dim=1)
        return tokens, feature


class TaskKeyRouterUnitTest(unittest.TestCase):
    def test_dataset_categories(self):
        self.assertEqual(dataset_object_category("RGBNT201"), "person")
        self.assertEqual(dataset_object_category("Market-MM"), "person")
        self.assertEqual(dataset_object_category("RGBNT100"), "vehicle")
        self.assertEqual(dataset_object_category("MSVR310"), "vehicle")
        self.assertEqual(dataset_object_category("WMVeID863"), "vehicle")

    def test_task_key_shape_and_gradients(self):
        bank = TaskModalityKeyBank(feature_dim=16, keys_per_modality=4)
        self.assertEqual(tuple(bank.keys.shape), (len(MODALITIES), 4, 16))
        feature = torch.randn(3, 16)
        keys = bank.for_modality("R")
        score = (
            torch.nn.functional.normalize(feature, dim=1)
            @ torch.nn.functional.normalize(keys, dim=1).t()
        ).mean()
        score.backward()
        self.assertIsNotNone(bank.keys.grad)
        self.assertGreater(float(bank.keys.grad.abs().sum()), 0.0)

    def test_unknown_modality_is_rejected(self):
        bank = TaskModalityKeyBank(feature_dim=8, keys_per_modality=1)
        with self.assertRaises(KeyError):
            bank.for_modality("X")

    def test_score_calibration_removes_task_baseline_bias(self):
        test_cfg = cfg.clone()
        test_cfg.merge_from_file("configs/lifelong/MDReID_TMDA_CSCR.yml")
        with patch(
            "modeling.lifelong_model.build_transformer",
            return_value=_FakeBackbone(),
        ):
            model = LifelongMDReID(test_cfg)
        model.register_task("person_a", 2, "person")
        model.register_task("person_b", 2, "person")
        query = torch.zeros(2, model.feature_dim)
        query[:, 0] = 1.0
        key_a = torch.zeros(model.feature_dim)
        key_a[0], key_a[1] = 0.8, 0.6
        key_b = torch.zeros(model.feature_dim)
        key_b[0], key_b[1] = 0.9, 0.4358899
        with torch.no_grad():
            for modality_index in range(len(MODALITIES)):
                model.task_router_keys["person_a"].keys[modality_index].copy_(
                    key_a.expand(model.keys_per_modality, -1)
                )
                model.task_router_keys["person_b"].keys[modality_index].copy_(
                    key_b.expand(model.keys_per_modality, -1)
                )
        model.task_router_keys["person_a"].set_calibration(
            median=torch.full((len(MODALITIES),), 0.8),
            iqr=torch.full((len(MODALITIES),), 0.1),
            sample_count=10,
        )
        model.task_router_keys["person_b"].set_calibration(
            median=torch.full((len(MODALITIES),), 0.95),
            iqr=torch.full((len(MODALITIES),), 0.1),
            sample_count=10,
        )
        raw_a = model._task_modality_key_score(
            query, "person_a", "R", calibrated=False
        )
        raw_b = model._task_modality_key_score(
            query, "person_b", "R", calibrated=False
        )
        calibrated_a = model._task_modality_key_score(
            query, "person_a", "R", calibrated=True
        )
        calibrated_b = model._task_modality_key_score(
            query, "person_b", "R", calibrated=True
        )
        self.assertTrue(bool((raw_b > raw_a).all()))
        self.assertTrue(bool((calibrated_a > calibrated_b).all()))

    def test_current_task_calibration_retains_no_samples(self):
        test_cfg = cfg.clone()
        test_cfg.merge_from_file("configs/lifelong/MDReID_TMDA_CSCR.yml")
        test_cfg.MODEL.DEVICE = "cpu"
        with patch(
            "modeling.lifelong_model.build_transformer",
            return_value=_FakeBackbone(),
        ):
            model = LifelongMDReID(test_cfg)
        model.register_task("person_a", 3, "person")
        images = {
            modality: torch.randn(6, 3, 8, 8)
            for modality in MODALITIES
        }
        model.forward_train(images, "person_a")
        batch = (
            images,
            torch.zeros(6, dtype=torch.long),
            torch.zeros(6, dtype=torch.long),
            torch.zeros(6, dtype=torch.long),
            tuple("sample" for _ in range(6)),
        )
        summary = fit_task_key_calibration(
            cfg=test_cfg,
            model=model,
            task_key="person_a",
            fingerprint_loader=[batch],
            device=torch.device("cpu"),
            logger=logging.getLogger("task-key-calibration-test"),
        )
        self.assertEqual(summary["retained_samples"], 0)
        bank = model.task_router_keys["person_a"]
        self.assertTrue(bool(bank.calibration_ready.item()))
        self.assertEqual(int(bank.calibration_count.item()), 6)

    def test_training_and_category_aware_auto_routing(self):
        test_cfg = cfg.clone()
        test_cfg.merge_from_file("configs/lifelong/MDReID_TMDA_CSCR.yml")
        with patch(
            "modeling.lifelong_model.build_transformer",
            return_value=_FakeBackbone(),
        ):
            model = LifelongMDReID(test_cfg)
        model.register_task("person_a", 3, "person")
        model.register_task("person_b", 4, "person")
        model.register_task("vehicle_a", 2, "vehicle")
        model.register_task("vehicle_b", 2, "vehicle")

        images = {
            modality: torch.randn(5, 3, 8, 8)
            for modality in MODALITIES
        }
        output = model.forward_train(images, "vehicle_b")
        self.assertIn("router_losses", output)
        self.assertTrue(
            bool(torch.isfinite(output["router_losses"]["margin"]))
        )
        output["router_losses"]["total"].backward()
        self.assertIsNotNone(
            model.task_router_keys["vehicle_b"].keys.grad
        )
        self.assertTrue(bool(
            model.task_router_keys["vehicle_b"].feature_initialized.item()
        ))
        for task_key in model.task_order:
            model.task_router_keys[task_key].set_calibration(
                median=torch.zeros(len(MODALITIES)),
                iqr=torch.ones(len(MODALITIES)),
                sample_count=5,
            )

        descriptors = model.extract_scenarios(
            images=images,
            scenarios=("RNT", "R"),
            routing="auto",
            candidate_tasks=model.task_order,
            object_category="person",
        )
        vehicle_index = model.task_order.index("vehicle_a")
        second_vehicle_index = model.task_order.index("vehicle_b")
        for _, selected, scores in descriptors.values():
            self.assertFalse(bool((selected == vehicle_index).any()))
            self.assertFalse(bool((selected == second_vehicle_index).any()))
            self.assertTrue(
                bool(
                    (
                        scores[:, vehicle_index]
                        == torch.finfo(scores.dtype).min
                    ).all()
                )
            )
            self.assertTrue(
                bool(
                    (
                        scores[:, second_vehicle_index]
                        == torch.finfo(scores.dtype).min
                    ).all()
                )
            )


if __name__ == "__main__":
    unittest.main()
