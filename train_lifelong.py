"""Entry point for Track A/B/C exemplar-free lifelong training."""

import argparse
from pathlib import Path

from config import cfg
from engine.lifelong_trainer import run_lifelong_training
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Lifelong multi-modal ReID training")
    parser.add_argument(
        "--config_file",
        default="configs/lifelong/MDReID_TMDA_CSCR.yml",
        type=str,
    )
    parser.add_argument("--track", choices=("A", "B", "C"), default=None)
    parser.add_argument(
        "--order",
        choices=("grouped", "interleaved", "alternate"),
        default=None,
        help="Only changes Track C; Track A/B use their fixed order.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="YACS overrides, e.g. DATASETS.ROOT_DIR /data/reid",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if not cfg.LIFELONG.ENABLED:
        raise RuntimeError("Set LIFELONG.ENABLED=True for this entry point.")
    track = args.track or str(cfg.LIFELONG.TRACK)
    order = args.order or str(cfg.LIFELONG.ORDER)
    log_dir = Path(cfg.OUTPUT_DIR).expanduser().resolve() / (
        "track_{}_{}".format(track, order)
    )
    logger = setup_logger("MDReID.lifelong", str(log_dir), if_train=True)
    logger.info("Configuration:\n%s", cfg)
    report = run_lifelong_training(cfg, track=track, order=order)
    logger.info("Completed %d lifelong stages.", len(report["stages"]))
