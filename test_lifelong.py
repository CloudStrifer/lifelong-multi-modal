"""Evaluate a lifelong checkpoint on the four currently enabled scenarios."""

import argparse
from pathlib import Path

from config import cfg
from engine.lifelong_trainer import run_lifelong_test
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Lifelong multi-modal ReID evaluation")
    parser.add_argument(
        "--config_file",
        default="configs/lifelong/MDReID_TMDA_CSCR.yml",
        type=str,
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--routing", choices=("auto", "oracle"), default="auto")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    log_dir = checkpoint.parent.parent
    logger = setup_logger("MDReID.lifelong.test", str(log_dir), if_train=False)
    logger.info("Configuration:\n%s", cfg)
    results = run_lifelong_test(
        cfg=cfg,
        checkpoint_path=str(checkpoint),
        routing=args.routing,
    )
    logger.info("Evaluated %d datasets.", len(results["datasets"]))
