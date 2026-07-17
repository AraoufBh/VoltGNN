"""
VoltGNN entry point.

Stages (mirroring Algorithm 1):
    pretrain : Phase 2 — supervised pre-training against DC-OPF targets
    rl       : Phase 3 — PPO fine-tuning of the policy-refinement network phi
    deploy   : Phase 4 — streaming deployment with periodic online adaptation
    eval     : compute PGA / OLR / RUR / EDE / TAS / DL on the held-out split

Usage:
    python main.py --config configs/ieee33.yaml --stage pretrain
    python main.py --config configs/ieee33.yaml --stage rl
    python main.py --config configs/ieee33.yaml --stage eval --checkpoint runs/ieee33/best.pt
"""

from __future__ import annotations

import argparse

from utils.config import load_config
from models.voltgnn import VoltGNN


def build_model(cfg):
    return VoltGNN(cfg)


def run(cfg, stage: str, checkpoint: str | None):
    model = build_model(cfg)

    if stage == "pretrain":
        from training.pretrain import supervised_pretrain
        # loaders = build_loaders(cfg)  # train / val / test
        # supervised_pretrain(model, loaders, cfg)
        raise NotImplementedError("Wire dataset loaders, then call supervised_pretrain.")

    elif stage == "rl":
        from training.rl_refinement import ppo_finetune
        # env = build_env(cfg)
        # ppo_finetune(model, env, cfg)
        raise NotImplementedError("Wire the grid environment, then call ppo_finetune.")

    elif stage == "deploy":
        from training.online_adaptation import deploy_stream
        # stream = build_stream(cfg)
        # deploy_stream(model, stream, cfg)
        raise NotImplementedError("Wire the IoT stream, then call deploy_stream.")

    elif stage == "eval":
        # from evaluate import evaluate
        # evaluate(model, cfg, checkpoint)
        raise NotImplementedError("Implement evaluate() over the held-out split.")

    else:
        raise ValueError(f"Unknown stage: {stage}")


def main():
    parser = argparse.ArgumentParser(description="VoltGNN")
    parser.add_argument("--config", required=True, help="Path to configs/*.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["pretrain", "rl", "deploy", "eval"],
        help="Which stage of Algorithm 1 to run",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for eval/deploy")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, args.stage, args.checkpoint)


if __name__ == "__main__":
    main()
