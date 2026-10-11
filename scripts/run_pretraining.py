#!/usr/bin/env python3
"""Run one Paper104 fixed reference or continued-pretraining arm."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from polyfid.models import arm_spec
from polyfid.paths import load_paths
from polyfid.training.runner import PretrainingRunConfig, run_pretraining


def _default_learning_rate(arm: str) -> float:
    return 1e-4 if arm in {"B0", "B_eqs"} else 3e-5


def _declared_arm_ids(values: Mapping[str, object]) -> set[str]:
    """Return the arm identifiers declared by ``arms.yaml``."""

    declared: set[str] = set()
    reference = values.get("reference_arm")
    if isinstance(reference, Mapping) and reference.get("id"):
        declared.add(str(reference["id"]))
    trained = values.get("trained_arms", ())
    if isinstance(trained, list):
        declared.update(
            str(row["id"])
            for row in trained
            if isinstance(row, Mapping) and row.get("id")
        )
    controls = values.get("controls")
    if isinstance(controls, Mapping):
        recipes = controls.get("recipes", ())
        if isinstance(recipes, list):
            declared.update(str(value) for value in recipes)
    return declared


def _load_and_validate_arm_config(path: Path, arm: str) -> Mapping[str, object]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("arms config must be a YAML mapping")
    if str(arm) not in _declared_arm_ids(values):
        raise ValueError(f"arm {arm!r} is not declared in {path}")
    return values


def _latest_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the readable checkpoint with the greatest optimizer progress."""

    import torch

    candidates = [checkpoint_dir / "last.pt", *sorted(checkpoint_dir.glob("token-*.pt"))]
    available: list[tuple[int, bool, Path]] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = torch.load(
                candidate,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except (OSError, RuntimeError, EOFError):
            continue
        progress = int(payload["training_state"]["global_step"])
        available.append((progress, candidate.name == "last.pt", candidate))
    return max(available, default=None, key=lambda item: item[:2])[2] if available else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--paths-config", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--pi1m-path", type=Path)
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--relation-path", type=Path)
    parser.add_argument("--protected-groups-path", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Reuse this run's latest readable checkpoint and exit cleanly "
            "if final_model exists."
        ),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--boundary-fraction", type=float, default=0.65)
    parser.add_argument("--interior-fraction", type=float, default=0.25)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument(
        "--stop-after-optimizer-steps",
        type=int,
        help="Stop at an optimizer boundary and save last.pt; used by GPU smoke/resume tests.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=("pilot", "formal", "control"), default="formal")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/pretraining/core.yaml"))
    parser.add_argument("--arms-config", type=Path, default=Path("configs/pretraining/arms.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.paths_config)
    core = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(core, Mapping):
        raise ValueError("pretraining core config must be a YAML mapping")
    _load_and_validate_arm_config(args.arms_config, args.arm)
    training = core.get("training", {})
    model_config = core.get("model", {})
    masking = core.get("masking", {})
    if not isinstance(training, Mapping) or not isinstance(model_config, Mapping):
        raise ValueError("model and training config sections must be YAML mappings")
    if not isinstance(masking, Mapping):
        raise ValueError("masking config section must be a YAML mapping")
    pi1m_path = args.pi1m_path or (
        paths.project_root / "data/derived/pi1m_periodic_identity.parquet"
    )
    model_checkpoint = args.model_checkpoint or (
        paths.shared_model_root / "chemberta_100m_mlm"
    )
    protected = args.protected_groups_path
    if protected is None:
        candidate = paths.project_root / "data/splits/protected_repeat_groups.txt"
        protected = candidate if candidate.exists() else None
    relation_path = args.relation_path or (
        paths.project_root / "data/relations/train/pi1m_triplets.parquet"
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else _default_learning_rate(args.arm)
    )
    selected_arm = arm_spec(args.arm)
    if selected_arm.needs_relation_views:
        default_micro_batch_size = int(
            training.get("relation_micro_batch_size", 8)
        )
        default_gradient_accumulation = int(
            training.get("relation_gradient_accumulation", 32)
        )
    else:
        default_micro_batch_size = int(training.get("micro_batch_size", 32))
        default_gradient_accumulation = int(
            training.get("gradient_accumulation", 8)
        )
    if args.run_id is not None:
        run_id = args.run_id
    elif args.stage == "pilot":
        run_id = (
            f"pilot_{args.arm}_seed-{args.seed}_lr-{learning_rate:g}"
            f"_b-{args.boundary_fraction:g}_i-{args.interior_fraction:g}"
        )
    else:
        run_id = f"{args.stage}_{args.arm}_seed-{args.seed}"
    controls = {
        "matched_random_masking",
        "permuted_boundary_masking",
        "equivalence_only",
        "topology_negative_only",
    }
    if args.stage == "pilot":
        checkpoint_dir = paths.checkpoint_root / "dev" / run_id
    elif args.arm in controls or args.stage == "control":
        checkpoint_dir = (
            paths.checkpoint_root / "pretraining/controls" / args.arm / f"seed-{args.seed}"
        )
    else:
        checkpoint_dir = (
            paths.checkpoint_root / "pretraining" / args.arm / f"seed-{args.seed}"
        )
    if args.resume_existing and (checkpoint_dir / "final_model").is_dir():
        existing = {
            "run_id": run_id,
            "arm": args.arm,
            "status": "already_complete",
            "checkpoint_dir": str(checkpoint_dir),
            "metrics_path": str(paths.project_root / "results/metrics" / f"{run_id}.jsonl"),
            "run_record_path": str(paths.project_root / "runs" / f"{run_id}.json"),
            "optimizer_steps": None,
            "non_padding_tokens": None,
        }
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            result_path = args.output_dir / "run_result.json"
            if not result_path.exists():
                result_path.write_text(
                    json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
        print(json.dumps(existing, indent=2, ensure_ascii=False))
        return
    resume_path = args.resume
    if resume_path is None and args.resume_existing:
        resume_path = _latest_resume_checkpoint(checkpoint_dir)
    result = run_pretraining(
        PretrainingRunConfig(
            arm=args.arm,
            seed=args.seed,
            learning_rate=learning_rate,
            pi1m_path=pi1m_path,
            model_checkpoint=model_checkpoint,
            paths=paths,
            relation_path=relation_path,
            protected_groups_path=protected,
            resume_path=resume_path,
            epochs=args.epochs if args.epochs is not None else int(training.get("epochs", 3)),
            max_length=args.max_length if args.max_length is not None else int(model_config.get("max_length", 256)),
            micro_batch_size=(
                args.micro_batch_size
                if args.micro_batch_size is not None
                else default_micro_batch_size
            ),
            gradient_accumulation=(
                args.gradient_accumulation
                if args.gradient_accumulation is not None
                else default_gradient_accumulation
            ),
            mask_probability=float(
                masking.get("atom_group_mask_probability", 0.15)
            ),
            boundary_fraction=args.boundary_fraction,
            interior_fraction=args.interior_fraction,
            weight_decay=float(training.get("weight_decay", 0.01)),
            warmup_fraction=float(training.get("warmup_fraction", 0.05)),
            optimizer_name=str(training.get("optimizer", "adamw")),
            scheduler_name=str(training.get("scheduler", "linear")),
            precision=str(model_config.get("precision", "bf16")),
            checkpoint_percentages=tuple(
                int(round(100 * float(value)))
                for value in training.get(
                    "checkpoint_token_fractions",
                    (0.01, 0.05, 0.10, 0.25, 0.50, 1.00),
                )
            ),
            max_train_rows=args.max_train_rows,
            max_validation_rows=args.max_validation_rows,
            stop_after_optimizer_steps=args.stop_after_optimizer_steps,
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
        )
    )
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run_result.json").write_text(
            json.dumps(result.__dict__, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
