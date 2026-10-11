#!/usr/bin/env python3
"""Set machine-local roots; does not download, train, or submit jobs."""
import argparse
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=root / "assets/data")
parser.add_argument("--model-root", type=Path, default=root / "assets/models")
parser.add_argument("--checkpoint-root", type=Path, default=root / "assets/checkpoints")
parser.add_argument("--exclusion-root", type=Path, default=root / "assets/exclusions")
args = parser.parse_args()
config = root / "configs/paths.yaml"
values = yaml.safe_load(config.read_text())
values.update(project_root=str(root), shared_data_root=str(args.data_root.expanduser().resolve()),
              shared_model_root=str(args.model_root.expanduser().resolve()),
              checkpoint_root=str(args.checkpoint_root.expanduser().resolve()),
              exclusion_root=str(args.exclusion_root.expanduser().resolve()))
config.write_text(yaml.safe_dump(values, sort_keys=False))
print(config)
