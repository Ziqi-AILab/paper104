#!/usr/bin/env python
"""Extract one formal representation cache on an A800 job."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import pandas as pd

from polyfid.evaluation.embeddings import save_embedding_table
from polyfid.evaluation.encoder import extract_embeddings, load_encoder_checkpoint
from polyfid.paths import load_paths
from polyfid.records import RunRecord, measured_resources, write_run_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--input-view", default="valid")
    parser.add_argument("--checkpoint-fraction", type=float)
    parser.add_argument("--checkpoint-tokens", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    paths = load_paths()
    frame = pd.read_parquet(args.dataset)
    if args.max_rows is not None:
        frame = frame.iloc[: args.max_rows]
    model, tokenizer = load_encoder_checkpoint(
        args.checkpoint,
        base_checkpoint=(
            args.base_checkpoint
            or str(paths.shared_model_root / "chemberta_100m_mlm")
        ),
        device=args.device,
    )
    table = extract_embeddings(
        model,
        tokenizer,
        frame[args.sample_id_column].astype(str).tolist(),
        frame[args.text_column].astype(str).tolist(),
        method=args.method,
        representation_seed=args.seed,
        input_view=args.input_view,
        batch_size=args.batch_size,
        device=args.device,
    )
    table = replace(
        table,
        checkpoint_fraction=args.checkpoint_fraction,
        checkpoint_tokens=args.checkpoint_tokens,
    )
    saved = save_embedding_table(table, args.output)
    write_run_record(
        RunRecord(
            run_id=(
                f"embedding_{args.method}_seed-{args.seed}_{Path(args.output).stem}"
            ),
            stage="embedding_cache",
            method=args.method,
            seed=args.seed,
            config_path="configs/downstream/tasks.yaml",
            command=tuple(sys.argv),
            input_paths=(str(args.dataset), str(args.checkpoint)),
            output_path=str(saved),
            status="complete",
            metrics={"records": float(len(table.sample_ids))},
            resources=measured_resources(
                started,
                device=args.device,
                batch_size=args.batch_size,
            ),
        ),
        Path(args.output).with_suffix(".run_record.json"),
    )


if __name__ == "__main__":
    main()
