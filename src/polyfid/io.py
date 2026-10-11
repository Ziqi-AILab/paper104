"""Small same-directory atomic writers for experiment outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


def _atomic_write(
    path: str | Path,
    writer: Callable[[Path], None],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_save_npz(
    path: str | Path,
    arrays: Mapping[str, Any],
) -> Path:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    return _atomic_write(path, write)


def atomic_write_parquet(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    return _atomic_write(
        path,
        lambda temporary: frame.to_parquet(temporary, index=index),
    )


def atomic_write_csv(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    sep: str = ",",
) -> Path:
    return _atomic_write(
        path,
        lambda temporary: frame.to_csv(temporary, index=index, sep=sep),
    )


def atomic_write_text(
    text: str,
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> Path:
    return _atomic_write(
        path,
        lambda temporary: temporary.write_text(text, encoding=encoding),
    )


__all__ = [
    "atomic_save_npz",
    "atomic_write_csv",
    "atomic_write_parquet",
    "atomic_write_text",
]
