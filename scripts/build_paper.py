#!/usr/bin/env python
"""Refresh observed tables and compile the Paper104 manuscript locally."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from polyfid.analysis.manuscript import render_observed_tables


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paper_root = project_root / "paper"
    if not (paper_root / "manuscript.tex").is_file():
        raise SystemExit("The optional legacy paper/manuscript.tex is not distributed in this code repository.")
    build_root = paper_root / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    tables = render_observed_tables(project_root)
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk:
        command = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={build_root}",
            "manuscript.tex",
        ]
        passes = 1
    elif pdflatex:
        command = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={build_root}",
            "manuscript.tex",
        ]
        passes = 2
    else:
        print(
            json.dumps(
                {
                    "status": "source_ready_compiler_unavailable",
                    "manuscript": str(paper_root / "manuscript.tex"),
                    "observed_tables": [str(path) for path in tables],
                    "message": "Install latexmk or pdflatex to compile the PDF locally.",
                },
                indent=2,
            )
        )
        return
    for _ in range(passes):
        completed = subprocess.run(command, cwd=paper_root, check=False)
        if completed.returncode:
            raise SystemExit(completed.returncode)
    print(
        json.dumps(
            {
                "status": "compiled",
                "pdf": str(build_root / "manuscript.pdf"),
                "observed_tables": [str(path) for path in tables],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
