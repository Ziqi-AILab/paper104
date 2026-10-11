"""Render manuscript tables strictly from observed analysis outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def render_primary_comparison_table(statistics_root: Path, output_path: Path) -> Path | None:
    declared = statistics_root / "formal_frozen_probes" / "primary_aggregate.csv"
    source = declared if declared.is_file() else statistics_root / "primary_aggregate.csv"
    if not source.is_file():
        output_path.unlink(missing_ok=True)
        return None
    frame = pd.read_csv(source)
    required = {"comparison", "estimate", "ci_lower", "ci_upper", "holm_adjusted_p"}
    if frame.empty or not required.issubset(frame.columns):
        output_path.unlink(missing_ok=True)
        return None
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Prespecified aggregate comparisons computed from per-sample predictions. Negative paired log error ratios favor the candidate.}",
        r"\label{tab:primary-comparisons}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Comparison & Effect & 95\% CI & Holm $p$ \\",
        r"\midrule",
    ]
    for row in frame.to_dict(orient="records"):
        label = latex_escape(str(row["comparison"]).replace("_", " "))
        interval = f"[{float(row['ci_lower']):.3f}, {float(row['ci_upper']):.3f}]"
        lines.append(
            f"{label} & {float(row['estimate']):.3f} & {interval} & "
            f"{float(row['holm_adjusted_p']):.3g} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def render_observed_tables(project_root: str | Path) -> list[Path]:
    project = Path(project_root).resolve()
    generated = project / "paper/generated"
    generated.mkdir(parents=True, exist_ok=True)
    outputs = [
        render_primary_comparison_table(
            project / "results/statistics", generated / "primary_comparisons.tex"
        )
    ]
    return [path for path in outputs if path is not None]


__all__ = ["latex_escape", "render_observed_tables", "render_primary_comparison_table"]
