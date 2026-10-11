#!/usr/bin/env python3
"""Render the development manuscript from completed results; no new inference."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAT = ROOT / "results/statistics"
OUT = ROOT / "paper/draft_v1"
FIG = OUT / "figures"
GEN = OUT / "generated"
ARMS = ["C", "B0", "A0", "A_att", "A_gcl", "A_eqs", "B_eqs"]
TASKS = ["EA", "IP", "EGC", "EGB", "EEA", "EI", "XC", "EPS", "NC", "OPV", "PE-II", "PE-I"]
SEVEN = ["EEA", "EGB", "EGC", "EI", "EPS", "NC", "XC"]
BASE = ["baseline_ecfp_ridge", "baseline_rdkit_ridge", "baseline_ecfp_xgboost", "baseline_rdkit_xgboost", "baseline_nearest_neighbor", "baseline_chemprop_dpnn"]
COLORS = dict(zip(ARMS, ["#89939B", "#CC9B44", "#547DA5", "#4B9996", "#8E79AA", "#CC6E57", "#7E923B"]))
COLORS.update({"chemprop": "#3C4E64", "transpolymer": "#AB5793", "periogt": "#2885A1", "grin": "#827561"})
LABEL = {m: m.replace("_", "-") for m in ARMS}
LABEL.update({"chemprop": "Chemprop", "transpolymer": "TransPolymer", "periogt": "PerioGT", "grin": "GRIN",
 "baseline_ecfp_ridge": "ECFP Ridge", "baseline_rdkit_ridge": "RDKit Ridge", "baseline_ecfp_xgboost": "ECFP XGB", "baseline_rdkit_xgboost": "RDKit XGB", "baseline_nearest_neighbor": "Nearest", "baseline_chemprop_dpnn": "D-MPNN"})
LABEL.update({k.removeprefix('baseline_'):v for k,v in list(LABEL.items()) if k.startswith('baseline_')})
FAMILY = {"overall": "Overall", "alternative_periodic_cut": "Periodic cut", "randomized_traversal": "Traversal", "reverse_when_non_noop": "Reversal", "endpoint_relocation": "Endpoint relocation", "closure_rewiring": "Closure rewiring"}
COMPS = [("molecular_initialization", "A0 / B0"), ("boundary_supervision", "A-att / A0"), ("verified_relations", "A-eqs / A-gcl")]
SOURCES: list[dict] = []
LEDGER: list[dict] = []


def read(name: str) -> pd.DataFrame:
    p = STAT / name
    d = pd.read_csv(p)
    SOURCES.append({"source": str(p.relative_to(ROOT)), "rows": len(d), "role": "completed result; read only"})
    return d


def record(data: pd.DataFrame, asset: str, source: str) -> None:
    data.to_csv(GEN / f"{asset}_data.csv", index=False)
    for i, row in data.iterrows():
        for col, value in row.items():
            if isinstance(value, (float, int, np.number)) and pd.notna(value):
                LEDGER.append({"asset": asset, "row": i, "field": col, "value": value, "source": source})


def save(fig: plt.Figure, name: str) -> None:
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=320, facecolor="white")
    plt.close(fig)


def panel(ax, letter: str, title: str):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=9.5, fontweight="bold", pad=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E8EBED", lw=.6)
    ax.set_axisbelow(True)


def forest(ax, d, labels, *, estimate="estimate", lower="ci_lower", upper="ci_upper", scale=1., log=False, color="#547DA5"):
    transform = (lambda v: 100 * np.expm1(v)) if log else (lambda v: scale * v)
    v, lo, hi = (transform(d[c].to_numpy(float)) for c in [estimate, lower, upper])
    yy = np.arange(len(d))[::-1]
    ax.errorbar(v, yy, xerr=np.vstack([v-lo, hi-v]), fmt="o", ms=4, capsize=2, lw=1.1, color=color)
    ax.axvline(0, c="#646B70", lw=.8, ls="--")
    ax.set_yticks(yy, labels)
    ax.set_ylim(-.6, len(d)-.4)


def figure2():
    geo = read("draft_closure/geometry/intervals.csv")
    dec = read("draft_closure/cosine_decomposition_descriptive.csv")
    ctr = read("mechanism_controls/relation_control_comparisons.csv")
    fig, axs = plt.subplots(2, 3, figsize=(7.05, 5.8), layout="constrained")
    ax = axs[0, 0]
    avg = dec.groupby("method")[["anchor_positive_cosine", "anchor_negative_cosine"]].mean()
    for j, method in enumerate(["A_gcl", "A_eqs"]):
        ax.plot([0, 1], avg.loc[method], "o-", color=COLORS[method], label=LABEL[method], ms=5)
    ax.set_xticks([0, 1], ["Positive", "Negative"]); ax.set_xlim(-.15, 1.15); ax.set_ylim(.70, .91)
    ax.set_ylabel("Mean cosine"); panel(ax, "a", "Relative, not absolute\nseparation")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    scopes = list(FAMILY)[:4]
    for ax, metric, letter, title, factor in [(axs[0, 1], "topology_margin", "b", "Margin by\nrelation family", 1), (axs[0, 2], "triplet_accuracy", "c", "Triplet accuracy", 100), (axs[1, 0], "recall_at_1", "d", "Retrieval Recall@1", 100)]:
        d = geo.loc[geo.metric.eq(metric) & geo.scope.isin(scopes)].set_index("scope").loc[scopes].reset_index()
        forest(ax, d, [FAMILY[x] for x in d.scope], scale=factor, color=COLORS["A_eqs"])
        ax.set_xlabel("Difference (pp)" if factor == 100 else "Cosine difference")
        panel(ax, letter, title)
    d = geo.loc[geo.metric.eq("triplet_accuracy") & geo.scope.isin(list(FAMILY)[4:])]
    forest(axs[1, 1], d, ["Endpoint\nrelocation", "Closure\nrewiring"], scale=100, color=COLORS["A_eqs"])
    axs[1, 1].set_xlabel("Accuracy difference (pp)"); panel(axs[1, 1], "e", "Explicit connectivity\nedits")
    d = ctr.loc[ctr.candidate.eq("equivalence_only") & ctr.relation_family.eq("alternative_periodic_cut")].set_index("metric")
    metrics = ["equivalence_cosine", "topology_margin", "triplet_accuracy"]
    x = np.arange(3)
    axs[1, 2].bar(x-.17, d.loc[metrics, "full_objective_mean"], .34, color=COLORS["A_eqs"], label="Full")
    axs[1, 2].bar(x+.17, d.loc[metrics, "candidate_mean"], .34, color="#8795A1", label="Eq-only")
    axs[1, 2].set_xticks(x, ["Cosine", "Margin", "Accuracy"], rotation=25, ha="right")
    axs[1, 2].set_ylabel("Mean value; cut family"); axs[1, 2].legend(frameon=False, fontsize=9, loc="upper right")
    axs[1, 2].set_ylim(-.07, 1.27)
    panel(axs[1, 2], "f", "Cut component\ncontrol (3 seeds)")
    save(fig, "figure2_structural_response")
    record(geo, "figure2_intervals", "draft_closure/geometry/intervals.csv")
    record(avg.reset_index(), "figure2_cosines", "draft_closure/cosine_decomposition_descriptive.csv")


def figure3():
    d = read("draft_closure/property_bridge/intervals.csv")
    order = SEVEN + ["aggregate"]
    fig, axs = plt.subplots(1, 3, figsize=(7.05, 3.8), layout="constrained")
    for ax, metric, letter, title in zip(axs, ["periodic_cut_prediction_sd", "traversal_prediction_sd", "reference_rmse"], "abc", ["Periodic-cut\nprediction SD", "Traversal\nprediction SD", "Reference\nprediction RMSE"]):
        f = d.loc[d.metric.eq(metric)].set_index("task").loc[order].reset_index()
        forest(ax, f, ["Task-equal" if x == "aggregate" else x for x in f.task], log=True, color=COLORS["A_eqs"])
        ax.set_xlabel("Relative change (%)"); panel(ax, letter, title)
    save(fig, "figure3_same_record_bridge")
    record(d, "figure3", "draft_closure/property_bridge/intervals.csv")


def figure4():
    d = read("completed_experiments/task_bootstrap.csv")
    agg = read("completed_experiments/secondary_aggregate.csv")
    fig, axs = plt.subplots(2, 3, figsize=(7.05, 7.8), layout="constrained")
    for ax, (comp, label), letter in zip(axs[0], COMPS, "abc"):
        f = d.loc[d.analysis_family.eq("frozen_task_intervals") & d.comparison.eq(comp)].set_index("task").loc[TASKS].reset_index()
        forest(ax, f, f.task, log=True); panel(ax, letter, f"Frozen\n{label}")
        ax.set_xlabel("Error change (%)")
    for ax, comp, label, letter in [(axs[1, 0], "molecular_initialization", "LoRA: A0 / B0", "d"), (axs[1, 1], "verified_relations", "LoRA: A-eqs / A-gcl", "e")]:
        f = pd.concat([d.loc[d.analysis_family.eq("lora") & d.comparison.eq(comp)], agg.loc[agg.analysis_family.eq("lora") & agg.comparison.eq(comp)]]).set_index("task").loc[["EA", "IP", "EGC", "XC", "aggregate"]].reset_index()
        forest(ax, f, ["Task-equal" if t == "aggregate" else t for t in f.task], log=True, color=COLORS["A_eqs"])
        panel(ax, letter, label.replace(": ","\n")); ax.set_xlabel("Error change (%)")
    f = d.loc[d.analysis_family.eq("adaptation_sensitivity")].copy().sort_values(["candidate", "task"])
    labs = [f"{r.candidate.replace('_full','').replace('_','-')}: {r.task}" for r in f.itertuples()]
    forest(axs[1, 2], f, labs, log=True, color="#87923F"); panel(axs[1, 2], "f", "Full fine-tuning\nversus LoRA")
    axs[1, 2].set_xlabel("Error change (%)")
    save(fig, "figure4_predictive_utility")
    record(d, "figure4_task_intervals", "completed_experiments/task_bootstrap.csv")


def figure5():
    d = read("token_efficiency/formal/fraction_effects.csv")
    b = read("mechanism_controls/boundary_main_comparison.csv")
    c = read("mechanism_controls/boundary_control_comparisons.csv")
    fig, axs = plt.subplots(2, 2, figsize=(7.05, 5.8), layout="constrained")
    ax = axs[0, 0]
    for method, field in [("A0", "candidate"), ("B0", "baseline")]:
        ax.plot(d[f"{field}_checkpoint_tokens_mean"] / 1e6, d[f"{field}_token_mean_nll"], "o-", label=LABEL[method], color=COLORS[method], ms=4)
    ax.set_xscale("log"); ax.set_xlabel("Observed anchor tokens (millions)"); ax.set_ylabel("Token NLL"); ax.legend(frameon=False)
    panel(ax, "a", "Initialization learning curves")
    forest(axs[0, 1], d, [f"{x:g}%" for x in d.checkpoint_percent], estimate="relative_percent", lower="relative_percent_ci_lower", upper="relative_percent_ci_upper", color=COLORS["A0"])
    axs[0, 1].set_xlabel("A0 / B0: NLL change (%)"); panel(axs[0, 1], "b", "Paired intervals (5 seeds)")
    forest(axs[1, 0], b, ["Distance 0", "Distance 1", "Distance 2", "Interior"], estimate="relative_nll_percent", lower="relative_nll_percent_ci_lower", upper="relative_nll_percent_ci_upper", color=COLORS["A_att"])
    axs[1, 0].set_xlabel("A-att / A0: NLL change (%)"); panel(axs[1, 0], "c", "Boundary supervision (5 seeds)")
    stratum = {"distance_0": "d0", "distance_1": "d1", "distance_2": "d2", "interior": "Interior"}
    labs = [("Matched" if "matched" in r.candidate else "Permuted") + ": " + stratum[r.stratum] for r in c.itertuples()]
    forest(axs[1, 1], c, labs, estimate="relative_nll_percent", lower="relative_nll_percent_ci_lower", upper="relative_nll_percent_ci_upper", color="#778792")
    axs[1, 1].set_xlabel("Control / A0: NLL change (%)"); panel(axs[1, 1], "d", "Masking controls (3 seeds)")
    save(fig, "figure5_initialization_boundary")
    record(d, "figure5_initialization", "token_efficiency/formal/fraction_effects.csv")


def figure6(batch=1):
    d = read("cost_analysis/profiling_cost_performance.csv")
    f = d.loc[d["mode"].eq("gpu") & d.regime.eq("prepared_input_end_to_end") & d.batch_size.eq(batch)]
    fig, axs = plt.subplots(2, 4, figsize=(7.05, 4.7), layout="constrained")
    methods = ARMS + ["chemprop", "transpolymer", "periogt", "grin"]
    markers = dict(zip(methods, ["o", "s", "^", "v", "D", "*", "P", "X", "<", ">", "h"]))
    for ax, task, letter in zip(axs.flat, SEVEN, "abcdefg"):
        for r in f.loc[f.task.eq(task)].itertuples():
            ax.scatter(r.amortized_latency_median_ms_per_record, r.rmse, s=38 if r.method == "A_eqs" else 24, marker=markers[r.method], color=COLORS[r.method], edgecolors="white", linewidths=.35)
        ax.set_xscale("log"); ax.set_xlabel("ms / record", fontsize=9)
        ax.xaxis.set_major_locator(FixedLocator([2,5,10,20] if batch==1 else [.1,.3,1,3,10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position: f"{value:g}"))
        unit = "eV" if task in ["EEA", "EGB", "EGC", "EI"] else "pp" if task == "XC" else "unitless"
        ax.set_ylabel(f"RMSE ({unit})", fontsize=9); panel(ax, letter, task)
    axs[1, 3].axis("off")
    handles = [Line2D([0], [0], marker=markers[m], linestyle="", color=COLORS[m], label=LABEL[m], markersize=5) for m in methods]
    axs[1, 3].legend(handles=handles, frameon=False, loc="center", fontsize=9, handletextpad=.5)
    name = "figure6_cost_error" if batch == 1 else "supp_cost_error_batch32"
    save(fig, name); record(f, name, "cost_analysis/profiling_cost_performance.csv")


def main_scope(d):
    protocols = d.task.map({"EA": "family_ood", "IP": "family_ood", "PE-I": "official_group_disjoint"}).fillna("group_ood")
    return d.loc[d.protocol.eq(protocols) & d.input_view.eq("valid")]


def tex(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def fmt(v, digits=4):
    if pd.isna(v): return "---"
    return f"{v:.{digits}f}"


def tabular(headers, rows, *, long=False, caption="", label="", note="", fontsize="small"):
    cols = "l" + "r" * (len(headers)-1)
    lines = ["{\\"+fontsize, r"\setlength{\tabcolsep}{3pt}"]
    if long:
        lines += [r"\begin{longtable}{"+cols+"}", r"\caption{"+caption+"}\\label{"+label+r"}\\", r"\toprule", " & ".join(headers)+r" \\", r"\midrule\endfirsthead", r"\toprule", " & ".join(headers)+r" \\", r"\midrule\endhead", r"\bottomrule\endfoot"]
    else:
        lines += [r"\begin{tabular}{"+cols+"}", r"\toprule", " & ".join(headers)+r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row))+r" \\" for row in rows)
    lines += [r"\bottomrule" if not long else "", r"\end{"+("longtable" if long else "tabular")+"}", "}"]
    if note: lines += [r"\par\smallskip{\small "+note+"}"]
    return "\n".join(lines)+"\n"


def write_table(name, text):
    (GEN / f"{name}.tex").write_text(text)


def raw_matrix(d, methods, tasks):
    p = d.pivot_table(index="task", columns="method", values="mean")
    return [[tex(task), *[fmt(p.loc[task, m] if task in p.index and m in p.columns else np.nan) for m in methods]] for task in tasks]


def tables():
    allm = read("completed_experiments/tables/all_method_summary.csv")
    full = main_scope(allm.loc[allm.label_fraction.eq(1.) & allm.subsample_id.eq(0)])
    for name, methods, tasks in [("main_core_performance", ARMS, TASKS), ("main_matched_baselines", BASE, TASKS), ("main_external_baselines", ["transpolymer", "periogt", "grin"], SEVEN)]:
        write_table(name, tabular(["Task"] + [tex(LABEL[m]) for m in methods], raw_matrix(full, methods, tasks)))
        record(full.loc[full.method.isin(methods)], name, "completed_experiments/tables/all_method_summary.csv")
    p = read("formal_frozen_probes/primary_aggregate.csv")
    rows = [[tex(f"{r.candidate} / {r.baseline}"), f"{r.estimate:+.5f}", f"{100*np.expm1(r.estimate):+.2f}", f"[{100*np.expm1(r.ci_lower):+.2f}, {100*np.expm1(r.ci_upper):+.2f}]", f"{r.holm_adjusted_p:.4f}"] for r in p.itertuples()]
    write_table("main_primary_tests", tabular(["Comparison", "Log ratio", "Change (\\%)", "95\\% CI (\\%)", "Holm $p$"], rows))
    record(p, "main_primary_tests", "formal_frozen_probes/primary_aggregate.csv")
    costs = read("cost_analysis/pretraining_allocations.csv")
    rows=[]
    for m in ARMS:
        a=costs.loc[costs.method.eq(m)]
        rows.append([tex(LABEL[m]), str(len(a)), fmt(a.allocated_4090_hours.mean(),3), fmt(a.allocated_4090_hours.std(),3), fmt(a.loc[a.representation_seed.eq(0 if m == 'C' else 42), 'allocated_4090_hours'].iloc[0],3)])
    write_table("main_pretraining_cost", tabular(["Arm", "$n$", "Mean (4090 h)", "SD (4090 h)", "rep42 (4090 h)"], rows, note="C has no PI1M training; its upstream cost is unknown, not zero. Allocations include observed resumed/failed attempts. The C scenario is fixed, not seed 42."))
    record(costs.groupby('method',as_index=False).agg(n_seeds=('representation_seed','size'),mean_4090_hours=('allocated_4090_hours','mean'),sd_4090_hours=('allocated_4090_hours','std')), 'main_pretraining_cost', 'cost_analysis/pretraining_allocations.csv')
    reuse=read("cost_analysis/seven_task_reuse_components.csv")
    rows=[[tex(LABEL.get(r.method,r.method)),fmt(r.shared_pi1m_allocated_4090_hours,3),fmt(r.seven_task_fit_allocated_a800_hours,3),fmt(r.seven_task_three_head_bundle_cpu_seconds,2),fmt(r.seven_task_cpu_fit_seconds,2)] for r in reuse.itertuples()]
    write_table("main_adoption_components",tabular(["Method","Shared 4090 h","Fit A800 h","3-head CPU s","Other fit CPU s"],rows,note="Seven-task totals; rep42 (C fixed), fold 0 and baseline seed 11. Missing components are unmeasured/not applicable, not zero. Resource columns are not added. Encoder time has broader coverage than seven tasks and is not included as a precise total."))
    record(reuse,'main_adoption_components','cost_analysis/seven_task_reuse_components.csv')
    intervals=pd.concat([read("completed_experiments/task_bootstrap.csv"), read("completed_experiments/secondary_aggregate.csv")], ignore_index=True)
    intervals=intervals.sort_values(["analysis_family","comparison","task"])
    record(intervals,'si_original_intervals','completed_experiments/task_bootstrap.csv;completed_experiments/secondary_aggregate.csv')
    rows=[[tex(r.analysis_family.replace("frozen_task_intervals","frozen").replace("adaptation_sensitivity","full/LoRA")),tex(f"{r.candidate}/{r.baseline}"),tex(r.task),f"{100*np.expm1(r.estimate):+.2f}",f"[{100*np.expm1(r.ci_lower):+.2f}, {100*np.expm1(r.ci_upper):+.2f}]"] for r in intervals.itertuples()]
    write_table("si_original_intervals",tabular(["Analysis","Candidate/reference","Task","Change (\\%)","95\\% CI (\\%)"],rows,long=True,caption="All 70 original task and adaptation intervals. Negative values indicate lower prediction error. Five representation seeds with three nested heads; 10,000 paired hierarchical resamples. These do not replace the original three aggregate primary tests.",label="tab:si-original"))
    geo=read("draft_closure/geometry/intervals.csv")
    mn={"equivalence_cosine":"Cosine", "recall_at_1":"Recall@1", "mrr":"MRR", "topology_margin":"Margin", "triplet_accuracy":"Accuracy"}
    rows=[[FAMILY[r.scope],mn[r.metric],fmt(r.candidate_value),fmt(r.baseline_value),f"{r.estimate:+.4f}",f"[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]"] for r in geo.itertuples()]
    write_table("si_geometry_intervals",tabular(["Family","Metric","A-eqs","A-gcl","Difference","95\\% CI"],rows,long=True,caption="All 24 secondary structural intervals. Recall, MRR and accuracy are fractions here, not percentage points. Five representation seeds; 10,000 seed-to-anchor paired resamples.",label="tab:si-geometry"))
    bridge=read("draft_closure/property_bridge/intervals.csv")
    mn={"periodic_cut_prediction_sd":"Cut SD","traversal_prediction_sd":"Traversal SD","reference_rmse":"Reference RMSE"}
    rows=[[tex(r.task),mn[r.metric],fmt(r.candidate_value),fmt(r.baseline_value),f"{100*np.expm1(r.estimate):+.2f}",f"[{100*np.expm1(r.ci_lower):+.2f}, {100*np.expm1(r.ci_upper):+.2f}]"] for r in bridge.itertuples()]
    write_table("si_bridge_intervals",tabular(["Task","Response","A-eqs","A-gcl","Change (\\%)","95\\% CI (\\%)"],rows,long=True,caption="All 24 same-record intervals. Task-specific records jointly eligible for both view families; 4,765 task records and 2,801 shared structure groups. Original reference predictions define RMSE. Aggregate raw-scale cells are not defined across different units.",label="tab:si-bridge"))
    tok=read("token_efficiency/formal/fraction_effects.csv")
    rows=[[f"{r.checkpoint_percent:g}",f"{r.candidate_checkpoint_tokens_mean/1e6:.4f}",fmt(r.candidate_token_mean_nll),fmt(r.baseline_token_mean_nll),f"{r.relative_percent:+.3f}",f"[{r.relative_percent_ci_lower:+.3f}, {r.relative_percent_ci_upper:+.3f}]"] for r in tok.itertuples()]
    write_table("si_token_efficiency",tabular(["Budget (\\%)","Anchor M tokens","A0 NLL","B0 NLL","Change (\\%)","95\\% CI (\\%)"],rows,long=True,caption="Six predeclared token checkpoints on the full PI1M validation partition. Actual nonpadding MLM anchor tokens are recorded; relation-view compute and upstream molecular pretraining are not included in this token count. Five representation seeds and 10,000 paired resamples.",label="tab:si-token-efficiency"))
    pv=read("property_views/formal/comparison_aggregate.csv")
    pvs=read("property_views/formal/sensitivity_aggregate.csv")
    rows=[[tex(f"{r.candidate}/{r.baseline}"),tex(r.view_family),str(r.n_tasks),f"{100*np.expm1(r.estimate):+.2f}",f"[{100*np.expm1(r.ci_lower):+.2f}, {100*np.expm1(r.ci_upper):+.2f}]"] for r in pv.itertuples()]
    write_table("si_original_view_stability",tabular(["Comparison","View","Tasks","SD change (\\%)","95\\% CI (\\%)"],rows,long=True,caption="Original property-view aggregate intervals. Their cohort/draw scope differs from the subsequent common-record bridge; both numerical results are retained rather than forcing identical decimals. These are secondary analyses.",label="tab:si-original-view-stability"))
    pieces=[]
    for analysis,q in pvs.groupby('analysis'):
        rows=[[tex(f"{r.candidate}/{r.baseline}"),tex(r.view_family),str(r.n_tasks),f"{100*np.expm1(r.estimate):+.2f}",f"[{100*np.expm1(r.ci_lower):+.2f}, {100*np.expm1(r.ci_upper):+.2f}]"] for r in q.itertuples()]
        pieces.append(tabular(["Comparison","View","Tasks","Change (\\%)","95\\% CI (\\%)"],rows,long=True,caption=f"Property-view sensitivity: {tex(analysis)}. Metric: {tex(q.metric.iloc[0])}. Same completed secondary analysis; no additional primary multiplicity family.",label='tab:si-view-'+analysis))
    write_table("si_view_sensitivities","\n".join(pieces))
    record(read("property_views/formal/scope.csv"),"si_property_view_eligibility","property_views/formal/scope.csv")
    # Designed-matrix reporting averages fixed subsample summaries only for display.
    # All individual subsample values remain in the companion CSV, not altered.
    record(allm, "si_complete_descriptive_matrix", "completed_experiments/tables/all_method_summary.csv")
    display=allm.groupby(["method","task","protocol","input_view","label_fraction"],as_index=False).agg(mean=("mean","mean"),n_subsamples=("subsample_id","nunique"))
    for group, methods in [("core",ARMS),("matched",BASE),("adaptation",["B0_lora","A0_lora","A_att_lora","A_gcl_lora","A_eqs_lora","B_eqs_lora","A0_full","A_eqs_full"]),("external",["transpolymer","periogt","grin"])]:
        f=display.loc[display.method.isin(methods)]
        pieces=[]
        for (protocol,view),q in f.groupby(["protocol","input_view"]):
            wide=q.pivot_table(index=["task","label_fraction"],columns="method",values="mean")
            present=[m for m in methods if m in wide.columns]
            rows=[[tex(t),f"{100*lf:g}",*[fmt(row.get(m,np.nan)) for m in present]] for (t,lf),row in wide.iterrows()]
            headers=["Task","Labels (\\%)"]+[tex(LABEL.get(m,m.replace('_lora','-L').replace('_full','-F').replace('_','-'))) for m in present]
            caption=f"Complete {group} descriptive matrix: {tex(protocol)}, {tex(view)}. MAE for EA/IP; RMSE otherwise. Held-out row errors are pooled before metrics; nested head means precede representation means. Low-data entries average the five fixed subsamples, not five new model seeds. L and F denote LoRA and full fine-tuning."
            pieces.append(tabular(headers,rows,long=True,caption=caption,label=f"tab:si-{group}-{protocol}-{view}",fontsize="footnotesize"))
        write_table(f"si_matrix_{group}","\n".join(pieces))
    for name, fn in [("boundary","boundary_main_comparison.csv"),("mask_controls","boundary_control_comparisons.csv")]:
        b=read("mechanism_controls/"+fn)
        record(b,'si_'+name,'mechanism_controls/'+fn)
        rows=[[tex(r.candidate.replace("matched_random_masking","matched-random").replace("permuted_boundary_masking","permuted")),tex(r.stratum.replace("distance_","d")),str(r.n_representation_seeds),f"{r.relative_nll_percent:+.2f}",f"[{r.relative_nll_percent_ci_lower:+.2f}, {r.relative_nll_percent_ci_upper:+.2f}]",f"{r.accuracy_difference_pp:+.2f}",f"[{r.accuracy_difference_pp_ci_lower:+.2f}, {r.accuracy_difference_pp_ci_upper:+.2f}]"] for r in b.itertuples()]
        write_table("si_"+name,tabular(["Arm / A0","Stratum","Seeds","NLL \\%","95\\% CI","Acc. pp","95\\% CI"],rows,long=True,caption="Boundary recovery with equal-atom-group NLL and accuracy. Controls use three representation seeds; the main contrast uses five.",label="tab:si-"+name))
    rel=read("mechanism_controls/relation_control_comparisons.csv")
    record(rel,'si_relation_controls','mechanism_controls/relation_control_comparisons.csv')
    rows=[["Eq-only" if r.candidate=='equivalence_only' else "Topo-only",FAMILY.get(r.relation_family,"Overall"),mn.get(r.metric,r.metric).replace("equivalence_cosine","Cosine").replace("topology_margin","Margin").replace("triplet_accuracy","Accuracy").replace("recall_at_1","R@1"),fmt(r.candidate_mean),fmt(r.full_objective_mean),f"[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]"] for r in rel.itertuples()]
    write_table("si_relation_controls",tabular(["Control","Family","Metric","Control","Full","Difference CI"],rows,long=True,caption="All component-control intervals versus A-eqs on three shared representation seeds. Accuracy and retrieval are fractions. A change in similarity alone is not evidence of representation collapse.",label="tab:si-relation-controls",fontsize="footnotesize"))
    # Cost timing tables retain all conditions, rather than selecting favorable rows.
    timing=read("cost_analysis/profiling_cost_performance.csv")
    for mode in ["gpu","cached-head"]:
        q=timing.loc[timing["mode"].eq(mode)].sort_values(["task","method","regime","batch_size"])
        rows=[[tex(r.task),tex(LABEL[r.method]),str(r.batch_size),"Prep" if r.regime=='prepared_input_end_to_end' else "Compute",fmt(r.amortized_latency_median_ms_per_record,3),fmt(r.amortized_latency_p95_ms_per_record,3),fmt(r.throughput_records_per_second,1),fmt(r.peak_allocated_gib,3)] for r in q.itertuples()]
        write_table("si_timing_"+mode.replace('-','_'),tabular(["Task","Method","Batch","Scope","Median ms","p95 ms","records/s","GiB"],rows,long=True,caption=f"Complete {mode} timing conditions. Prep is prepared-input end-to-end prediction; loading, offline preprocessing and writing are excluded. Batch 32 latency is amortized per record, not single-request latency. Twenty warmup batches and ten complete test passes; repetitions are timing measurements, not model seeds.",label="tab:si-timing-"+mode,fontsize="footnotesize"))
    q=timing.loc[timing['mode'].eq('gpu')&timing.batch_size.eq(1)&timing.regime.eq('prepared_input_end_to_end')].sort_values(['task','method'])
    rows=[[tex(r.task),tex(LABEL[r.method]),str(int(r.test_records)),fmt(r.rmse),tex(r.dtype),fmt(r.max_length,0),fmt(r.parameter_count/1e6,3),fmt(r.resident_state_bytes/(1024**2),2)] for r in q.itertuples()]
    write_table("si_timing_models",tabular(["Task","Method","Test n","RMSE","dtype","Max length","M params","State MiB"],rows,long=True,caption="All 77 measured model-task units. RMSE is from the timed model's own predictions: the recovered baseline fits do not replace the original performance matrix. State MiB is resident parameter/buffer storage, not archive file size. Molecular arms retain the original MLM forward; workflows are not precision/length matched.",label="tab:si-timing-models",fontsize="footnotesize"))
    # Whole-study resources stay separated by device and by measurement scope.
    gpu=read("cost_analysis/research_gpu_cost.csv")
    rows=[[tex(r.stage),tex(r.gpu_type),tex(r.allocation_outcome),str(r.n_allocations),fmt(r.allocated_gpu_hours,3)] for r in gpu.itertuples()]
    write_table("si_research_gpu",tabular(["Stage","GPU","Outcome","Allocations","GPU hours"],rows,long=True,caption="Observed whole-study GPU allocations. Different GPU types are not summed into a hardware-independent total. Interrupted/resumed allocations remain recorded.",label="tab:si-research-gpu",fontsize="footnotesize"))
    cpu=read("cost_analysis/research_cpu_cost.csv")
    rows=[[tex(r.role),tex(r.method),str(r.expected_records),str(r.measured_timers),fmt(r.cpu_process_wall_hours,3)] for r in cpu.itertuples()]
    write_table("si_research_cpu",tabular(["Role","Method","Expected records","Timers","Process-wall h"],rows,long=True,caption="Observed CPU process-wall time, not CPU core-hours or elapsed project time. Each frozen output is one three-head training bundle counted once. Missing timer scopes remain unknown.",label="tab:si-research-cpu",fontsize="footnotesize"))


def supplement():
    d=read("completed_experiments/tables/all_method_summary.csv")
    main=main_scope(d.loc[d.method.isin(ARMS)])
    fig,axs=plt.subplots(2,2,figsize=(7.05,5.3),layout="constrained")
    for ax,task,letter in zip(axs.flat,["EA","IP","EGC","XC"],"abcd"):
        f=main.loc[main.task.eq(task)].groupby(["method","label_fraction"],as_index=False)["mean"].mean()
        for m in ARMS:
            q=f.loc[f.method.eq(m)].sort_values("label_fraction")
            ax.plot(q.label_fraction*100,q["mean"],"o-",ms=3,color=COLORS[m],label=LABEL[m],lw=1)
        ax.set_xscale("log");ax.set_xlabel("Labeled training groups (%)");ax.set_ylabel("MAE (eV)" if task in ["EA","IP"] else "RMSE (eV)" if task=='EGC' else "RMSE (pp)");panel(ax,letter,task)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=7,fontsize=9,loc='outside lower center')
    save(fig,"supp_low_data_curves")
    z=[]; rel=[]; inv=[];app=[];drift=[]
    for sub,store in [("retention/zinc50k_mlm",z),("fidelity/relations",rel),("fidelity/invariance",inv)]:
        for p in sorted((ROOT/"results/metrics"/sub).glob("*/metrics.json")):
            j=json.loads(p.read_text())
            if j.get('method') in ARMS and (j.get('method')=='C' or j.get('representation_seed') in [42,123,777,2025,2026]):
                store.append(j);SOURCES.append({"source":str(p.relative_to(ROOT)),"rows":1,"role":"completed per-checkpoint metric summary"})
    z=pd.DataFrame(z)
    cols=["token_mean_nll","token_accuracy","atom_group_mean_nll","atom_group_accuracy"]
    avg=z.groupby("method")[cols].mean().reindex(ARMS)
    write_table("si_retention_mlm",tabular(["Arm","Token NLL","Token accuracy","Atom NLL","Atom accuracy"],[[tex(LABEL[m]),*[fmt(v) for v in row]] for m,row in avg.iterrows()]))
    rel=pd.DataFrame(rel)
    cols=["equivalence_cosine","recall_at_1","mrr","topology_margin","triplet_accuracy"]
    avg=rel.groupby("method")[cols].mean().reindex(ARMS)
    write_table("si_relation_all_arms",tabular(["Arm","Cosine","Recall@1","MRR","Margin","Accuracy"],[[tex(LABEL[m]),*[fmt(v) for v in row]] for m,row in avg.iterrows()]))
    iv=[]
    for j in inv:
        iv += [{"method":j['method'],"seed":j['representation_seed'],**r} for r in j['equivalence_summary']]
        app += [{"method":j['method'],"seed":j['representation_seed'],**r} for r in j['applicability_summary']]
        drift += [{"method":j['method'],"seed":j['representation_seed'],**r} for r in j['layer_drift_summary']]
    iv=pd.DataFrame(iv);app=pd.DataFrame(app).rename(columns={'control_family':'view_family'});drift=pd.DataFrame(drift)
    record(z.drop(columns=['dataset'],errors='ignore'),"si_retention_source_metrics","results/metrics/retention/zinc50k_mlm/*/metrics.json")
    record(iv,"si_invariance_source_metrics","results/metrics/fidelity/invariance/*/metrics.json")
    record(app,"si_semantics_source_metrics","results/metrics/fidelity/invariance/*/metrics.json")
    record(drift,"si_drift_source_metrics","results/metrics/fidelity/invariance/*/metrics.json")
    for name,q in [("invariance",iv),("semantic_applicability",app)]:
        a=q.groupby(["view_family","layer","method"],as_index=False).mean(numeric_only=True)
        rows=[]
        for (family,layer),g in a.groupby(["view_family","layer"]):
            vals=g.set_index('method')['mean_cosine']
            rows.append([tex(family.replace("alternative_periodic_cut","periodic-cut").replace("atom_order_permutation","atom-permutation").replace("randomized_smiles","random-SMILES").replace("marker_substituted","marker-substitute")),str(layer),*[fmt(vals.get(m,np.nan)) for m in ARMS]])
        write_table("si_"+name,tabular(["View","Layer"]+[tex(LABEL[m]) for m in ARMS],rows,long=True,caption=f"Layer-resolved {name.replace('_',' ')}: seed-mean cosine, descriptive only. Semantic controls have no target distance or property response.",label="tab:si-"+name,fontsize="footnotesize"))
    fig,axs=plt.subplots(2,2,figsize=(7.05,5.1),layout="constrained")
    for j,field in enumerate(["token_mean_nll","atom_group_mean_nll"]):
        avg=z.groupby('method')[field].mean().reindex(ARMS)
        axs[0,j].bar(range(7),avg,color=[COLORS[m] for m in ARMS]);axs[0,j].set_xticks(range(7),[LABEL[m] for m in ARMS],rotation=40,ha='right');axs[0,j].set_ylabel('NLL');panel(axs[0,j],'ab'[j],['ZINC token-weighted NLL','ZINC atom-group-equal NLL'][j])
    q=d.loc[d.method.isin(ARMS)&d.task.isin(['ESOL','FREESOLV','LIPOPHILICITY'])&d.label_fraction.eq(1)&d.input_view.eq('valid')]
    a=q.pivot_table(index='method',columns='task',values='mean').reindex(ARMS)
    ratio=np.log(a/a.loc['C'])
    im=axs[1,0].imshow(ratio.to_numpy(),aspect='auto',cmap='RdBu_r',vmin=-.23,vmax=.23)
    axs[1,0].set_xticks(range(3),['ESOL','FreeSolv','Lipophilicity'],rotation=20,ha='right');axs[1,0].set_yticks(range(7),[LABEL[m] for m in ARMS]);panel(axs[1,0],'c','Property log(error / C)');fig.colorbar(im,ax=axs[1,0],fraction=.045,pad=.03)
    # The source field names are fixed by the completed invariance evaluator.
    field='mean_checkpoint_cosine'
    a=drift.groupby(['method','layer'],as_index=False)[field].mean()
    for m in ARMS:
        q=a.loc[a.method.eq(m)];axs[1,1].plot(q.layer,q[field],'o-',ms=3,color=COLORS[m],label=LABEL[m])
    axs[1,1].set_xlabel('Layer (0 = embedding)');axs[1,1].set_ylabel('Cosine to fixed C');panel(axs[1,1],'d','PI1M coordinate drift')
    handles,labels=axs[1,1].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=7,fontsize=9,loc='outside lower center')
    save(fig,'supp_retention_and_drift')


def main():
    FIG.mkdir(parents=True,exist_ok=True);GEN.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7,'savefig.pad_inches':.04})
    figure2();figure3();figure4();figure5();figure6();figure6(32);tables();supplement()
    pd.DataFrame(SOURCES).drop_duplicates('source').to_csv(GEN/'asset_sources.csv',index=False)
    pd.DataFrame(LEDGER).to_csv(GEN/'number_ledger.csv',index=False)
    print(json.dumps({'figure_pdf_files':len(list(FIG.glob('*.pdf'))),'table_snippets':len(list(GEN.glob('*.tex'))),'source_files':len(pd.DataFrame(SOURCES).drop_duplicates('source')),'ledger_cells':len(LEDGER)},indent=2))


if __name__=='__main__':
    main()
