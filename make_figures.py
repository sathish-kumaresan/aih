from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

REPORTS = Path("results/reports")
RUNS = Path("results/runs")
FIGS = Path("results/figures")
PFTAS_PATH = Path("results/features/pftas_400X.parquet")
FIGS.mkdir(parents=True, exist_ok=True)

MODELS = [1, 2, 3]
MODEL_NAMES = {
    1: "PFTAS + RBF SVM",
    2: "EfficientNetV2-S",
    3: "EfficientNetV2-S + CBAM",
}
MODEL_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
SPLITS = ["honest", "provided"]
SPLIT_TITLE = {
    "honest": "Honest (disjoint)",
    "provided": "Provided (leaked)",
}
LEVELS = ["image", "patient"]
SEEDS = [1, 2, 3, 4, 5]
W = 13.0

SUBTYPE_ORDER = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
SUBTYPE_FULL = {
    "A": "Adenosis (B)", "F": "Fibroadenoma (B)",
    "PT": "Phyllodes T. (B)", "TA": "Tubular A. (B)",
    "DC": "Ductal C. (M)", "LC": "Lobular C. (M)",
    "MC": "Mucinous C. (M)", "PC": "Papillary C. (M)",
}

TABLE_METRICS = [
    ("auroc", "AUROC"),
    ("sensitivity", "Sens."),
    ("specificity", "Spec."),
    ("f1", "F1"),
    ("accuracy", "Acc."),
]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
})


def save_fig(fig, name):
    out = FIGS / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def load_predictions(m, split, seed):
    return pd.read_parquet(RUNS / f"tier{m}" / split / f"seed{seed}" / "predictions.parquet")


def load_all_predictions():
    return {(m, sp, sd): load_predictions(m, sp, sd)
            for m in MODELS for sp in SPLITS for sd in SEEDS}


def load_summary(m, split):
    return pd.read_csv(REPORTS / f"tier{m}_{split}_summary.csv")


def load_subtype_map():
    pf = pd.read_parquet(PFTAS_PATH)
    return dict(zip(pf["path"], pf["subtype"]))


def parse_confusion_txt(path):
    pat = re.compile(r"seed (\d+) (image|patient): \[\[(\d+), (\d+)\], \[(\d+), (\d+)\]\]")
    out = {}
    for ln in path.read_text().splitlines():
        m = pat.search(ln)
        if m:
            tn, fp, fn, tp = (int(g) for g in m.groups()[2:])
            out[(m.group(2), int(m.group(1)))] = np.array([[tn, fp], [fn, tp]], dtype=int)
    return out


def metric_meanstd(model, split, metric, level):
    df = load_summary(model, split)
    sub = df[(df["metric"] == metric) & (df["level"] == level)]
    if sub.empty:
        return None, None
    r = sub.iloc[0]
    return float(r["mean"]), float(r["std"])


def per_seed_accuracy(model, split, level):
    cms = parse_confusion_txt(REPORTS / f"tier{model}_{split}_confusion.txt")
    return np.array([cms[(level, s)].diagonal().sum() / cms[(level, s)].sum() for s in SEEDS])


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def table_cells():
    for split in SPLITS:
        for level in LEVELS:
            for m in MODELS:
                cells = []
                for metric, _ in TABLE_METRICS:
                    if metric == "accuracy":
                        accs = per_seed_accuracy(m, split, level)
                        cells.append(fmt(accs.mean(), accs.std()))
                    else:
                        mean, std = metric_meanstd(m, split, metric, level)
                        cells.append(fmt(mean, std) if mean is not None else "—")
                yield split, level, m, cells


def write_table_tex():
    cols = [lab for _, lab in TABLE_METRICS]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l " + "c" * len(cols) + "}",
             r"\toprule", "Model & " + " & ".join(cols) + r" \\"]
    prev = None
    for split, level, m, cells in table_cells():
        if (split, level) != prev:
            lines.append(r"\midrule")
            label = f"{SPLIT_TITLE[split]} -- {level.title()} level"
            lines.append(r"\multicolumn{" + str(1 + len(cols))
                         + r"}{l}{\textit{" + label + r"}} \\")
            prev = (split, level)
        lines.append(f"{MODEL_NAMES[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Five-seed mean $\pm$ std for the three model architectures"
              r" under both evaluation protocols. ``Acc.'' is plain accuracy computed"
              r" from the summed per-seed confusion matrix.}",
              r"\label{tab:master}", r"\end{table}"]
    out = FIGS / "table_master_results.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def render_table_png():
    cols = [lab for _, lab in TABLE_METRICS]
    rows = [[""] + cols]
    bg = ["#dadfe1"]
    bold = {0}
    prev = None
    for split, level, m, cells in table_cells():
        if (split, level) != prev:
            rows.append([f"{SPLIT_TITLE[split]} — {level.title()} level"] + [""] * len(cols))
            bg.append("#c0d6e4")
            bold.add(len(rows) - 1)
            prev = (split, level)
        rows.append([MODEL_NAMES[m]] + cells)
        bg.append("white")

    fig, ax = plt.subplots(figsize=(W, len(rows) * 0.32))
    ax.axis("off")
    tbl = ax.table(cellText=rows, loc="center", cellLoc="center",
                   colWidths=[0.35] + [0.13] * len(cols))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for i, color in enumerate(bg):
        for j in range(len(rows[i])):
            cell = tbl[i, j]
            cell.set_facecolor(color)
            cell.set_edgecolor("#bbbbbb")
            if i in bold:
                cell.set_text_props(weight="bold")
        tbl[i, 0].set_text_props(ha="left")
    return fig


def fig_leakage_gap():
    fig, axes = plt.subplots(1, 2, figsize=(W, 4.5), sharey=True)
    width = 0.36
    for ax, level in zip(axes, LEVELS):
        x = np.arange(len(MODELS))
        h_means, h_stds = zip(*(metric_meanstd(m, "honest", "auroc", level) for m in MODELS))
        p_means, p_stds = zip(*(metric_meanstd(m, "provided", "auroc", level) for m in MODELS))
        ax.bar(x - width / 2, h_means, width, yerr=h_stds,
               label="Honest", color="#3b78c2", capsize=4)
        ax.bar(x + width / 2, p_means, width, yerr=p_stds,
               label="Provided", color="#c25a3b", capsize=4)
        for xi, mh, mp in zip(x, h_means, p_means):
            ax.annotate(f"Δ = +{mp - mh:.3f}", xy=(xi, max(mh, mp) + 0.03),
                        ha="center", fontsize=9, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_NAMES[m] for m in MODELS],
                           rotation=12, ha="right", fontsize=9)
        ax.set_ylim(0.6, 1.15)
        ax.set_ylabel("AUROC (mean ± std)")
        ax.set_title(f"{level.title()}-level AUROC, honest vs. provided")
    axes[0].legend(loc="upper left", frameon=True, edgecolor="0.8")
    fig.suptitle("Leakage inflation: AUROC on the provided split is optimistic by the Δ shown",
                 y=1.02)
    fig.tight_layout()
    return fig


def fig_confusion_patient():
    fig, axes = plt.subplots(3, 2, figsize=(W, 13))
    for r, m in enumerate(MODELS):
        for c, split in enumerate(SPLITS):
            cms = parse_confusion_txt(REPORTS / f"tier{m}_{split}_confusion.txt")
            mean = np.stack([cms[("patient", sd)] for sd in SEEDS]).mean(axis=0)
            row_tot = mean.sum(axis=1)
            pct = mean / row_tot[:, None] * 100
            disp = np.round(mean).astype(int)
            annot = np.array([[f"{disp[i, j]}\n({pct[i, j]:.0f}%)" for j in range(2)]
                              for i in range(2)])
            ax = axes[r, c]
            sns.heatmap(mean, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                        annot_kws={"size": 14, "weight": "bold"},
                        xticklabels=["Benign", "Malignant"],
                        yticklabels=["Benign", "Malignant"],
                        linewidths=0.5, linecolor="white")
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title(f"{MODEL_NAMES[m]}   ·   {SPLIT_TITLE[split]}", weight="bold")
    fig.suptitle("Patient-level confusion matrices (per-seed average across 5 seeds)",
                 y=1.005, fontsize=12, weight="bold")
    fig.text(0.5, -0.01,
             "Each cell = average patient count per seed (rounded), with row percentage.\n"
             "Honest split tests on 17 patient-disjoint patients; "
             "provided split tests on 80 patients with 80/80 leakage to training.",
             ha="center", va="top", fontsize=9, style="italic", color="#444444")
    fig.tight_layout()
    return fig


def fig_proba_histograms(preds):
    fig, axes = plt.subplots(2, 3, figsize=(W, 7.5), sharey="row")
    classes = [(0, "tab:blue", "Benign (true)"), (1, "tab:red", "Malignant (true)")]
    for r, split in enumerate(SPLITS):
        for c, m in enumerate(MODELS):
            ax = axes[r, c]
            all_p = np.concatenate([preds[(m, split, sd)]["proba"] for sd in SEEDS])
            all_y = np.concatenate([preds[(m, split, sd)]["label"] for sd in SEEDS])
            for lab, color, name in classes:
                ax.hist(all_p[all_y == lab], bins=40, color=color, alpha=0.55,
                        density=True, label=name if (r == 0 and c == 0) else None)
            ax.axvline(0.5, color="black", lw=0.8, ls="--",
                       label="Decision threshold" if (r == 0 and c == 0) else None)
            auroc, _ = metric_meanstd(m, split, "auroc", "image")
            ax.text(0.03, 0.96, f"AUROC = {auroc:.3f}", transform=ax.transAxes,
                    fontsize=9, color="#777777", ha="left", va="top")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Predicted P(malignant)")
            ax.set_title(MODEL_NAMES[m])
            if c == 0:
                ax.set_ylabel("Density")
            if r == 0 and c == 0:
                ax.legend(loc="upper center", frameon=True, edgecolor="0.8", fontsize=8)
        axes[r, 0].annotate(split.capitalize(),
                            xy=(-0.28, 0.5), xycoords="axes fraction",
                            ha="center", va="center", rotation=90,
                            fontsize=12, weight="bold", color="#333333")
    fig.suptitle("Predicted P(malignant) distributions by true class. "
                 "Overlap of blue and red = the model cannot separate the classes; "
                 "clean bimodal separation = confident discrimination.",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    return fig


def fig_subtype_error_rate(preds, subtype_map):
    n = {s: 0 for s in SUBTYPE_ORDER}
    err = {m: {s: 0 for s in SUBTYPE_ORDER} for m in MODELS}
    for m in MODELS:
        for sd in SEEDS:
            df = preds[(m, "honest", sd)]
            sub = df["path"].map(subtype_map)
            wrong = df["label"] != df["pred"]
            for s in SUBTYPE_ORDER:
                mask = sub == s
                err[m][s] += int((wrong & mask).sum())
                if m == MODELS[0]:
                    n[s] += int(mask.sum())

    fig, ax = plt.subplots(figsize=(W, 6))
    x = np.arange(len(SUBTYPE_ORDER))
    width = 0.26
    for i, m in enumerate(MODELS):
        rates = [100 * err[m][s] / n[s] if n[s] > 0 else 0 for s in SUBTYPE_ORDER]
        bars = ax.bar(x + (i - 1) * width, rates, width,
                      color=MODEL_COLORS[m], label=MODEL_NAMES[m],
                      edgecolor="white", lw=0.5)
        for b, s in zip(bars, SUBTYPE_ORDER):
            if err[m][s] > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                        f"{err[m][s]}", ha="center", va="bottom",
                        fontsize=7.5, color="#333333")
    for j, s in enumerate(SUBTYPE_ORDER):
        if n[s] == 0:
            ax.axvspan(j - 0.45, j + 0.45, color="#eeeeee", alpha=0.5, zorder=0)
            ax.text(j, 8, "n = 0\n(absent from\nhonest test)",
                    ha="center", va="bottom", fontsize=8,
                    style="italic", color="#888888")
    ax.set_xticks(x)
    ax.set_xticklabels([SUBTYPE_FULL[s] for s in SUBTYPE_ORDER], rotation=22, ha="right")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Honest-protocol error rate per BreaKHis subtype "
                 "(numbers above bars = absolute misclassified images, 5-seed sum)")
    ax.set_ylim(0, max(105, ax.get_ylim()[1]))
    ax.legend(loc="upper right", frameon=True, edgecolor="0.8")
    fig.tight_layout()
    return fig


def fig_qualitative(preds, subtype_map):
    df = preds[(2, "honest", 1)].copy()
    df["subtype"] = df["path"].map(subtype_map)
    rows_spec = [
        (1, 1, True,  "TP (M → M)", "#2e8b57"),
        (0, 0, False, "TN (B → B)", "#2e8b57"),
        (0, 1, True,  "FP (B → M)", "#c25a3b"),
        (1, 0, False, "FN (M → B)", "#c25a3b"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(W, W * 1.05))
    for r, (lbl, pred, biggest, text, frame) in enumerate(rows_spec):
        match = df[(df["label"] == lbl) & (df["pred"] == pred)]
        picks = (match.nlargest if biggest else match.nsmallest)(4, "proba")
        for c in range(4):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(frame); sp.set_linewidth(2.5)
            if c < len(picks):
                row = picks.iloc[c]
                ax.imshow(np.asarray(Image.open(row["path"]).convert("RGB")))
                truth = "M" if row["label"] == 1 else "B"
                ax.text(0.5, 1.10,
                        f"P(M) = {row['proba']:.2f}  |  true: {truth}  |  "
                        f"{SUBTYPE_FULL[row['subtype']]}",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
                ax.text(0.5, 1.02, f"(pt. {row['patient_id']})",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=6, color="#888888")
        axes[r, 0].text(-0.04, 0.5, text, transform=axes[r, 0].transAxes,
                        rotation=90, ha="right", va="center",
                        fontsize=11, weight="bold", color=frame)
    fig.suptitle(f"Qualitative examples — {MODEL_NAMES[2]}, honest split, seed 1.   "
                 "Each row = one outcome category, showing the four most-confident "
                 "examples.   Green border = correct, red = wrong.",
                 fontsize=11, y=1.005)
    fig.text(0.5, -0.015,
             "Note: rows 3 (FP) and 4 (FN) show that when this model is wrong, it is "
             "highly confident in being wrong (P(M) ≈ 1.0 for benign tissue; "
             "P(M) ≈ 0.0 for cancer). This argues against autonomous deployment and "
             "in favor of triage-only use with human review of low-confidence cases.",
             ha="center", va="top", fontsize=9, style="italic", color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def main():
    preds = load_all_predictions()
    smap = load_subtype_map()
    figs = [
        ("table_master_results",        lambda: render_table_png()),
        ("leakage_gap",                 lambda: fig_leakage_gap()),
        ("confusion_matrices_patient",  lambda: fig_confusion_patient()),
        ("proba_histograms",            lambda: fig_proba_histograms(preds)),
        ("subtype_error_rate",          lambda: fig_subtype_error_rate(preds, smap)),
        ("qualitative_grid",            lambda: fig_qualitative(preds, smap)),
    ]
    outputs = [write_table_tex()] + [save_fig(fn(), name) for name, fn in figs]

    print("\n" + "=" * 70)
    print(f"Generated {len(outputs)} artefacts in {FIGS}/")
    print("=" * 70)
    for p in outputs:
        print(f"  {p.name:<38}  {p.stat().st_size / 1024:>8.1f} KB")
    print("\nRe-runs triggered: NONE.  Compute units consumed: 0.")


if __name__ == "__main__":
    main()
