"""Generate every figure from result files (no numbers typed by hand).

Figures are written as vector PDFs to PLOS/figures/ (the figure files of PLOS/BSCAN_PLOS_ONE.tex).
If any input of a figure is a synthetic placeholder, the figure is overprinted with a red
"SYNTHETIC PLACEHOLDER - NOT A RESULT" banner, the same text is written into the PDF metadata (Keywords)
so that .scripts/final_qc.py can detect it in the file itself, and the figure is listed as such in
results/figures/figure_manifest.csv.  Every figure has exactly one manifest row per run: a figure whose
inputs do not exist yet gets a 'missing' row, and figure files of the previous run are deleted first, so a
stale file can never stand in for a figure that was not regenerated.

Usage:
    python .scripts/generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bscan.registry import load_registry, resolve  # noqa: E402

FIG = ROOT / "PLOS" / "figures"
OUT = ROOT / "results" / "figures"
MANIFEST: list[dict] = []
PLACEHOLDER_MARK = "SYNTHETIC PLACEHOLDER - NOT A RESULT"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
# colour-blind-safe palette (Okabe-Ito)
C = {"bscan_controlled": "#0072B2", "lcnn_lfcc": "#D55E00", "lfcc_gmm_controlled": "#009E73",
     "stats_hgb_controlled": "#7F7F7F", "mel_only": "#E69F00", "lfcc_only": "#CC79A7", "mfcc_only": "#56B4E9",
     "mel_mfcc": "#F0E442", "bscan_orig6s": "#000000", "stats_hgb_orig6s": "#BBBBBB",
     "dual_plain": "#D9D9D9", "dual_no_se": "#A6A6A6", "dual_no_tattn": "#595959"}
# readable names of test-time conditions (robustness perturbations and counterfactual probes)
COND = {"noise_snr20": "noise, 20 dB SNR", "noise_snr10": "noise, 10 dB SNR", "noise_snr5": "noise, 5 dB SNR",
        "mp3_64k": "MP3, 64 kbit/s", "mp3_32k": "MP3, 32 kbit/s", "aac_32k": "AAC, 32 kbit/s",
        "telephone": "telephone channel", "clip_+6dB": "clipping (+6 dB gain)",
        "speed_0.95": "speed \u00d70.95", "speed_1.05": "speed \u00d71.05", "pitch_+1st": "pitch +1 semitone",
        "pitch_-1st": "pitch \u22121 semitone", "reverb_rt0.3": "reverb, RT60 0.3 s", "reverb_rt0.6": "reverb, RT60 0.6 s",
        "append_silence_0.44": "append 0.44 s silence", "trim_trailing": "trim trailing silence",
        "add_dc_neg": "add DC offset", "remove_dc": "remove DC offset", "dither_-60dBFS": "add \u221260 dBFS noise"}
LABEL = {k: v["label"] for k, v in load_registry()["models"].items()}
SPLITNAME = {("P1", "internal_test"): "BF-SUST (internal test)", ("P1", "external_BF-MOZ"): "BF-MOZ",
             ("P1", "external_MEN"): "MEN (unseen generator)", ("P2", "internal_test"): "MEN (internal test)",
             ("P2", "external_BF-SUST"): "BF-SUST", ("P2", "external_BF-MOZ"): "BF-MOZ"}


def rel(path) -> str:
    """Source path relative to the repository root (the manifest never records local absolute paths)."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def save(fig, name: str, caption: str, status: str, sources: list[str]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"Title": f"{name}: {caption}"}
    if status != "real":
        fig.text(0.5, 0.5, PLACEHOLDER_MARK, color="red", alpha=0.35, fontsize=22,
                 ha="center", va="center", rotation=25, weight="bold", transform=fig.transFigure, zorder=1000)
        meta["Keywords"] = PLACEHOLDER_MARK
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata=meta)  # vector figure of the manuscript
    plt.close(fig)
    MANIFEST.append({"figure": name, "status": status, "caption": caption, "sources": ";".join(rel(x) for x in sources)})


def skip(name: str, caption: str, reason: str) -> None:
    """Manifest row for a figure that could not be drawn (its inputs do not exist yet); no file is written."""
    MANIFEST.append({"figure": name, "status": "missing", "caption": caption, "sources": reason})


def label_panels(axes, x: float = -0.02, y: float = 1.02) -> None:
    """Bold panel letters (A, B, C, ...) referenced by the figure captions."""
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.text(x, y, "ABCDEFGH"[i], transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")


def combine(sts: list[str]) -> str:
    return "missing" if "missing" in sts else ("placeholder" if "placeholder" in sts else "real")


# ----------------------------------------------------------------------------- diagrams (not data)
def box(ax, x, y, w, h, text, fc="#EAF2FB", ec="#0072B2", fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#333333"))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fs = 6.6
    box(ax, 0.000, 0.40, 0.120, 0.22, "Recording\n(any rate)\n→ 16 kHz\nmono", fc="#F4F4F4", ec="#555555", fs=fs)
    box(ax, 0.145, 0.30, 0.150, 0.42, "Controlled\npreprocessing\n• trim silence\n• repeat-pad\n• 6 s windows,\n  10 % overlap\n• DC removal\n• −60 dBFS dither",
        fc="#F4F4F4", ec="#555555", fs=fs)
    arrow(ax, 0.120, 0.51, 0.145, 0.51)
    for yb, name in ((0.66, "Mel spectrogram\n128 bands, dB re max\n+ Δ, ΔΔ\n3 × 128 × 188"),
                     (0.12, "LFCC\n40 linear filters, DCT\n+ Δ, ΔΔ\n3 × 40 × 188")):
        box(ax, 0.325, yb, 0.175, 0.22, name, fs=fs)
        arrow(ax, 0.295, 0.51, 0.325, yb + 0.11)
        box(ax, 0.530, yb, 0.170, 0.22, "SE-ResBlock 3→32\nmax-pool 2×2\nSE-ResBlock 32→64\nfrequency avg-pool", fs=fs)
        arrow(ax, 0.500, yb + 0.11, 0.530, yb + 0.11)
        box(ax, 0.725, yb + 0.03, 0.105, 0.16, "temporal\nattention\npooling → 64", fs=fs)
        arrow(ax, 0.700, yb + 0.11, 0.725, yb + 0.11)
        arrow(ax, 0.830, yb + 0.11, 0.855, 0.51)
    box(ax, 0.855, 0.33, 0.145, 0.36, "concat (128)\nFC 128, ReLU\ndropout 0.3\nFC 1 → logit\n\nrecording score\n= mean window\nlogit", fc="#FDF1E6", ec="#D55E00", fs=fs)
    ax.text(0.5, 0.985, "BSCAN: two parallel residual branches with squeeze-and-excitation and temporal attention, late fusion",
            ha="center", va="top", fontsize=7.5)
    save(fig, "Fig1", "Architecture and preprocessing", "real", ["src/bscan/models/bscan.py", "src/bscan/audio.py"])


def fig_corpus_cues():
    d = pd.read_csv(ROOT / "metadata/recording_descriptors.csv")
    m = pd.read_csv(ROOT / "metadata/unified_metadata.csv", usecols=["audio_path", "dataset_id", "label"], low_memory=False)
    d = d.merge(m, on="audio_path")
    w = pd.read_csv(ROOT / "results/phase3/window_descriptors.csv.gz", usecols=["scheme", "audio_path", "dc"])
    w = w[w.scheme == "trim_repeat"].groupby("audio_path").dc.mean().rename("dc_win").reset_index()
    d = d.merge(w, on="audio_path", how="left")
    groups = [("BF-SUST", 0), ("BF-SUST", 1), ("BF-MOZ", 0), ("BF-MOZ", 1), ("BF-NEWS", 0), ("MEN", 0), ("MEN", 1)]
    names = [f"{g}\n{'spoof' if l else 'bona fide'}" for g, l in groups]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    for ax, col, lab in ((axes[0], "duration_sec", "Duration (s)"), (axes[1], "trailing_silence_sec", "Trailing silence (s)"),
                         (axes[2], "dc_win", "DC offset after peak normalisation")):
        data = [d[(d.dataset_id == g) & (d.label == l)][col].dropna().to_numpy() for g, l in groups]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        for patch, (g, l) in zip(bp["boxes"], groups):
            patch.set_facecolor("#D55E00" if l else "#0072B2")
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_ylabel(lab)
    axes[0].set_ylim(0, 16)
    fig.tight_layout()
    label_panels(axes)
    save(fig, "Fig3", "Recording-level cues by corpus and class", "real",
         ["metadata/recording_descriptors.csv", "results/phase3/window_descriptors.csv.gz"])


def fig_shortcuts():
    r = pd.read_csv(ROOT / "results/phase3/model_visible_shortcut_diagnostics.csv")
    r = r[(r.model == "hgb") & (r.features == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6), sharey=True)
    schemes = [("orig6s", "original", "#000000"), ("trim_repeat", "trim + repeat-pad", "#E69F00"),
               ("controlled", "controlled", "#0072B2")]
    for ax, proto, splits in ((axes[0], "P1", ["internal_test", "external_BF-MOZ", "external_MEN"]),
                              (axes[1], "P2", ["internal_test", "external_BF-SUST", "external_BF-MOZ"])):
        x = np.arange(len(splits))
        for i, (sc, lab, col) in enumerate(schemes):
            v = [float(r[(r.protocol == proto) & (r.scheme == sc) & (r.test_split == s)].auc_file.iloc[0]) for s in splits]
            ax.bar(x + (i - 1) * 0.26, v, 0.26, label=lab, color=col, alpha=0.8)
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels([SPLITNAME[(proto, s)] for s in splits], fontsize=6.5)
        ax.set_title(f"{proto}: trained on {'BF-SUST' if proto == 'P1' else 'MEN'}")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("AUC without speech content")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    label_panels(axes)
    save(fig, "Fig4", "Label information available from content-agnostic descriptors", "real",
         ["results/phase3/model_visible_shortcut_diagnostics.csv"])


# ----------------------------------------------------------------------------- result figures
def fig_roc():
    models = ["bscan_controlled", "lcnn_lfcc", "lfcc_gmm_controlled", "stats_hgb_controlled"]
    splits = ["internal_test", "external_BF-MOZ", "external_MEN"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), sharey=True)
    sts, srcs = [], []
    for ax, sp in zip(axes, splits):
        for mdl in models:
            name = f"{mdl}__P1__seed42"
            st, path = resolve(name)
            sts.append(st)
            if path is None or not (path / f"predictions_{sp}.csv").exists():
                continue
            f = pd.read_csv(path / f"predictions_{sp}.csv")
            fpr, tpr, _ = roc_curve(f.label, f.score)
            ax.plot(fpr, tpr, color=C[mdl], lw=1.1, label=LABEL[mdl] + ("†" if st != "real" else ""))
            srcs.append(str(path / f"predictions_{sp}.csv"))
        ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="grey")
        ax.set_title(SPLITNAME[("P1", sp)])
        ax.set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(frameon=False, loc="lower right", fontsize=6)
    fig.tight_layout()
    label_panels(axes)
    save(fig, "Fig5", "ROC curves under P1", combine(sts), srcs)


def fig_cross():
    s = pd.read_csv(ROOT / "results/stats/cross_corpus_auc.csv")
    if s.empty:
        return skip("Fig6", "Cross-corpus AUC matrix", "results/stats/cross_corpus_auc.csv is empty")
    rows = [f"{LABEL[r.model]} ← {r.train_corpus}" for r in s.itertuples()]
    mat = s[["test_BF-SUST", "test_BF-MOZ", "test_MEN"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(rows) + 1.0))
    im = ax.imshow(mat, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] == mat[i, j]:
                ax.text(j, i, f"{mat[i, j]:.2f}" + ("†" if s.status.iloc[i] != "real" else ""), ha="center", va="center",
                        fontsize=6.5, color="white" if abs(mat[i, j] - 0.5) > 0.3 else "black")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["test: BF-SUST", "test: BF-MOZ", "test: MEN"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.04, label="AUC")
    fig.tight_layout()
    save(fig, "Fig6", "Cross-corpus AUC matrix", combine(list(s.status)), ["results/stats/cross_corpus_auc.csv"])


def fig_repr_ablation():
    s = pd.read_csv(ROOT / "results/stats/multiseed_summary.csv")
    groups = [("Representation", ["mel_only", "lfcc_only", "mfcc_only", "mel_mfcc", "bscan_controlled"]),
              ("Ablation", ["dual_plain", "dual_no_se", "dual_no_tattn", "bscan_controlled"])]
    splits = ["internal_test", "external_BF-MOZ", "external_MEN"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.7), sharey=True)
    sts = []
    for ax, (title, models) in zip(axes, groups):
        x = np.arange(len(splits))
        wdt = 0.8 / len(models)
        for i, mdl in enumerate(models):
            sub = s[(s.model == mdl) & (s.protocol == "P1")].set_index("split")
            v = [sub.auc_mean.get(sp, np.nan) for sp in splits]
            e = [sub.auc_sd.get(sp, np.nan) for sp in splits]
            sts += list(sub.status)
            ax.bar(x + (i - (len(models) - 1) / 2) * wdt, v, wdt, yerr=np.nan_to_num(e), capsize=1.5,
                   color=C.get(mdl, "#999999"), label=LABEL[mdl], alpha=0.85)
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels([SPLITNAME[("P1", sp)] for sp in splits], fontsize=6.5)
        ax.set_title(title + " (P1)")
        ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("AUC (mean ± SD over seeds)")
    fig.tight_layout()
    label_panels(axes)
    save(fig, "Fig7", "Representation study and trained ablation", combine(sts), ["results/stats/multiseed_summary.csv"])


def _analysis_frame(kind: str) -> tuple[pd.DataFrame, list[str]]:
    reg = load_registry()
    frames, sts = [], []
    for rn in reg["analyses"][kind]["runs"]:
        real = ROOT / "results/runs" / rn / f"{kind}_summary.csv"
        ph = ROOT / "results/runs_PLACEHOLDER" / rn / f"{kind}_summary.csv"
        if real.exists():
            f, st = pd.read_csv(real), "real"
        elif ph.exists():
            f, st = pd.read_csv(ph), "placeholder"
        else:
            continue
        f["run"], f["status"] = rn, st
        frames.append(f)
        sts.append(st)
    return (pd.concat(frames) if frames else pd.DataFrame()), sts


def fig_robustness_probes():
    rb, s1 = _analysis_frame("robustness")
    pr, s2 = _analysis_frame("probes")
    if rb.empty and pr.empty:
        return skip("Fig8", "Robustness and counterfactual probes", "no robustness_summary.csv or probes_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    if not rb.empty:
        ax = axes[0]
        conds = [c for c in rb.condition.unique() if c != "clean"]
        for rn, g in rb[rb.split == "internal_test"].groupby("run"):
            clean = float(g[g.condition == "clean"].auc.iloc[0])
            d = [float(g[g.condition == c].auc.iloc[0]) - clean for c in conds]
            ax.plot(d, range(len(conds)), "o", ms=3, color=C.get(rn.split("__")[0], "#999"), label=LABEL[rn.split("__")[0]])
        ax.axvline(0, lw=0.6, color="grey")
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([COND.get(c, c) for c in conds], fontsize=6)
        ax.set_xlabel("ΔAUC vs clean (BF-SUST internal test)")
        ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    if not pr.empty:
        ax = axes[1]
        probes = [c for c in pr.condition.unique() if c != "clean"]
        runs = list(pr.run.unique())
        wdt = 0.8 / max(1, len(runs))
        for i, rn in enumerate(runs):
            g = pr[(pr.run == rn) & (pr.split == "internal_test")].set_index("condition")
            v = [float(g.decision_flip_rate.get(p, np.nan)) * 100 for p in probes]
            ax.barh(np.arange(len(probes)) + (i - (len(runs) - 1) / 2) * wdt, v, wdt,
                    color=C.get(rn.split("__")[0], "#999"), label=LABEL[rn.split("__")[0]])
        ax.set_yticks(range(len(probes)))
        ax.set_yticklabels([COND.get(c, c) for c in probes], fontsize=6)
        ax.set_xlabel("Decisions flipped (%) (BF-SUST internal test)")
        ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    label_panels(axes)
    save(fig, "Fig8", "Robustness and counterfactual probes", combine(s1 + s2), ["robustness_summary.csv", "probes_summary.csv"])


def fig_errors_calibration():
    ge = ROOT / "results/analysis/error_groups.csv"
    if not ge.exists():
        return skip("Fig9", "Systematic error analysis of BSCAN on MEN", "results/analysis/error_groups.csv missing")
    g = pd.read_csv(ge)
    g = g[g.run == "bscan_controlled__P1__seed42"]
    if g.empty:
        return skip("Fig9", "Systematic error analysis of BSCAN on MEN", "no rows for bscan_controlled__P1__seed42")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    for ax, grouping, title in ((axes[0], "duration_tertile", "Duration"), (axes[1], "snr_proxy_tertile", "SNR proxy"),
                                (axes[2], "trailing_silence", "Trailing silence")):
        sub = g[(g.grouping == grouping) & (g.split == "external_MEN")]
        ax.bar(np.arange(len(sub)) - 0.2, sub.fnr * 100, 0.4, color="#D55E00", label="FNR (spoof missed)")
        ax.bar(np.arange(len(sub)) + 0.2, sub.fpr * 100, 0.4, color="#0072B2", label="FPR (bona fide flagged)")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub.level, fontsize=6.5)
        ax.set_title(f"{title} (MEN)")
    axes[0].set_ylabel("Error rate (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    label_panels(axes)
    save(fig, "Fig9", "Systematic error analysis of BSCAN on MEN", combine(list(g.status.unique())),
         ["results/analysis/error_groups.csv"])


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    st, path = resolve("bscan_controlled__P1__seed42")
    if path is None or not (path / "train_log.csv").exists():
        plt.close(fig)
        return skip("S1_Fig", "Training and validation loss of BSCAN (P1, seed 42)",
                    "train_log.csv of bscan_controlled__P1__seed42 missing")
    t = pd.read_csv(path / "train_log.csv")
    ax.plot(t.epoch, t.train_loss, label="train", color="#0072B2")
    ax.plot(t.epoch, t.val_loss, label="validation", color="#D55E00")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "S1_Fig", "Training and validation loss of BSCAN (P1, seed 42)", st, [str(path / "train_log.csv")])


def fig_confusion():
    """Confusion matrices of BSCAN (P1, seed 42) at the validation EER threshold, one per evaluation tier."""
    st, path = resolve("bscan_controlled__P1__seed42")
    if path is None:
        return skip("Fig_confusion", "Confusion matrices of BSCAN across the evaluation tiers",
                    "run bscan_controlled__P1__seed42 missing")
    thr = json.loads((path / "metrics.json").read_text()).get("threshold_logit")
    splits = [("internal_test", "Tier 1: internal test (BF-SUST)"), ("external_BF-MOZ", "Tier 2: independent (Mozilla)"),
              ("external_MEN", "Tier 3: cross-dataset (Mendeley)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    srcs = []
    for ax, (sp, title) in zip(axes, splits):
        f = path / f"predictions_{sp}.csv"
        if thr is None or not f.exists():
            ax.axis("off")
            continue
        d = pd.read_csv(f)
        pred = (d.score >= thr).astype(int)
        cm = np.array([[int(((d.label == 0) & (pred == 0)).sum()), int(((d.label == 0) & (pred == 1)).sum())],
                       [int(((d.label == 1) & (pred == 0)).sum()), int(((d.label == 1) & (pred == 1)).sum())]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred. bona fide", "pred. spoof"], fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["bona fide", "spoof"], fontsize=7)
        ax.set_title(title, fontsize=8)
        srcs.append(str(f))
    label_panels(axes)
    fig.tight_layout()
    save(fig, "Fig_confusion", "Confusion matrices of BSCAN across the evaluation tiers", st, srcs)


# file name of every generated figure (PLOS/figures/<name>.pdf)
FIGURES = ["Fig1", "Fig3", "Fig4", "Fig5", "Fig6", "Fig7", "Fig8", "Fig9", "S1_Fig", "Fig_confusion"]


def main() -> None:
    # remove the figure files of the previous run (and any listed in the previous manifest) so that only
    # figures regenerated now exist; other files in PLOS/figures/ are left alone
    old = OUT / "figure_manifest.csv"
    names = set(FIGURES) | (set(pd.read_csv(old).figure.astype(str)) if old.exists() else set())
    for n in names:
        (FIG / f"{n}.pdf").unlink(missing_ok=True)
    for f in (fig_architecture, fig_corpus_cues, fig_shortcuts, fig_roc, fig_cross, fig_repr_ablation,
              fig_robustness_probes, fig_errors_calibration, fig_training_curves, fig_confusion):
        try:
            f()
        except Exception as exc:  # noqa: BLE001
            MANIFEST.append({"figure": f.__name__, "status": "error", "caption": str(exc), "sources": ""})
            print(f"[figure error] {f.__name__}: {exc}")
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(MANIFEST)
    m.to_csv(OUT / "figure_manifest.csv", index=False)
    print(m[["figure", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
