"""Generate the data figures of the manuscript as vector PDF files.

Figure 3  Sampling regimes: speech segments, uniform sampling and speech-anchored
          sampling on a schematic 60 s example (fig_sampling_regimes_*.pdf).
Figure 5  Results of the five configurations on MLVU (fig_configuration_results_*.pdf).
Figure 6  Change of C4 and C5 relative to C1 (fig_segmentation_effect_*.pdf).
Figure 8  Spread of the scores along the three design axes (fig_effect_magnitude_*.pdf).
Figure 9  Comparison with published end-to-end models (fig_state_of_the_art.pdf).

The scores are the values reported in the manuscript (Figure 5 and Tables 6 and 9),
and every difference shown in Figures 6 and 8 is derived from them below. The
scores of the published models are those of the MLVU benchmark paper (Zhou et al.,
CVPR 2025). Text is set in TeX Gyre Pagella, the OpenType version of the Palatino
face used by the MDPI class; if the font files are not found, the default font
is used. Each panel is written to its own file so that LaTeX can set the (a), (b)
and (c) captions below each panel.

Usage:
    python make_figures.py output_folder [--fonts folder_with_texgyrepagella_ttf]

Requirements: matplotlib.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ap = argparse.ArgumentParser(description="Generate the data figures of the manuscript.")
ap.add_argument("out", help="output folder")
ap.add_argument("--fonts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"),
                help="folder with the texgyrepagella-*.ttf files")
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)

font_files = glob.glob(os.path.join(args.fonts, "texgyrepagella-*.ttf"))
for f in font_files:
    fm.fontManager.addfont(f)
FAMILY = fm.FontProperties(fname=font_files[0]).get_name() if font_files else "DejaVu Serif"
if not font_files:
    print("TeX Gyre Pagella not found; using the default serif font")

plt.rcParams.update({
    "font.family": FAMILY,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#5F6368",
    "axes.linewidth": 0.6,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
})

LW_IN = 394.35522 / 72.27  # \linewidth of the MDPI layout in inches
COL = {"C1": "#1F4E79", "C2": "#7BA7CC", "C3": "#BDD3E8", "C4": "#C0392B", "C5": "#E8907E"}
NAMES = {"C1": "C1  Full fusion", "C2": "C2  Captions only", "C3": "C3  Transcript only",
         "C4": "C4  Speech-anchored, fixed", "C5": "C5  Speech-anchored, variable"}
CONFIGS = ["C1", "C2", "C3", "C4", "C5"]
TR = {"C1": 80.22, "C2": 59.34, "C3": 60.44, "C4": 82.42, "C5": 81.32}
NQA = {"C1": 56.67, "C2": 41.67, "C3": 13.33, "C4": 50.00, "C5": 51.67}
VS = {"C1": 4.28, "C2": 4.10, "C3": 3.38, "C4": 4.63, "C5": 4.55}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5
H5 = 2.35
fig = plt.figure(figsize=(0.62 * LW_IN, H5))
ax = fig.add_axes([0.13, 0.12, 0.85, 0.84])
w = 0.16
for i, c in enumerate(CONFIGS):
    for g, vals in enumerate([TR, NQA]):
        x = g + (i - 2) * w
        ax.bar(x, vals[c], width=w, color=COL[c], edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(x, vals[c] + 1.5, f"{vals[c]:.2f}", rotation=90, ha="center", va="bottom", fontsize=6.5)
ax.axhline(16.7, color="#5F6368", linestyle=(0, (4, 3)), linewidth=0.8, zorder=4)
ax.set_xticks([0, 1], ["TR", "NQA"])
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
ax.yaxis.grid(True, zorder=0)
save(fig, "fig_configuration_results_a.pdf")

fig = plt.figure(figsize=(0.36 * LW_IN, H5))
ax = fig.add_axes([0.22, 0.12, 0.75, 0.84])
for i, c in enumerate(CONFIGS):
    ax.bar(i, VS[c], width=0.62, color=COL[c], zorder=3)
    ax.text(i, VS[c] + 0.1, f"{VS[c]:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(range(5), CONFIGS)
ax.set_ylim(0, 6.2)
ax.set_ylabel("Judge score (0--10)".replace("--", "–"))
ax.yaxis.grid(True, zorder=0)
save(fig, "fig_configuration_results_b.pdf")

fig = plt.figure(figsize=(LW_IN, 0.42))
handles = [Patch(facecolor=COL[c], label=NAMES[c]) for c in CONFIGS] + [
    Line2D([], [], color="#5F6368", linestyle=(0, (4, 3)), linewidth=0.8, label="Random baseline, 16.7%")]
fig.legend(handles=handles, loc="center", ncol=3, frameon=False, handlelength=1.4,
           columnspacing=1.6, handletextpad=0.5)
save(fig, "fig_configuration_results_legend.pdf")

# ---------------------------------------------------------------- Figure 6
H6 = 2.15
fig = plt.figure(figsize=(0.60 * LW_IN, H6))
ax = fig.add_axes([0.15, 0.11, 0.83, 0.86])
d = {c: (round(TR[c] - TR["C1"], 2), round(NQA[c] - NQA["C1"], 2)) for c in ("C4", "C5")}
w = 0.34
for i, c in enumerate(["C4", "C5"]):
    for g in range(2):
        x = g + (i - 0.5) * w
        v = d[c][g]
        ax.bar(x, v, width=w, color=COL[c], edgecolor="white", linewidth=0.4, zorder=3,
               label=(f"{c}  fixed windows" if c == "C4" else f"{c}  variable windows") if g == 0 else None)
        ax.text(x, v + (0.35 if v > 0 else -0.35), f"{v:+.2f}".replace("-", "−"),
                ha="center", va="bottom" if v > 0 else "top", fontsize=6.5)
ax.axhline(0, color="#5F6368", linewidth=0.7, zorder=4)
ax.set_xticks([0, 1], ["TR", "NQA"])
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-9, 5)
ax.set_ylabel("Change vs. C1 (pp)")
ax.yaxis.grid(True, zorder=0)
ax.legend(loc="lower left", frameon=False)
save(fig, "fig_segmentation_effect_a.pdf")

fig = plt.figure(figsize=(0.38 * LW_IN, H6))
ax = fig.add_axes([0.24, 0.11, 0.73, 0.86])
for i, (c, v) in enumerate([(c, round(VS[c] - VS["C1"], 2)) for c in ("C4", "C5")]):
    ax.bar(i, v, width=0.5, color=COL[c], zorder=3)
    ax.text(i, v + 0.008, f"+{v:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks([0, 1], ["C4", "C5"])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0, 0.46)
ax.set_ylabel("Change vs. C1 (score points)")
ax.yaxis.grid(True, zorder=0)
save(fig, "fig_segmentation_effect_b.pdf")

# ---------------------------------------------------------------- Figure 8
H8 = 2.2
cats = ["Channel composition\nC1 vs. C2 vs. C3", "Sampling criterion\nC1 vs. C4", "Window granularity\nC4 vs. C5"]


def spread(scores, configs):
    vals = [scores[c] for c in configs]
    return round(max(vals) - min(vals), 2)


axes_cfg = [("C1", "C2", "C3"), ("C1", "C4"), ("C4", "C5")]
tr = [spread(TR, a) for a in axes_cfg]
nqa = [spread(NQA, a) for a in axes_cfg]
vs = [spread(VS, a) for a in axes_cfg]
ypos = [2, 1, 0]
BOTTOM, TOP = 0.21, 0.97
fig = plt.figure(figsize=(0.66 * LW_IN, H8))
ax = fig.add_axes([0.33, BOTTOM, 0.64, TOP - BOTTOM])
h = 0.32
for y, a_, b_ in zip(ypos, tr, nqa):
    ax.barh(y + h / 2, a_, height=h, color=COL["C1"], zorder=3)
    ax.barh(y - h / 2, b_, height=h, color=COL["C2"], zorder=3)
    ax.text(a_ + 0.8, y + h / 2, f"{a_:.2f}", va="center", fontsize=6.5)
    ax.text(b_ + 0.8, y - h / 2, f"{b_:.2f}", va="center", fontsize=6.5)
ax.set_yticks(ypos, cats)
ax.set_ylim(-0.6, 2.6)
ax.set_xlim(0, 50)
ax.set_xlabel("Spread in accuracy (pp)")
ax.xaxis.grid(True, zorder=0)
ax.legend(handles=[Patch(facecolor=COL["C1"], label="TR"),
                   Patch(facecolor=COL["C2"], label="NQA")], loc="lower right", frameon=False)
save(fig, "fig_effect_magnitude_a.pdf")

fig = plt.figure(figsize=(0.32 * LW_IN, H8))
ax = fig.add_axes([0.08, BOTTOM, 0.86, TOP - BOTTOM])
for y, v in zip(ypos, vs):
    ax.barh(y, v, height=0.42, color=COL["C4"], zorder=3)
    ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=6.5)
ax.set_yticks(ypos, ["", "", ""])
ax.set_ylim(-0.6, 2.6)
ax.set_xlim(0, 1.15)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("Spread in score points")
ax.xaxis.grid(True, zorder=0)
save(fig, "fig_effect_magnitude_b.pdf")

# ---------------------------------------------------------------- Figure 9
# Published end-to-end models: (name, TR, NQA, VS, label offset); values from Zhou et al. (MLVU)
pub = [("GPT-4o", 83.7, 42.9, 4.94, (9, -14)), ("LLaVA-OneVision", 83.5, 46.7, 3.75, (-22, 14)),
       ("LongVA", 81.3, 46.7, 2.90, (-30, -16)), ("Video-CCAM", 79.1, 45.0, 2.65, (-40, 7)),
       ("Video-XL", 78.0, 50.0, 3.40, (-36, 8))]
ours = [(c, TR[c], NQA[c], VS[c], off) for c, off in (("C1", (9, 4)), ("C4", (10, 4)), ("C5", (10, 4)))]
fig = plt.figure(figsize=(0.82 * LW_IN, 0.55 * LW_IN))
ax = fig.add_axes([0.11, 0.14, 0.86, 0.84])
size = lambda v: 9.0 * v ** 2
for n, x, y, v, off in pub:
    ax.scatter(x, y, s=size(v), facecolor="white", edgecolor=COL["C1"], linewidth=1.1, zorder=3)
    ax.annotate(f"{n}\nVS {v:.2f}", (x, y), xytext=off, textcoords="offset points", fontsize=6.8,
                color=COL["C1"], va="center")
for n, x, y, v, off in ours:
    ax.scatter(x, y, s=size(v), facecolor=COL["C4"], edgecolor="#7B241C", linewidth=0.8, alpha=0.9, zorder=3)
    ax.annotate(f"{n}\nVS {v:.2f}", (x, y), xytext=off, textcoords="offset points", fontsize=6.8,
                color="#7B241C", fontweight="bold", va="center")
ax.set_xlim(76, 86.5)
ax.set_ylim(39, 60)
ax.set_xlabel("TR accuracy (%)")
ax.set_ylabel("NQA accuracy (%)")
ax.grid(True, zorder=0)
ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=COL["C1"],
                          markersize=7, label="Published end-to-end models"),
                   Line2D([], [], marker="o", linestyle="", markerfacecolor=COL["C4"], markeredgecolor="#7B241C",
                          markersize=7, label="Configurations of the proposed pipeline")],
          loc="lower left", frameon=False)
ax.text(86.3, 59.2, "Marker area is proportional to the squared summarization score", ha="right",
        fontsize=6.3, color="#5F6368", va="top")
save(fig, "fig_state_of_the_art.pdf")

# ---------------------------------------------------------------- Figure 3
SEGS = [("S1", 0.0, 6.5), ("S2", 7.0, 19.0), ("S3", 19.5, 23.5), ("S4", 24.0, 42.0), ("S5", 50.0, 60.0)]
SIL = (42.0, 50.0)
BLUE, RED, GREY = COL["C1"], COL["C4"], "#5F6368"
PW = LW_IN
XL = (-1.5, 66.5)


def timeline_axes(height, show_time):
    fig = plt.figure(figsize=(PW, height))
    ax = fig.add_axes([0.01, 0.36 if show_time else 0.02, 0.98, 0.62 if show_time else 0.96])
    ax.set_xlim(*XL)
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    if show_time:
        ax.set_xticks(range(0, 61, 10))
        ax.set_xlabel("Time (s)", labelpad=2)
        ax.spines["bottom"].set_bounds(0, 60)
    else:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    return fig, ax


fig, ax = timeline_axes(0.62, False)
ax.set_ylim(0, 1)
for name, a, b in SEGS:
    ax.add_patch(Rectangle((a, 0.18), b - a, 0.64, facecolor="#BDD3E8", edgecolor=BLUE, linewidth=0.8))
    ax.text((a + b) / 2, 0.5, f"{name}\n{b - a:.1f} s", ha="center", va="center", fontsize=7)
ax.add_patch(Rectangle((SIL[0], 0.18), SIL[1] - SIL[0], 0.64, facecolor="#F1F3F4", edgecolor=GREY,
                       linewidth=0.8, linestyle=(0, (3, 2))))
ax.text(46, 0.5, "silence\n8.0 s", ha="center", va="center", fontsize=7, color=GREY)
save(fig, "fig_sampling_regimes_a.pdf")

fig, ax = timeline_axes(0.70, False)
ax.set_ylim(-1, 1.25)
ax.plot([0, 60], [0, 0], color=GREY, linewidth=0.7)
ax.plot(range(0, 60, 2), [0] * 30, linestyle="", marker="v", markersize=4.2, color=BLUE)
for xb in (0, 30, 60):
    ax.plot([xb, xb], [-0.8, 0.8], color=RED, linestyle=(0, (3, 2)), linewidth=0.8)
ax.text(15, 0.95, "window 1", ha="center", fontsize=7, color=RED)
ax.text(45, 0.95, "window 2", ha="center", fontsize=7, color=RED)
ax.text(61.2, 0, "30 frames", va="center", fontsize=7, color=BLUE, fontweight="bold")
save(fig, "fig_sampling_regimes_b.pdf")

fig, ax = timeline_axes(1.08, True)
ax.set_ylim(-1, 1.25)
ax.plot([0, 60], [0, 0], color=GREY, linewidth=0.7)
anch = [3.25, 11.0, 15.0, 21.5, 28.5, 33.0, 37.5, 50 + 10 / 3, 50 + 20 / 3]  # Equation 2: t_k = t_s + k d / (n + 1)
ref = [42 + 8 / 3, 42 + 16 / 3]
ax.plot(anch, [0] * len(anch), linestyle="", marker="v", markersize=5, color=BLUE)
ax.plot(ref, [0] * len(ref), linestyle="", marker="v", markersize=5, markerfacecolor="white",
        markeredgecolor=GREY, markeredgewidth=0.9)
for xb in (0, 6.5, 7.0, 19.0, 19.5, 23.5, 24.0, 42.0, 50.0, 60.0):
    ax.plot([xb, xb], [-0.8, 0.8], color=RED, linestyle=(0, (3, 2)), linewidth=0.8)
for (name, a, b), n in zip(SEGS, [1, 2, 1, 3, 2]):
    ax.text((a + b) / 2, 0.95, str(n), ha="center", fontsize=7, color=BLUE, fontweight="bold")
ax.text(46, 0.95, "2", ha="center", fontsize=7, color=GREY, fontweight="bold")
ax.text(61.2, 0, "11 frames", va="center", fontsize=7, color=BLUE, fontweight="bold")
save(fig, "fig_sampling_regimes_c.pdf")

fig = plt.figure(figsize=(LW_IN, 0.30))
fig.legend(handles=[Line2D([], [], marker="v", linestyle="", color=BLUE, markersize=5, label="Sampled frame"),
                    Line2D([], [], marker="v", linestyle="", markerfacecolor="white", markeredgecolor=GREY,
                           markersize=5, label="Reference frame inserted in a silent interval"),
                    Line2D([], [], color=RED, linestyle=(0, (3, 2)), linewidth=0.8, label="Window boundary")],
           loc="center", ncol=3, frameon=False, columnspacing=1.8, handletextpad=0.5)
save(fig, "fig_sampling_regimes_legend.pdf")
print("figures written to", OUT)
