"""Figure 1 of the paper (two panels, vector PDF).

Usage (from the analysis/ folder; needs matplotlib):
    python3 make_figures.py .

Numbers are copied from `python3 stats.py ../results qwen4 qwen8 qwen14 qwen32 gemma12 llama8` (300 MedQA questions, bootstrap B=2000, seed 0).
If the runs change, re-run stats.py and update DATA below.
  (a) AUROC of disagreement -> vote wrong, Qwen3 4B-32B: round 1, round 2, SC-6, with 95% CIs
  (b) fate after round 2 of the questions that were SPLIT in round 1 (section 4 of stats.py)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (point, lo, hi)
AUROC = {
    "SC-6": {4: (.669, .609, .730), 8: (.688, .627, .750), 14: (.727, .654, .792), 32: (.800, .732, .863)},
    "Debate round 1": {4: (.647, .591, .701), 8: (.656, .599, .715), 14: (.705, .638, .777), 32: (.767, .701, .832)},
    "Debate round 2": {4: (.602, .557, .649), 8: (.561, .521, .605), 14: (.601, .547, .657), 32: (.588, .532, .654)},
}
# round-1 split questions -> state after round 2: (unanimous right, unanimous wrong, still split)
SPLIT = [
    ("Qwen3-4B", 27, 22, 32),
    ("Qwen3-8B", 31, 22, 19),
    ("Qwen3-14B", 25, 23, 27),
    ("Qwen3-32B", 17, 25, 21),
    ("Gemma-3-12B", 3, 6, 76),
    ("Llama-3.1-8B", 12, 28, 77),
]

# validated categorical slots 1-3 (dataviz reference palette, all-pairs safe) + neutral gray
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # color, marker, linestyle, x offset (so CI bars do not overlap)
    "SC-6": (AQUA, "s", "-", 1.06),
    "Debate round 1": (BLUE, "o", "-", 1.0),
    "Debate round 2": (ORANGE, "^", "--", 0.94),
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
})


def panel_a(ax):
    sizes = [4, 8, 14, 32]
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":", zorder=1)
    ax.text(16.5, 0.505, "no information (0.5)", color=INK2, fontsize=6.5, va="bottom")
    for name, d in AUROC.items():
        c, m, ls, off = STYLE[name]
        xs = [s * off for s in sizes]
        ys = [d[s][0] for s in sizes]
        err = [[d[s][0] - d[s][1] for s in sizes], [d[s][2] - d[s][0] for s in sizes]]
        ax.errorbar(xs, ys, yerr=err, color=c, marker=m, ms=4.5, lw=1.6, ls=ls, capsize=2,
                    elinewidth=0.8, mec="white", mew=0.6, label=name, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{s}B" for s in sizes])
    ax.minorticks_off()
    ax.set_xlim(3.4, 38)
    ax.set_ylim(0.45, 0.90)
    ax.set_xlabel("Qwen3 model size")
    ax.set_ylabel("AUROC of disagreement for a wrong vote")
    ax.set_title("(a) Error signal before and after debate", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_b(ax):
    names = [s[0] for s in SPLIT][::-1]
    parts = [("unanimous right", BLUE), ("unanimous wrong", ORANGE), ("still split", GRAY)]
    left = [0] * len(names)
    rows = SPLIT[::-1]
    for k, (lab, col) in enumerate(parts):
        vals = [r[k + 1] for r in rows]
        ax.barh(names, vals, left=left, color=col, height=0.62, edgecolor="white", linewidth=1.0, label=lab)
        for y, (v, l0) in enumerate(zip(vals, left)):
            if v >= 8:  # label only blocks wide enough to hold the number
                ax.text(l0 + v / 2, y, str(v), ha="center", va="center", fontsize=6.8,
                        color="white" if col != GRAY else INK)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("Questions split in round 1 (count)")
    ax.set_title("(b) Where round-1 disagreements go after debate", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55), frameon=False, handlelength=1.2)
    ax.set_xlim(0, 140)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main(out_dir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.55), gridspec_kw={"width_ratios": [1, 1.1]})
    panel_a(a)
    panel_b(b)
    fig.tight_layout(w_pad=2.0)
    out = Path(out_dir)
    fig.savefig(out / "fig_signal.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_signal_preview.png", dpi=200, bbox_inches="tight")
    print("wrote", out / "fig_signal.pdf")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
