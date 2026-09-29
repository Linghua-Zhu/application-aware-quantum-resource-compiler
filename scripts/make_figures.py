"""fig1 to fig5 and figS1 from the files in data/, PNG and PDF."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from aqrc.qec_model import LogicalErrorModel, fit_threshold, per_round_from_rounds
from aqrc.sensitivity import participation_eta

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Morandi palette. One colour per quantity, kept across figures.
C = dict(d3="#6C8EAD", d5="#8FA98A", d7="#B5695A", X="#A08BA6", Y="#C9B27C", Z="#6FA3A0",
         lih="#7C8A93", h4="#B08968", conv="#6C8EAD", t3="#C9B27C", t1="#B5695A", ref="#3B3B3B", grey="#B9B4AC")
CMAP = LinearSegmentedColormap.from_list("morandi", ["#F4F1EA", "#C9B27C", "#8FA98A", "#6C8EAD", "#3B4A5C"])
plt.rcParams.update({"font.size": 8, "axes.grid": False, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight"})
SYS = ("lih", "h4")
MOL = {"lih": "LiH", "h4": "H$_4$"}
MK = {"lih": "o", "h4": "s"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)


def label(ax, s):
    ax.text(-0.12, 1.02, s, transform=ax.transAxes, fontweight="bold", fontsize=9)


def fig1():
    raw = pd.read_csv(DATA / "round_sweep.csv")
    tab = pd.read_csv(DATA / "qec_per_round.csv")
    model = LogicalErrorModel.load(DATA / "qec_fit_uniform.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ps = np.logspace(-3.1, -2.0, 60)
    for ax, lt, fit, lab in ((axes[0], "X", model.fit_x, "a"), (axes[1], "Z", model.fit_z, "b")):
        sub = tab[tab.logical_type == lt]
        hold = fit_threshold(sub[(sub.d <= 5) & (sub.p <= 0.003)], min_errors=10)
        for d in (3, 5, 7):
            g = sub[sub.d == d]
            col = C[f"d{d}"]
            # 1-sigma from the slope fit: refit with each P(r) perturbed by its binomial error
            err = []
            for _, row in g.iterrows():
                r = raw[(raw.logical_type == lt) & (raw.d == d) & (raw.p == row.p)].sort_values("rounds")
                sig = np.sqrt(r.p_shot * (1 - r.p_shot) / r.shots)
                eps_hi, _ = per_round_from_rounds(r.rounds, r.p_shot + sig)
                eps_lo, _ = per_round_from_rounds(r.rounds, np.maximum(r.p_shot - sig, 1e-12))
                err.append(0.5 * abs(eps_hi - eps_lo))
            fit_pts = g.p <= 0.003
            ax.errorbar(g.p[fit_pts], g.p_round[fit_pts], yerr=np.array(err)[fit_pts.values], fmt="o", ms=4,
                        color=col, capsize=2, label=f"d = {d}")
            ax.errorbar(g.p[~fit_pts], g.p_round[~fit_pts], yerr=np.array(err)[~fit_pts.values], fmt="o", ms=4,
                        mfc="white", color=col, capsize=2)
            ax.plot(ps, fit.p_round(d, ps), "-", color=col, lw=1)
            if d == 7:
                ax.plot(ps, hold.p_round(d, ps), "--", color=C["ref"], lw=0.8, label="d = 7 from d $\\leq$ 5")
        ax.axvline(fit.p_th, color=C["grey"], ls=":", lw=1)
        ax.set(xscale="log", yscale="log", xlabel="physical error rate $p$", ylabel=f"logical {lt} error per round",
               xlim=(8e-4, 1.1e-2), ylim=(1e-6, 2e-1))
        ax.legend(loc="lower right", fontsize=7)
        label(ax, lab)
    # inset: slope method for d = 5, p = 0.2 %
    ins = axes[0].inset_axes([0.24, 0.66, 0.30, 0.30])
    r = raw[(raw.logical_type == "X") & (raw.d == 5) & (raw.p == 0.002)].sort_values("rounds")
    y = np.log(1 - 2 * r.p_shot)
    ins.plot(r.rounds, y, "o", ms=3, color=C["d5"])
    sl, ic = np.polyfit(r.rounds, y, 1)
    rr = np.array([0, 16]); ins.plot(rr, ic + sl * rr, "-", color=C["d5"], lw=0.8)
    ins.set(xlabel="rounds $r$", ylabel="ln(1 $-$ 2P)"); ins.tick_params(labelsize=6)
    ins.xaxis.label.set_size(6); ins.yaxis.label.set_size(6)
    fig.tight_layout(); save(fig, "fig1_physical_layer")


def fig2():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    for row, s in enumerate(SYS):
        z = np.load(DATA / f"sensitivity_{s}.npz"); S = z["s_energy"]; K1, n, _ = S.shape
        meta = json.loads((DATA / f"{s}.json").read_text())
        ax = axes[row, 0]; M = S.mean(axis=0)
        im = ax.imshow(M.T, aspect="auto", cmap=CMAP, vmin=0)
        ax.set(xticks=range(n), yticks=range(3), yticklabels=list("XYZ"), xlabel="logical qubit")
        for i in range(n):
            for a in range(3):
                ax.text(i, a, f"{M[i, a]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, a] > 0.6 * M.max() else C["ref"])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("time-averaged $s$ [Ha]", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax.text(0.01, 1.04, MOL[s], transform=ax.transAxes, fontsize=8)
        label(ax, "ac"[row])
        ax = axes[row, 1]
        for g in meta["generator_groups"]:
            if len(g) == 2:
                ax.axvspan(g[0], g[-1] + 1, color=C["grey"], alpha=0.35, lw=0)
        for a, lab in enumerate("XYZ"):
            ax.plot(range(K1), S[:, :, a].sum(axis=1), color=C[lab], lw=1.2, label=f"{lab}")
        ax.set(xlabel="rotations executed $t$", ylabel="$\\sum_i s_{t,i,Q}$ [Ha]", xlim=(0, K1 - 1))
        ax.legend(loc="center right" if row == 0 else "upper left", fontsize=7)
        label(ax, "bd"[row])
    fig.tight_layout(); save(fig, "fig2_sensitivity_structure")


def figS1():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, s, lab in zip(axes, SYS, "ab"):
        S = np.load(DATA / f"sensitivity_{s}.npz")["s_energy"]
        for name, v, col in (("(qubit, type)", S.sum(axis=0).ravel(), C["X"]),
                             ("max over type", S.max(axis=2).sum(axis=0), C["Z"]),
                             ("all entries", S.ravel(), C["Y"])):
            v = np.sort(v)[::-1]
            x = np.concatenate([[0], np.arange(1, v.size + 1) / v.size]); y = np.concatenate([[0], np.cumsum(v) / v.sum()])
            ax.plot(x, y, color=col, lw=1.2, label=f"{name}, $\\eta$ = {participation_eta(v):.2f}")
        ax.plot([0, 1], [0, 1], ":", color=C["ref"], lw=0.8)
        ax.set(xlabel="fraction of entries", ylabel="cumulative share", xlim=(0, 1), ylim=(0, 1))
        ax.text(0.01, 1.04, MOL[s], transform=ax.transAxes, fontsize=8); ax.legend(fontsize=7, loc="lower right")
        label(ax, lab)
    fig.tight_layout(); save(fig, "figS1_concentration")


ORB = {"lih": ["2σ", "2σ", "3σ", "3σ", "4σ", "4σ"], "h4": ["1", "1", "2", "2", "3", "3", "4", "4"]}


def fig3_4():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    ax = axes[0]
    for s in SYS:
        meta = json.loads((DATA / f"{s}.json").read_text()); S = np.load(DATA / f"sensitivity_{s}.npz")["s_energy"]
        eps = np.abs(np.array(meta["orbital_energies"])); y = S[-1, :, 0]
        ax.plot(eps, y, MK[s], ms=5, color=C[s], label=MOL[s])
        for i in range(0, len(eps), 2):
            ax.annotate(ORB[s][i], (eps[i], max(y[i], y[i + 1])), textcoords="offset points", xytext=(4, 3), fontsize=6, color=C[s])
    ax.plot([0, 0.8], [0, 0.8], ":", color=C["ref"], lw=0.8, label="$s_X = |\\varepsilon_i|$")
    ax.set(xlabel="$|\\varepsilon_i|$ [Ha]", ylabel="$s_{X,i}$ at $t = K$ [Ha]", xlim=(0, 0.8), ylim=(0, 0.8), aspect="equal")
    ax.legend(fontsize=7, loc="upper left"); label(ax, "a")
    ax = axes[1]
    for s in SYS:
        meta = json.loads((DATA / f"{s}.json").read_text()); z = np.load(DATA / f"sensitivity_{s}.npz")
        occ = z["occupations"]; W = np.array(meta["occupation_changing_weight"]); y = z["s_energy"][-1, :, 2]
        b = 2 * W * np.minimum(np.sqrt(occ), np.sqrt(1 - occ))
        ax.plot(b, y, MK[s], ms=5, color=C[s], label=MOL[s])
        for i in range(0, len(b), 2):
            ax.annotate(ORB[s][i], (b[i], y[i]), textcoords="offset points", xytext=(4, 3), fontsize=6, color=C[s])
    ax.plot([1e-2, 1e1], [1e-2, 1e1], ":", color=C["ref"], lw=0.8, label="bound")
    ax.set(xscale="log", yscale="log", xlabel="$2W_i\\,\\min(\\sqrt{n_i},\\sqrt{1-n_i})$ [Ha]", ylabel="$s_{Z,i}$ at $t = K$ [Ha]",
           xlim=(1e-2, 1e1), ylim=(1e-3, 1e1))
    ax.legend(fontsize=7, loc="upper left"); label(ax, "b")
    fig.tight_layout(); save(fig, "fig3_structure_propositions")


def fig5():
    d = json.loads((DATA / "prop5_check.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    cols = [C["conv"], C["t3"], C["t1"]]; names = ["converged", "3 generators", "1 generator"]
    for entry, col, name in zip(d["cases"], cols, names):
        eps = np.array(entry["eps"]); sh = np.abs(np.array(entry["shifts"]))
        first = np.array([abs(s[-1]) / eps[-1] for s in sh])
        hi, lo = sh[np.argmax(first)], sh[np.argmin(first)]
        ax.loglog(eps, hi, "-", color=col, lw=1.3, label=name)
        ax.loglog(eps, lo, "--", color=col, lw=1.0)
    eps = np.array(d["cases"][0]["eps"])
    ax.loglog(eps, 0.5 * eps ** 2, ":", color=C["ref"], lw=0.8); ax.text(1.2e-3, 0.5 * 1.2e-3 ** 2 * 0.5, "$\\varepsilon^2$", fontsize=7)
    ax.loglog(eps, 0.05 * eps, ":", color=C["ref"], lw=0.8); ax.text(4e-4, 0.05 * 4e-4 * 1.8, "$\\varepsilon$", fontsize=7)
    ax.set(xlabel="coherent error size $\\varepsilon$", ylabel="$|E(\\varepsilon) - E(0)|$ [Ha]")
    ax.legend(fontsize=7, loc="upper left"); label(ax, "a")
    ax = axes[1]
    x = np.arange(3)
    firsts = [c["max_first_order"] for c in d["cases"]]; bounds = [2 * c["sigma_H"] for c in d["cases"]]
    ax.bar(x - 0.18, firsts, 0.36, color=[C["conv"], C["t3"], C["t1"]], label="max $|dE/d\\varepsilon|$")
    ax.bar(x + 0.18, bounds, 0.36, color="white", edgecolor=[C["conv"], C["t3"], C["t1"]], linewidth=1.2, label="$2\\sigma_H$")
    for xi, f, b in zip(x, firsts, bounds):
        ax.text(xi - 0.18, f + 0.002, f"{f:.1e}" if f < 1e-3 else f"{f:.3f}", ha="center", fontsize=6)
        ax.text(xi + 0.18, b + 0.002, "0" if b == 0 else f"{b:.3f}", ha="center", fontsize=6)
    ax.set(xticks=x, xticklabels=names, ylabel="[Ha]"); ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, loc="upper left"); label(ax, "b")
    fig.tight_layout(); save(fig, "fig5_coherent_errors")


if __name__ == "__main__":
    fig1(); fig2(); figS1(); fig3_4(); fig5()
    for old in ("fig3_koopmans", "fig4_fluctuation_bound", "fig5_prop5"):
        for ext in ("png", "pdf"):
            p = FIG / f"{old}.{ext}"
            if p.exists():
                p.unlink()
    print("figures written to", FIG)
