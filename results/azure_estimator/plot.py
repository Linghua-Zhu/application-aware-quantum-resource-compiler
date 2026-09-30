"""Space-time frontier and qubit breakdown for LiH, from qre_comparison.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "qre_comparison.json").read_text())

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                     "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
                     "legend.frameon": False, "savefig.bbox": "tight", "figure.dpi": 150})
style = {"default": dict(color="#8E9AAF", marker="o", mfc="white", label="default"),
         "aqrc": dict(color="#C08B6E", marker="o", label="AQRC"),
         "aqrc_first_order": dict(color="#8FA98A", marker="s", ms=3, label="AQRC, first order")}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
for name in ("default", "aqrc", "aqrc_first_order"):
    f = sorted(d[name]["frontier"], key=lambda e: e["runtime_ms"])
    kw = dict(ms=3.5, lw=0.9)
    kw.update(style[name])
    ax.plot([e["runtime_ms"] for e in f], [e["physical_qubits"] / 1e3 for e in f], **kw)
ax.set(xscale="log", yscale="log", xlabel="Runtime (ms)", ylabel=r"Physical qubits ($10^3$)", xlim=(1.5, 40), ylim=(14, 140))
ax.set_xticks([2, 5, 10, 20]); ax.set_xticklabels(["2", "5", "10", "20"])
ax.set_yticks([20, 50, 100]); ax.set_yticklabels(["20", "50", "100"])
ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
ax.legend(loc="upper right", fontsize=8, handlelength=1.8)
ax.text(-0.16, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)


def nearest(name, t):
    return min(d[name]["frontier"], key=lambda e: abs(np.log(e["runtime_ms"] / t)))


pts = [nearest("default", 10.0), nearest("aqrc", 10.0)]
x = np.arange(2)
alg = np.array([p["qubits_algorithm"] for p in pts]) / 1e3
fac = np.array([p["qubits_factories"] for p in pts]) / 1e3
bx.bar(x, fac, 0.55, color="#C9B27C", label="T factories")
bx.bar(x, alg, 0.55, bottom=fac, color="#6FA3A0", label="algorithm")
bx.set(xticks=x, ylabel=r"Physical qubits ($10^3$)", xlim=(-0.6, 1.6), ylim=(0, 42))
bx.set_xticklabels([f"default\n{pts[0]['runtime_ms']:.1f} ms", f"AQRC\n{pts[1]['runtime_ms']:.1f} ms"])
bx.tick_params(axis="x", which="both", top=False, bottom=False)
bx.minorticks_off(); bx.tick_params(axis="y", which="both", right=True)
bx.legend(loc="upper right", fontsize=8, handlelength=1.2)
bx.text(-0.3, 1.02, "b", transform=bx.transAxes, fontweight="bold", fontsize=10)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"frontier.{ext}")
