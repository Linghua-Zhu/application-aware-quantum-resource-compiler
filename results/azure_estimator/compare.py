"""LiH resource estimates from the Azure Quantum Resource Estimator with the default and the AQRC error budgets."""
import json
from pathlib import Path

import numpy as np
from qdk.estimator import ErrorBudgetPartition, EstimatorParams, LogicalCounts, QECScheme, QubitParams

from aqrc.circuit import PauliRotationCircuit

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"

circ, _ = PauliRotationCircuit.from_json(DATA / "lih.json")
S = np.load(DATA / "sensitivity_lih.npz")["s_energy"]
alloc = json.loads((DATA / "results_lih_uniform_energy.json").read_text())
prop5 = json.loads((DATA / "prop5_check.json").read_text())

layer = []
for k, supp in enumerate(circ.supports):
    prev = [layer[j] for j in range(k) if set(circ.supports[j]) & set(supp)]
    layer.append(1 + max(prev, default=0))
counts = {"numQubits": circ.n, "rotationCount": circ.depth, "rotationDepth": max(layer), "measurementCount": circ.n}

avail = alloc["budget"] - alloc["delta_T"]
delta_mem = alloc["memory_fraction"] * avail
delta_syn = (1 - alloc["memory_fraction"]) * avail
during = np.maximum(S[:-1], S[1:])
s_bar = float(np.mean(0.5 * (during[:, :, 0] + during[:, :, 2])))
two_sigma = 2 * next(c["sigma_H"] for c in prop5["cases"] if c["label"].startswith("truncated (3"))

budgets = {
    "default": (1e-3 / 3, 1e-3 / 3, 1e-3 / 3),
    "aqrc": (delta_mem / s_bar, 1e-3 / 3, float(np.sum(alloc["eps_aqrc"]))),
    "aqrc_first_order": (delta_mem / s_bar, 1e-3 / 3, delta_syn / two_sigma),
}


def params(budget, frontier=False):
    p = EstimatorParams()
    p.qubit_params.name = QubitParams.GATE_NS_E3
    p.qec_scheme.name = QECScheme.SURFACE_CODE
    e = ErrorBudgetPartition()
    e.logical, e.t_states, e.rotations = budget
    p.error_budget = e
    if frontier:
        p.estimate_type = "frontier"
    return p


def summary(r):
    b = r["physicalCounts"]["breakdown"]
    return {"code_distance": r["logicalQubit"]["codeDistance"], "t_per_rotation": b["numTsPerRotation"],
            "t_states": b["numTstates"], "t_factories": b["numTfactories"],
            "physical_qubits": r["physicalCounts"]["physicalQubits"],
            "qubits_algorithm": b["physicalQubitsForAlgorithm"], "qubits_factories": b["physicalQubitsForTfactories"],
            "runtime_ms": r["physicalCounts"]["runtime"] / 1e6}


out = {"logical_counts": counts, "s_bar_Ha": s_bar, "delta_mem_Ha": delta_mem, "delta_syn_Ha": delta_syn,
       "two_sigma_H_Ha": two_sigma}
for name, budget in budgets.items():
    single = LogicalCounts(counts).estimate(params(budget))
    front = LogicalCounts(counts).estimate(params(budget, frontier=True))
    out[name] = {"budget": dict(zip(("logical", "t_states", "rotations"), budget)), "estimate": summary(single),
                 "frontier": [summary(e) for e in front["frontierEntries"]]}
(HERE / "qre_comparison.json").write_text(json.dumps(out, indent=1))

print(counts)
print(f"s_bar {s_bar:.3f} Ha, delta_mem {delta_mem:.3e} Ha, delta_syn {delta_syn:.3e} Ha, 2 sigma_H {two_sigma:.3e} Ha")
for n in budgets:
    b = out[n]["budget"]
    print(f"{n:17s} budget: logical {b['logical']:.2e}, T states {b['t_states']:.2e}, rotations {b['rotations']:.3g}")
keys = ["code_distance", "t_per_rotation", "t_states", "t_factories", "physical_qubits", "qubits_algorithm",
        "qubits_factories", "runtime_ms"]
print(f"{'':18s}" + "".join(f"{n:>18s}" for n in budgets))
for k in keys:
    print(f"{k:18s}" + "".join(f"{round(out[n]['estimate'][k], 3):>18}" for n in budgets))
