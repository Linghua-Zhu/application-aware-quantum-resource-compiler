"""Two checks behind Figure 2 on H4: intra-generator transients against the string angle, and the
alpha/beta asymmetry of X sensitivities against the spin-orbital ordering."""
import json
import numpy as np
from aqrc.circuit import PauliRotationCircuit
from aqrc.sensitivity import pauli_error_sensitivity
from aqrc.chemistry import build_active_space_hamiltonian, adapt_vqe
import aqrc.chemistry as chem

out = {}
circ, H = PauliRotationCircuit.from_json("data/h4.json")
meta = json.load(open("data/h4.json"))
S = pauli_error_sensitivity(circ, H, "energy").s
groups = meta["generator_groups"]
K = circ.depth
# slice t lies inside generator g if the first rotation of g has index <= t-1 < last; boundaries are t = 0 and the
# indices right after each generator ends
ends = {g[-1] + 1 for g in groups} | {0}
inside = np.array([t not in ends for t in range(K + 1)])
tot = S.sum(axis=(1, 2))
# excess of an interior slice over the mean of the two enclosing boundaries, against the angle of the string
rows = []
for g in groups:
    t0, t1 = g[0], g[-1] + 1
    base = 0.5 * (tot[t0] + tot[t1])
    for t in range(t0 + 1, t1):
        ang = float(np.abs(circ.angles[t - 1]))    # string just executed
        rows.append((ang, tot[t] - base, len(g)))
rows = np.array(rows)
sin2 = np.sin(rows[:, 0]) ** 2
c = np.corrcoef(sin2, rows[:, 1])[0, 1]
slope = np.polyfit(sin2, rows[:, 1], 1)
out["transients"] = dict(n_interior_slices=int(len(rows)), corr_sin2_excess=float(c), slope_Ha_per_sin2=float(slope[0]),
                         max_excess=float(rows[:, 1].max()), max_angle=float(rows[:, 0].max()),
                         mean_excess_singles=float(rows[rows[:, 2] == 2, 1].mean()) if (rows[:, 2] == 2).any() else None,
                         mean_excess_doubles=float(rows[rows[:, 2] == 8, 1].mean()))
print("transients:", out["transients"])

# ordering check: rebuild H4 with the beta spin-orbital of each spatial orbital on the even qubit
atom = "; ".join(f"H 0 0 {1.5 * i:.3f}" for i in range(4))
res = {}
for order in ("ab", "ba"):
    ham = build_active_space_hamiltonian(atom, "sto-3g", [0, 1, 2, 3], [], order, spin_order=order)
    r = adapt_vqe(ham, grad_tol=1e-4, energy_tol=1e-7, max_ops=40, verbose=False)
    res[order] = (r.energies[-1], pauli_error_sensitivity(r.circuit, ham.H, "energy").s[-1])
xa, xb = res["ab"][1][:, 0], res["ba"][1][:, 0]
za, zb = res["ab"][1][:, 2], res["ba"][1][:, 2]
out["ordering"] = dict(E_ab=float(res["ab"][0]), E_ba=float(res["ba"][0]),
                       sX_alpha_first=xa.round(4).tolist(), sX_beta_first=xb.round(4).tolist(),
                       sZ_alpha_first=za.round(4).tolist(), sZ_beta_first=zb.round(4).tolist(),
                       max_abs_diff_Z=float(np.abs(za - zb).max()),
                       max_abs_diff_X_after_swapping_pairs=float(np.abs(xa - xb.reshape(-1, 2)[:, ::-1].ravel()).max()))
print("ordering:", out["ordering"])
json.dump(out, open("data/fig2_checks.json", "w"), indent=1)
