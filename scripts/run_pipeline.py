"""Sensitivity tensors, allocation, controls and protection specs for the systems given on the command line."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from aqrc.allocator import AllocationProblem, compile_allocation, control_mean_field, control_shuffled, protection_spec, type_agnostic_problem
from aqrc.circuit import PauliRotationCircuit
from aqrc.qec_model import LogicalErrorModel
from aqrc.sensitivity import coherent_angle_sensitivity, energy_std, one_rdm, pauli_error_sensitivity
from aqrc.synthesis import closed_form_saving

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CHEMICAL_ACCURACY = 1.6e-3
P_PHYS = 1e-3
DISTANCES = tuple(range(3, 34, 2))


def sensitivities(system: str):
    circ, H = PauliRotationCircuit.from_json(DATA / f"{system}.json")
    meta = json.loads((DATA / f"{system}.json").read_text())
    t = time.time()
    s_e = pauli_error_sensitivity(circ, H, "energy")
    s_o = pauli_error_sensitivity(circ, H, "overlap", particle_number=meta["n_electrons"])
    g, h = coherent_angle_sensitivity(circ, H)
    occ = np.real(np.diag(one_rdm(circ.run(), circ.n)))
    np.savez(DATA / f"sensitivity_{system}.npz", s_energy=s_e.s, s_overlap=s_o.s, g=g, h=h, occupations=occ,
             supports=np.array([list(S) + [-1] * (circ.n - len(S)) for S in circ.supports]),
             ideal_energy=s_e.ideal_energy, overlap_reference=s_o.reference_value)
    print(f"[{system}] sensitivities in {time.time()-t:.1f}s; n={circ.n}, K={circ.depth}; "
          f"eta(type)={s_e.eta('type'):.3f} eta(qubit)={s_e.eta('qubit'):.3f} eta(time)={s_e.eta('time'):.3f} "
          f"eta(qubit,type)={s_e.eta('qubit_type'):.3f}; X/Z={s_e.by_type()[0]/s_e.by_type()[2]:.2f}; "
          f"|dE/dth|max={g.max():.1e}, curvature median={np.median(h):.2f}")
    return circ, H, meta, s_e, s_o, g, h


H_of = {}


def allocate(system: str, model_name: str, circ, meta, sens, g, h, tag="energy"):
    model = LogicalErrorModel.load(DATA / f"qec_fit_{model_name}.json")
    budget = CHEMICAL_ACCURACY - abs(meta.get("ansatz_error", 0.0))
    sigma_h = energy_std(circ, H_of[system])
    g = np.maximum(g, 2.0 * sigma_h)
    rep = compile_allocation(sens, circ.supports, g, h, model, P_PHYS, budget, distances=DISTANCES)
    summ = rep.summary()

    S = sens.during_step()
    prob_syn_only = AllocationProblem(S[:, :, 0], S[:, :, 2], circ.supports, rep.synthesis.n_t, model, P_PHYS,
                                      rep.memory_fraction * budget, DISTANCES)
    d_syn_only = prob_syn_only.uniform()
    summ["v_data_uniform_d_aware_synthesis"] = prob_syn_only.cost(d_syn_only)["v_data"]
    summ["v_total_uniform_d_aware_synthesis"] = prob_syn_only.cost(d_syn_only)["v_total"]
    summ["closed_form_synthesis"] = closed_form_saving(g, h)
    summ["eps_aqrc"] = rep.synthesis.eps.tolist()
    summ["n_t_per_rotation_aqrc"] = rep.synthesis.n_t.tolist()
    summ["n_t_per_rotation_uniform"] = rep.synthesis_uniform.n_t.tolist()
    summ["per_qubit_error_aqrc"] = rep.err_aqrc["per_qubit"].tolist()
    summ["per_qubit_error_uniform"] = rep.err_uniform["per_qubit"].tolist()
    summ["error_x_part_aqrc"] = rep.err_aqrc["x_part"]
    summ["error_z_part_aqrc"] = rep.err_aqrc["z_part"]
    summ["required_uniform_logical_rate_per_cycle"] = rep.extras["required_uniform_logical_rate"]
    summ["logical_rates_at_d"] = {int(d): [float(x) for x in model.rates(d, P_PHYS)] for d in DISTANCES}
    summ["budget"] = budget
    summ["sigma_H"] = sigma_h
    summ["delta_T"] = rep.extras["delta_T"]
    summ["control_shuffled"] = control_shuffled(rep, n_perm=5)
    if "orbital_energies" in meta:
        summ["control_mean_field"] = control_mean_field(rep, np.array(meta["orbital_energies"]))
    spec = protection_spec(rep, circ.paulis, tag, meta.get("ansatz_error", 0.0),
                           fixed=dict(alpha=0.5, c_T=1.9e5, a=3, b=0, eps_max=1e-2, distances=list(DISTANCES)))
    (DATA / f"spec_{system}_{model_name}_{tag}.json").write_text(json.dumps(spec, indent=1))
    summ["system"] = system
    summ["noise_model"] = model_name
    summ["observable"] = tag

    curve = []
    for target in np.logspace(np.log10(2e-4), np.log10(1.6e-2), 8):
        r = compile_allocation(sens, circ.supports, g, h, model, P_PHYS, target, memory_fractions=(rep.memory_fraction,),
                               distances=DISTANCES)
        c = r.summary()
        curve.append(dict(target=float(target), saving_v_data=c["saving_v_data_vs_uniform"],
                          saving_v_total=c["saving_v_total_vs_uniform"], d_uniform=c["d_uniform"],
                          d_aqrc=c["d_aqrc"]))
    summ["saving_vs_target"] = curve
    out = DATA / f"results_{system}_{model_name}_{tag}.json"
    out.write_text(json.dumps(summ, indent=1, default=float))
    print(f"[{system}/{model_name}/{tag}] d_uniform={summ['d_uniform']} d_aqrc={summ['d_aqrc']} "
          f"d_type_agnostic={summ['d_type_agnostic']} f_mem={summ['memory_fraction']} | "
          f"V_data saving {summ['saving_v_data_vs_uniform']:.1%} (vs type-agnostic {summ['saving_v_data_vs_type_agnostic']:.1%}), "
          f"V_total saving {summ['saving_v_total_vs_uniform']:.1%}; T: {summ['n_t_uniform_synthesis']} -> {summ['n_t_aqrc_synthesis']}; "
          f"idle share {summ['idle_share_aqrc']:.2f}")
    return summ


def main(systems=("lih", "h4")):
    models = [p.stem.replace("qec_fit_", "") for p in DATA.glob("qec_fit_*.json") if "report" not in p.stem]
    for system in systems:
        circ, H, meta, s_e, s_o, g, h = sensitivities(system)
        H_of[system] = H
        for m in models:
            allocate(system, m, circ, meta, s_e, g, h, "energy")
        if "uniform" in models:
            allocate(system, "uniform", circ, meta, s_o, g, h, "overlap")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("lih", "h4"))
