"""LiH and H4 inputs, data/lih.json and data/h4.json. Needs pyscf."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from aqrc.chemistry import adapt_vqe, build_active_space_hamiltonian, casci_reference, occupation_changing_weight

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SYSTEMS = {
    "lih": dict(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", active=[1, 2, 5], core=[0],
                label="LiH, R = 1.6 A, STO-3G, frozen Li 1s, active (2e, 3o) = the three valence A1 orbitals"),
    "h4": dict(atom="; ".join(f"H 0 0 {1.5 * i:.3f}" for i in range(4)), basis="sto-3g",
               active=[0, 1, 2, 3], core=[],
               label="H4 chain, R = 1.5 A, STO-3G, (4e, 4o)"),
}


def main(names=("lih", "h4")):
    DATA.mkdir(exist_ok=True)
    for name in names:
        spec = SYSTEMS[name]
        print(f"== {name}: {spec['label']}")
        ham = build_active_space_hamiltonian(spec["atom"], spec["basis"], spec["active"], spec["core"], spec["label"])
        e_casci = casci_reference(spec["atom"], spec["basis"], spec["active"], spec["core"])
        assert abs(e_casci - ham.e_exact) < 1e-8, (e_casci, ham.e_exact)
        res = adapt_vqe(ham, grad_tol=1e-4, energy_tol=1e-7, max_ops=40, verbose=False)
        circ = res.circuit
        e_ansatz = circ.energy(ham.H)
        print(f"   qubits={ham.n_qubits} electrons={ham.n_electrons} terms={len(ham.H)} "
              f"generators={len(res.chosen)} rotations={circ.depth}")
        print(f"   E_HF={ham.e_hf:.8f}  E_exact={ham.e_exact:.8f}  E_ansatz={e_ansatz:.8f}  "
              f"ansatz error={e_ansatz - ham.e_exact:.2e} Ha")
        circ.to_json(DATA / f"{name}.json", hamiltonian=ham.H,
                     n_electrons=ham.n_electrons, e_hf=ham.e_hf, e_exact=ham.e_exact, e_ansatz=e_ansatz,
                     ansatz_error=e_ansatz - ham.e_exact, active_orbitals=ham.active_orbitals,
                     core_orbitals=ham.core_orbitals, basis=spec["basis"], geometry=spec["atom"],
                     generator_groups=res.groups, orbital_energies=ham.orbital_energies.tolist(),
                     occupation_changing_weight=occupation_changing_weight(ham).tolist())
        print(f"   wrote {DATA / f'{name}.json'}")


if __name__ == "__main__":
    main(sys.argv[1:] or ("lih", "h4"))
