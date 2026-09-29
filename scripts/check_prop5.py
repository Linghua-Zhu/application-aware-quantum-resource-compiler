"""Proposition 5 check on LiH, converged and truncated ansatz, number-conserving error directions."""
import json, numpy as np
from aqrc.circuit import PauliRotationCircuit
from aqrc.pauli import Pauli, PauliSum
from aqrc.sensitivity import energy_std
from aqrc.chemistry import build_active_space_hamiltonian, adapt_vqe, fermionic_sd_pool

ham = build_active_space_hamiltonian("Li 0 0 0; H 0 0 1.6", "sto-3g", active=[1,2,5], core=[0])
H = ham.H
pool = fermionic_sd_pool(ham.n_qubits, ham.hf_bitstring)
gens = []
for op in pool:
    M = sum(c * Pauli.from_string(s).apply(np.eye(64, dtype=complex)) for s, c in zip(op.paulis, op.coeffs))
    M = M / np.linalg.norm(M, 2)
    gens.append((op.label, M))
eps = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4])
rng = np.random.default_rng(0)
RECORD = []

def check(circ, label):
    states = circ.states(); psi = states[-1]; E0 = H.expectation(psi); sig = energy_std(circ, H)
    firsts, slopes = [], []
    SHIFTS = []
    for name, G in gens:
        for t in [circ.depth] + list(rng.integers(0, circ.depth + 1, size=2)):
            t = int(t); psi_t = states[t]

            lam = H.apply(circ.run(psi_t, start=t))
            for k in range(circ.depth - 1, t - 1, -1):
                lam = circ.rotate(lam, k, angle=-circ.angles[k])
            first = 2.0 * np.imag(np.vdot(lam, G @ psi_t))
            w, v = np.linalg.eigh(G)
            sh = []
            for e in eps:
                phi = (v * np.exp(-1j * e * w)) @ (v.conj().T @ psi_t)
                sh.append(H.expectation(circ.run(phi, start=t)) - E0)
            sh = np.array(sh)
            SHIFTS.append(sh.tolist())
            slope = np.polyfit(np.log(eps[-3:]), np.log(np.abs(sh[-3:]) + 1e-300), 1)[0]
            firsts.append(abs(first)); slopes.append(slope)
    RECORD.append(dict(label=label.strip(), ansatz_error=float(E0-ham.e_exact), sigma_H=float(sig), eps=eps.tolist(), shifts=SHIFTS, max_first_order=float(max(firsts)), slopes=[float(x) for x in slopes]))
    print(f"{label}: E-E_exact = {E0-ham.e_exact:.2e} Ha, sigma_H = {sig:.2e}, bound 2*sigma_H = {2*sig:.2e}, "
          f"max |first-order| = {max(firsts):.2e}, scaling exponent at small eps (min/median/max) = "
          f"{min(slopes):.2f}/{np.median(slopes):.2f}/{max(slopes):.2f}, directions x insertions = {len(firsts)}")

circ, _ = PauliRotationCircuit.from_json("data/lih.json")
check(circ, "converged (8 generators)  ")
for m in (3, 1):
    res = adapt_vqe(ham, grad_tol=1e-4, energy_tol=1e-7, max_ops=m, verbose=False)
    check(res.circuit, f"truncated ({m} generator{'s' if m>1 else ' '})")

json.dump(dict(cases=RECORD), open("data/prop5_check.json", "w"), indent=1)
