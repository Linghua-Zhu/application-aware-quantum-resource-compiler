"""Active-space Hamiltonians (PySCF, Jordan-Wigner) and ADAPT-VQE circuits (D1 to D5)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .circuit import PauliRotationCircuit
from .pauli import Pauli, PauliSum


def _ladder(j: int, dagger: bool, n: int) -> dict[tuple[int, int], complex]:
    """a_j^dag or a_j under Jordan-Wigner, as {(x, z): coeff} with operator = coeff X^x Z^z."""
    zstring = (1 << j) - 1
    xj, zj = 1 << j, 1 << j


    # a^dag = (X + XZ)/2 Z_string, a = (X - XZ)/2 Z_string
    return {(xj, zstring): 0.5, (xj, zj | zstring): 0.5 if dagger else -0.5}


def _mul(A: dict, B: dict) -> dict:
    out: dict[tuple[int, int], complex] = {}
    for (xa, za), ca in A.items():
        for (xb, zb), cb in B.items():
            sign = -1.0 if bin(za & xb).count("1") % 2 else 1.0
            key = (xa ^ xb, za ^ zb)
            out[key] = out.get(key, 0.0) + ca * cb * sign
    return out


def _add(A: dict, B: dict, scale: complex = 1.0) -> None:
    for k, v in B.items():
        A[k] = A.get(k, 0.0) + scale * v


def _to_terms(D: dict, n: int, tol: float = 1e-12) -> list[tuple[str, complex]]:
    terms = []
    for (x, z), c in D.items():
        if abs(c) < tol:
            continue
        ny = bin(x & z).count("1")
        coeff = c / (1j ** ny)
        terms.append((Pauli(n, x, z).to_string(), coeff))
    return terms


def _product(ops: list[tuple[int, bool]], n: int) -> dict:
    out = {(0, 0): 1.0 + 0j}
    for j, dag in ops:
        out = _mul(out, _ladder(j, dag, n))
    return out


@dataclass
class ActiveSpaceHamiltonian:
    H: PauliSum
    n_qubits: int
    n_electrons: int
    hf_bitstring: str
    e_hf: float
    e_exact: float
    label: str
    active_orbitals: list[int]
    core_orbitals: list[int]
    fermion_terms: list[tuple[tuple[tuple[int, bool], ...], float]] | None = None
    orbital_energies: np.ndarray | None = None


def build_active_space_hamiltonian(atom: str, basis: str, active: list[int], core: list[int],
                                   label: str = "", charge: int = 0, spin: int = 0,
                                   spin_order: str = "ab") -> ActiveSpaceHamiltonian:
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(mol).run()
    C = mf.mo_coeff
    norb = C.shape[1]
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), norb)


    e_core = mol.energy_nuc()
    for c in core:
        e_core += 2 * h1[c, c]
        for c2 in core:
            e_core += 2 * eri[c, c, c2, c2] - eri[c, c2, c2, c]
    na = len(active)
    h1a = np.zeros((na, na))
    for P, p in enumerate(active):
        for Q, q in enumerate(active):
            v = h1[p, q]
            for c in core:
                v += 2 * eri[p, q, c, c] - eri[p, c, c, q]
            h1a[P, Q] = v
    eria = eri[np.ix_(active, active, active, active)]

    n = 2 * na
    so = (lambda P, s: 2 * P + s) if spin_order == "ab" else (lambda P, s: 2 * P + 1 - s)   # D2, or beta first
    D: dict[tuple[int, int], complex] = {(0, 0): complex(e_core)}
    fterms: list[tuple[tuple[tuple[int, bool], ...], float]] = []
    for P in range(na):
        for Q in range(na):
            if abs(h1a[P, Q]) < 1e-12:
                continue
            for s in (0, 1):
                ops = ((so(P, s), True), (so(Q, s), False))
                _add(D, _product(list(ops), n), h1a[P, Q])
                fterms.append((ops, float(h1a[P, Q])))
    for P, Q, R, S in itertools.product(range(na), repeat=4):
        v = eria[P, Q, R, S]
        if abs(v) < 1e-12:
            continue
        for s in (0, 1):
            for t in (0, 1):

                ops = ((so(P, s), True), (so(R, t), True), (so(S, t), False), (so(Q, s), False))
                if len({o[0] for o in ops if o[1]}) < 2 or len({o[0] for o in ops if not o[1]}) < 2:
                    continue
                _add(D, _product(list(ops), n), 0.5 * v)
                fterms.append((ops, 0.5 * float(v)))
    terms = _to_terms(D, n)
    for s, c in terms:
        if abs(c.imag) > 1e-9:
            raise RuntimeError("non-Hermitian term produced")
    H = PauliSum([(s, c.real) for s, c in terms], n=n)

    n_elec = mol.nelectron - 2 * len(core)
    n_occ_active = n_elec // 2
    hf_bits = "".join("1" if j < 2 * n_occ_active else "0" for j in range(n))
    from .pauli import basis_state
    e_hf = H.expectation(basis_state(hf_bits))
    e_exact, _ = H.ground_state(particle_number=n_elec, n_alpha=n_elec // 2)
    eps_so = np.repeat(np.asarray(mf.mo_energy)[active], 2)
    return ActiveSpaceHamiltonian(H, n, n_elec, hf_bits, e_hf, e_exact, label, list(active), list(core), fterms, eps_so)


def casci_reference(atom: str, basis: str, active: list[int], core: list[int]) -> float:
    """CASCI energy from PySCF for the same orbitals, used to check E_exact."""
    from pyscf import gto, mcscf, scf

    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    ncas = len(active)
    nelec = mol.nelectron - 2 * len(core)
    mc = mcscf.CASCI(mf, ncas, nelec)

    mo = mc.sort_mo([i + 1 for i in active], base=1)
    assert mc.ncore == len(core), (mc.ncore, len(core))
    return float(mc.kernel(mo)[0])


@dataclass
class PoolOperator:
    label: str
    paulis: list[str]
    coeffs: np.ndarray


def fermionic_sd_pool(n_qubits: int, hf_bitstring: str) -> list[PoolOperator]:
    """Spin-conserving single and double excitation generators T - T^dag."""
    occ = [j for j, b in enumerate(hf_bitstring) if b == "1"]
    vir = [j for j, b in enumerate(hf_bitstring) if b == "0"]
    pool: list[PoolOperator] = []

    def make(label, ops):
        T = _product(ops, n_qubits)
        Td = _product([(j, not d) for j, d in reversed(ops)], n_qubits)
        G = {}
        _add(G, T)
        _add(G, Td, -1.0)
        terms = _to_terms(G, n_qubits)
        if not terms:
            return

        paulis = [s for s, _ in terms]
        coeffs = np.array([c.imag for _, c in terms])
        assert np.allclose([c.real for _, c in terms], 0.0, atol=1e-9)
        pool.append(PoolOperator(label, paulis, coeffs))

    for i in occ:
        for a in vir:
            if (i % 2) == (a % 2):
                make(f"S({i}->{a})", [(a, True), (i, False)])
    for i, j in itertools.combinations(occ, 2):
        for a, b in itertools.combinations(vir, 2):
            if (i % 2 + j % 2) == (a % 2 + b % 2):
                make(f"D({i},{j}->{a},{b})", [(a, True), (b, True), (j, False), (i, False)])
    return pool


@dataclass
class AdaptResult:
    circuit: PauliRotationCircuit
    energies: list[float]
    chosen: list[str]
    thetas: np.ndarray
    groups: list[list[int]]


def adapt_vqe(ham: ActiveSpaceHamiltonian, grad_tol: float = 1e-3, energy_tol: float = 1e-6,
              max_ops: int = 40, verbose: bool = True) -> AdaptResult:
    H = ham.H
    n = ham.n_qubits
    pool = fermionic_sd_pool(n, ham.hf_bitstring)
    pool_P = [[Pauli.from_string(s) for s in op.paulis] for op in pool]

    chosen: list[PoolOperator] = []
    thetas = np.zeros(0)
    energies = [ham.e_hf]

    def flatten(th):
        paulis, angles, groups = [], [], []
        for m, op in enumerate(chosen):
            idx = []
            for s, c in zip(op.paulis, op.coeffs):
                idx.append(len(paulis))
                paulis.append(s)
                angles.append(-th[m] * c)
            groups.append(idx)
        return paulis, np.array(angles), groups

    for it in range(max_ops):
        paulis, angles, groups = flatten(thetas)
        circ = PauliRotationCircuit(n, paulis, angles, ham.hf_bitstring)
        psi = circ.run()
        Hpsi = H.apply(psi)
        grads = np.zeros(len(pool))
        for m, (op, Ps) in enumerate(zip(pool, pool_P)):

            acc = 0.0 + 0j
            for c, P in zip(op.coeffs, Ps):
                acc += c * np.vdot(Hpsi, P.apply(psi))
            grads[m] = 2.0 * np.real(1j * acc)
        best = int(np.argmax(np.abs(grads)))
        if verbose:
            print(f"  ADAPT iter {it:2d}: E = {energies[-1]:.8f}  (exact {ham.e_exact:.8f}, "
                  f"err {energies[-1]-ham.e_exact:.2e})  max|g| = {abs(grads[best]):.2e}  -> {pool[best].label}")
        if abs(grads[best]) < grad_tol or energies[-1] - ham.e_exact < energy_tol:
            break
        chosen.append(pool[best])
        th0 = np.append(thetas, 0.0)

        def fun(th):
            paulis, angles, _ = flatten(th)
            c = PauliRotationCircuit(n, paulis, angles, ham.hf_bitstring)
            e = c.energy(H)
            g_phi = c.gradient(H)
            g = np.zeros(len(th))
            pos = 0
            for m, op in enumerate(chosen):
                k = len(op.coeffs)
                g[m] = np.dot(g_phi[pos:pos + k], -op.coeffs)
                pos += k
            return e, g

        res = minimize(fun, th0, jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 500})
        thetas = res.x
        energies.append(float(res.fun))

    paulis, angles, groups = flatten(thetas)
    circ = PauliRotationCircuit(n, paulis, angles, ham.hf_bitstring,
                                metadata={"molecule": ham.label, "generators": [op.label for op in chosen]})
    return AdaptResult(circ, energies, [op.label for op in chosen], thetas, groups)


def occupation_changing_weight(ham: ActiveSpaceHamiltonian) -> np.ndarray:
    """W_i, sum of |c| over the fermionic terms in which spin-orbital i appears once (D11)."""
    W = np.zeros(ham.n_qubits)
    for ops, c in ham.fermion_terms:
        idx = [o[0] for o in ops]
        for i in set(idx):
            if idx.count(i) == 1:
                W[i] += abs(c)
    return W
