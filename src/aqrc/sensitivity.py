"""Sensitivity of an observable to Pauli errors, resolved by time, qubit and type (D6 to D13)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit import PauliRotationCircuit
from .pauli import Pauli, PauliSum, single_qubit_pauli

PAULI_TYPES = ("X", "Y", "Z")


@dataclass
class SensitivityResult:
    s: np.ndarray
    observable: str
    reference_value: float
    ideal_energy: float
    exact_energy: float | None = None

    @property
    def K(self) -> int:
        return self.s.shape[0] - 1

    @property
    def n(self) -> int:
        return self.s.shape[1]

    def during_step(self, mode: str = "max") -> np.ndarray:
        """Sensitivity during rotation k, max or mean of the two endpoint values (D9)."""
        if mode == "max":
            return np.maximum(self.s[:-1], self.s[1:])
        if mode == "mean":
            return 0.5 * (self.s[:-1] + self.s[1:])
        raise ValueError(mode)


    def by_type(self) -> np.ndarray:
        return self.s.sum(axis=(0, 1))

    def by_qubit(self) -> np.ndarray:
        return self.s.sum(axis=(0, 2))

    def by_time(self) -> np.ndarray:
        return self.s.sum(axis=(1, 2))

    def by_qubit_and_type(self) -> np.ndarray:
        return self.s.sum(axis=0)

    def eta(self, axis: str) -> float:
        """Concentration along one axis (D13)."""
        v = {
            "type": self.by_type(),
            "qubit": self.by_qubit(),
            "time": self.by_time(),
            "qubit_type": self.by_qubit_and_type().ravel(),
            "time_qubit_max": self.s.max(axis=2).ravel(),
            "all": self.s.ravel(),
        }[axis]
        return participation_eta(v)

    def cumulative_curve(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.sort(values.ravel())[::-1]
        frac = np.arange(1, v.size + 1) / v.size
        return frac, np.cumsum(v) / v.sum()


def participation_eta(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float).ravel()
    tot = v.sum()
    if tot <= 0:
        return 0.0
    k_eff = tot ** 2 / np.sum(v ** 2)
    return float(1.0 - k_eff / v.size)


def pauli_error_sensitivity(
    circuit: PauliRotationCircuit,
    H: PauliSum,
    observable: str = "energy",
    particle_number: int | None = None,
) -> SensitivityResult:
    """s[t, i, Q] by state-vector simulation (D7)."""
    states = circuit.states()
    psi_final = states[-1]
    E_ideal = H.expectation(psi_final)
    exact_E = None
    if observable == "energy":
        ref = E_ideal

        def value(psi):
            return H.expectation(psi)
    elif observable == "overlap":
        exact_E, gs = H.ground_state(particle_number)
        ref = float(abs(np.vdot(gs, psi_final)) ** 2)

        def value(psi):
            return float(abs(np.vdot(gs, psi)) ** 2)
    else:
        raise ValueError(observable)

    K, n = circuit.depth, circuit.n
    s = np.zeros((K + 1, n, 3))
    for t in range(K + 1):
        psi_t = states[t]
        for i in range(n):
            for a, kind in enumerate(PAULI_TYPES):
                phi = single_qubit_pauli(n, i, kind).apply(psi_t)
                phi = circuit.run(phi, start=t)
                s[t, i, a] = abs(value(phi) - ref)
    return SensitivityResult(s=s, observable=observable, reference_value=ref,
                             ideal_energy=E_ideal, exact_energy=exact_E)


def coherent_angle_sensitivity(circuit: PauliRotationCircuit, H: PauliSum) -> tuple[np.ndarray, np.ndarray]:
    """|dE/dtheta_k| and |d2E/dtheta_k2| for every rotation."""
    g = np.abs(circuit.gradient(H))
    h = circuit.hessian_diagonal(H)
    return g, h


def energy_std(circuit: PauliRotationCircuit, H: PauliSum) -> float:
    """sigma_H of the ideal output."""
    psi = circuit.run()
    Hpsi = H.apply(psi)
    e = float(np.real(np.vdot(psi, Hpsi)))
    e2 = float(np.real(np.vdot(Hpsi, Hpsi)))
    return float(np.sqrt(max(e2 - e * e, 0.0)))


def coherent_error_response(circuit: PauliRotationCircuit, H: PauliSum, t: int, qubit: int,
                            direction: np.ndarray, eps_values) -> tuple[float, np.ndarray]:
    """Energy shift for a coherent single-qubit error exp(-i eps n.sigma) after t rotations, with the exact first-order coefficient."""
    n = circuit.n
    nvec = np.asarray(direction, float); nvec = nvec / np.linalg.norm(nvec)
    Ps = [single_qubit_pauli(n, qubit, k) for k in PAULI_TYPES]
    states = circuit.states()
    psi_t = states[t]
    E0 = H.expectation(states[-1])

    def G_apply(phi):
        return sum(c * P.apply(phi) for c, P in zip(nvec, Ps))


    psi_final = circuit.run(psi_t, start=t)
    lam = H.apply(psi_final)
    for k in range(circuit.depth - 1, t - 1, -1):
        lam = circuit.rotate(lam, k, angle=-circuit.angles[k])
    first = 2.0 * np.imag(np.vdot(lam, G_apply(psi_t)))
    shifts = []
    for eps in eps_values:
        phi = np.cos(eps) * psi_t - 1j * np.sin(eps) * G_apply(psi_t)
        shifts.append(H.expectation(circuit.run(phi, start=t)) - E0)
    return float(first), np.array(shifts)


def one_rdm(psi: np.ndarray, n: int) -> np.ndarray:
    """1-RDM gamma_pq = <psi|a_p^dag a_q|psi> (D10)."""
    from .chemistry import _product, _to_terms
    g = np.zeros((n, n), dtype=complex)
    for p_ in range(n):
        for q in range(n):
            terms = _to_terms(_product([(p_, True), (q, False)], n), n)
            g[p_, q] = sum(c * Pauli.from_string(s_).expectation(psi) for s_, c in terms)
    return g


def anticommuting_part(H: PauliSum, Q: Pauli) -> PauliSum:
    """Terms of H that anticommute with Q (D12)."""
    return PauliSum([(s_, c) for s_, c in H.terms if not Pauli.from_string(s_).commutes(Q)], n=H.n)
