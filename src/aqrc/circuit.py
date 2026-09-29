"""Circuits as sequences of Pauli product rotations exp(-i theta_k P_k) (D4)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .pauli import Pauli, PauliSum, basis_state


@dataclass
class PauliRotationCircuit:
    n: int
    paulis: list[str]
    angles: np.ndarray
    initial_bitstring: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.angles = np.asarray(self.angles, dtype=float)
        if len(self.paulis) != len(self.angles):
            raise ValueError("paulis and angles length mismatch")
        self._P = [Pauli.from_string(s) for s in self.paulis]


    @property
    def depth(self) -> int:
        return len(self.paulis)

    @property
    def supports(self) -> list[tuple[int, ...]]:
        return [P.support for P in self._P]

    def initial_state(self) -> np.ndarray:
        return basis_state(self.initial_bitstring)

    def rotate(self, psi: np.ndarray, k: int, angle: float | None = None) -> np.ndarray:
        th = self.angles[k] if angle is None else angle
        return np.cos(th) * psi - 1j * np.sin(th) * self._P[k].apply(psi)

    def run(self, psi: np.ndarray | None = None, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Apply rotations start..stop-1 to psi, from the initial state by default."""
        if psi is None:
            psi = self.initial_state()
        stop = self.depth if stop is None else stop
        for k in range(start, stop):
            psi = self.rotate(psi, k)
        return psi

    def states(self) -> list[np.ndarray]:
        """states()[t] is the state after t rotations."""
        out = [self.initial_state()]
        for k in range(self.depth):
            out.append(self.rotate(out[-1], k))
        return out


    def energy(self, H: PauliSum, angles: np.ndarray | None = None) -> float:
        if angles is not None:
            saved = self.angles
            self.angles = np.asarray(angles, dtype=float)
            try:
                return H.expectation(self.run())
            finally:
                self.angles = saved
        return H.expectation(self.run())

    def gradient(self, H: PauliSum, angles: np.ndarray | None = None) -> np.ndarray:
        """dE/dtheta_k by the adjoint method, two passes over the circuit."""
        if angles is not None:
            saved = self.angles
            self.angles = np.asarray(angles, dtype=float)
        try:
            fw = self.states()
            lam = H.apply(fw[-1])
            g = np.zeros(self.depth)
            for k in range(self.depth - 1, -1, -1):

                g[k] = 2.0 * np.imag(np.vdot(lam, self._P[k].apply(fw[k + 1])))

                lam = self.rotate(lam, k, angle=-self.angles[k])
            return g
        finally:
            if angles is not None:
                self.angles = saved

    def hessian_diagonal(self, H: PauliSum, step: float = 1e-4) -> np.ndarray:
        """d2E/dtheta_k2 by central differences of the gradient."""
        out = np.zeros(self.depth)
        base = self.angles.copy()
        for k in range(self.depth):
            a = base.copy()
            a[k] += step
            gp = self.gradient(H, a)[k]
            a[k] -= 2 * step
            gm = self.gradient(H, a)[k]
            out[k] = (gp - gm) / (2 * step)
        return out


    def to_json(self, path: str | Path, hamiltonian: PauliSum | None = None, **extra) -> None:
        payload = {
            "n_qubits": self.n,
            "hf_bitstring": self.initial_bitstring,
            "rotations": [[s, float(a)] for s, a in zip(self.paulis, self.angles)],
            "rotation_convention": "exp(-i * angle * P)",
            "qubit_convention": "character j of a Pauli string acts on qubit j; bit j of a basis index is qubit j",
            **self.metadata,
            **extra,
        }
        if hamiltonian is not None:
            payload["hamiltonian"] = [[s, float(np.real(c))] for s, c in hamiltonian.terms]
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> tuple["PauliRotationCircuit", PauliSum | None]:
        d = json.loads(Path(path).read_text())
        rot = d["rotations"]
        meta = {k: v for k, v in d.items() if k not in ("rotations", "hamiltonian", "n_qubits", "hf_bitstring")}
        circ = cls(d["n_qubits"], [r[0] for r in rot], np.array([r[1] for r in rot]), d["hf_bitstring"], meta)
        H = PauliSum([(s, c) for s, c in d["hamiltonian"]], n=d["n_qubits"]) if "hamiltonian" in d else None
        return circ, H
