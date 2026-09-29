"""Pauli strings, Pauli sums and state-vector operations (D2)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

PAULI_CHARS = "IXYZ"


@dataclass(frozen=True)
class Pauli:
    """Pauli string as (x, z, phase), P = phase X^x Z^z, phase = i^(number of Y)."""

    n: int
    x: int
    z: int
    phase: complex = 1.0

    @classmethod
    def from_string(cls, s: str, coeff: complex = 1.0) -> "Pauli":
        x = z = 0
        ny = 0
        for j, ch in enumerate(s):
            if ch == "X":
                x |= 1 << j
            elif ch == "Z":
                z |= 1 << j
            elif ch == "Y":
                x |= 1 << j
                z |= 1 << j
                ny += 1
            elif ch != "I":
                raise ValueError(f"bad Pauli character {ch!r} in {s!r}")
        return cls(len(s), x, z, coeff * (1j ** ny))

    def to_string(self) -> str:
        out = []
        for j in range(self.n):
            xb = (self.x >> j) & 1
            zb = (self.z >> j) & 1
            out.append("IXZY"[xb + 2 * zb])
        return "".join(out)

    @property
    def weight(self) -> int:
        return bin(self.x | self.z).count("1")

    @property
    def support(self) -> tuple[int, ...]:
        m = self.x | self.z
        return tuple(j for j in range(self.n) if (m >> j) & 1)

    def string_coeff(self) -> complex:
        """Coefficient in front of to_string()."""
        ny = bin(self.x & self.z).count("1")
        return self.phase / (1j ** ny)

    def __mul__(self, other: "Pauli") -> "Pauli":
        """Operator product with phase."""
        if self.n != other.n:
            raise ValueError("qubit count mismatch")

        sign = (-1) ** (bin(self.z & other.x).count("1") % 2)
        return Pauli(self.n, self.x ^ other.x, self.z ^ other.z, self.phase * other.phase * sign)

    def commutes(self, other: "Pauli") -> bool:
        a = bin(self.x & other.z).count("1")
        b = bin(self.z & other.x).count("1")
        return (a + b) % 2 == 0

    def apply(self, psi: np.ndarray) -> np.ndarray:
        """P|psi> for a state vector of length 2**n."""
        idx = np.arange(psi.shape[0], dtype=np.int64)
        signs = 1.0 - 2.0 * (popcount(idx & self.z) & 1)   # Z part gives the sign
        out = np.empty_like(psi)
        out[idx ^ self.x] = self.phase * signs * psi        # X part flips the index
        return out

    def expectation(self, psi: np.ndarray) -> complex:
        return np.vdot(psi, self.apply(psi))


def popcount(a: np.ndarray) -> np.ndarray:
    """Vectorised popcount for int64 arrays."""
    a = a.astype(np.uint64)
    a = a - ((a >> np.uint64(1)) & np.uint64(0x5555555555555555))
    a = (a & np.uint64(0x3333333333333333)) + ((a >> np.uint64(2)) & np.uint64(0x3333333333333333))
    a = (a + (a >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((a * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


class PauliSum:
    """Hermitian operator sum_j c_j P_j from [(string, coeff), ...]."""

    def __init__(self, terms: Iterable[tuple[str, complex]], n: int | None = None):
        acc: dict[str, complex] = {}
        for s, c in terms:
            if n is None:
                n = len(s)
            if len(s) != n:
                raise ValueError("inconsistent Pauli lengths")
            acc[s] = acc.get(s, 0.0) + c
        self.n = n if n is not None else 0
        self.terms: list[tuple[str, complex]] = [(s, c) for s, c in acc.items() if abs(c) > 1e-14]
        self._paulis = [Pauli.from_string(s, c) for s, c in self.terms]
        self._dense: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.terms)

    @property
    def dim(self) -> int:
        return 1 << self.n

    def dense(self) -> np.ndarray:
        """Dense matrix, cached, n <= 12."""
        if self._dense is None:
            if self.n > 12:
                raise ValueError("dense matrix too large")
            dim = self.dim
            idx = np.arange(dim, dtype=np.int64)
            M = np.zeros((dim, dim), dtype=complex)
            for P in self._paulis:
                signs = 1.0 - 2.0 * (popcount(idx & P.z) & 1)
                M[idx ^ P.x, idx] += P.phase * signs
            self._dense = M
        return self._dense

    def apply(self, psi: np.ndarray) -> np.ndarray:
        if self.n <= 12:
            return self.dense() @ psi
        out = np.zeros_like(psi)
        for P in self._paulis:
            out += P.apply(psi)
        return out

    def expectation(self, psi: np.ndarray) -> float:
        return float(np.real(np.vdot(psi, self.apply(psi))))

    def ground_state(self, particle_number: int | None = None, n_alpha: int | None = None) -> tuple[float, np.ndarray]:
        """Lowest eigenpair, optionally in a particle-number and S_z sector (D3)."""
        H = self.dense()
        idx = np.arange(self.dim, dtype=np.int64)
        if particle_number is None:
            w, v = np.linalg.eigh(H)
            return float(w[0]), v[:, 0]
        mask = popcount(idx) == particle_number
        if n_alpha is not None:
            alpha_bits = sum(1 << j for j in range(0, self.n, 2))
            mask &= popcount(idx & alpha_bits) == n_alpha
        sector = np.nonzero(mask)[0]
        Hs = H[np.ix_(sector, sector)]
        w, v = np.linalg.eigh(Hs)
        psi = np.zeros(self.dim, dtype=complex)
        psi[sector] = v[:, 0]
        return float(w[0]), psi

    def norm_bound(self) -> float:
        """Triangle-inequality bound on the operator norm."""
        return float(sum(abs(c) for _, c in self.terms))


def basis_state(bitstring: str) -> np.ndarray:
    """Basis state from a bitstring, character j is qubit j."""
    n = len(bitstring)
    idx = 0
    for j, ch in enumerate(bitstring):
        if ch == "1":
            idx |= 1 << j
    psi = np.zeros(1 << n, dtype=complex)
    psi[idx] = 1.0
    return psi


@lru_cache(maxsize=None)
def single_qubit_pauli(n: int, qubit: int, kind: str) -> Pauli:
    s = ["I"] * n
    s[qubit] = kind
    return Pauli.from_string("".join(s))
