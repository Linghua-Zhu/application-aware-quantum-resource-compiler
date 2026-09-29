"""Rotation-synthesis precision allocation and the T-count model (D22, D23)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LN2 = np.log(2.0)


def t_count(eps: np.ndarray, a: float = 3.0, b: float = 0.0) -> np.ndarray:
    eps = np.asarray(eps, dtype=float)
    return np.maximum(1, np.ceil(a * np.log2(1.0 / eps) + b)).astype(int)


def _budget_used(eps, g, h):
    return float(np.sum(g * eps + 0.5 * h * eps ** 2))


@dataclass
class SynthesisAllocation:
    eps: np.ndarray
    n_t: np.ndarray
    budget: float
    budget_used: float
    scheme: str
    a: float
    b: float

    @property
    def total_t(self) -> int:
        return int(self.n_t.sum())


def uniform_precision(g: np.ndarray, h: np.ndarray, budget: float, eps_max: float = 1e-2,
                      a: float = 3.0, b: float = 0.0) -> SynthesisAllocation:
    """One eps for every rotation, as large as the budget allows."""
    g, h = np.asarray(g, float), np.asarray(h, float)
    G, Hh = g.sum(), h.sum()
    if Hh > 0:
        eps = (-G + np.sqrt(G ** 2 + 2 * Hh * budget)) / Hh
    else:
        eps = budget / G
    eps = min(eps, eps_max)
    e = np.full(g.shape, eps)
    return SynthesisAllocation(e, t_count(e, a, b), budget, _budget_used(e, g, h), "uniform", a, b)


def sensitivity_aware_precision(g: np.ndarray, h: np.ndarray, budget: float, eps_max: float = 1e-2,
                                a: float = 3.0, b: float = 0.0, eps_floor: float = 1e-14) -> SynthesisAllocation:
    """min sum_k a log2(1/eps_k) subject to sum_k (g_k eps_k + h_k eps_k^2/2) <= budget, by bisection on the multiplier."""
    g, h = np.asarray(g, float), np.asarray(h, float)
    g = np.maximum(g, 0.0)
    h = np.maximum(h, 0.0)

    def eps_of_lambda(lam):
        c = a / (lam * LN2)
        with np.errstate(divide="ignore", invalid="ignore"):
            e_quad = (-g + np.sqrt(g ** 2 + 4 * h * c)) / (2 * h)
            e_lin = c / g
        e = np.where(h > 0, e_quad, e_lin)
        e = np.where((h <= 0) & (g <= 0), eps_max, e)
        return np.clip(e, eps_floor, eps_max)

    lo, hi = 1e-12, 1e12
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if _budget_used(eps_of_lambda(mid), g, h) > budget:
            lo = mid
        else:
            hi = mid
    e = eps_of_lambda(hi)
    return SynthesisAllocation(e, t_count(e, a, b), budget, _budget_used(e, g, h), "sensitivity-aware", a, b)


def closed_form_saving(g: np.ndarray, h: np.ndarray, a: float = 3.0) -> dict:
    """Closed-form T savings per rotation in the first-order and second-order limits."""
    out = {}
    g = np.asarray(g, float)
    h = np.asarray(h, float)
    for name, v, factor in (("first_order", g, a), ("second_order", h, a / 2)):
        v = v[v > 0]
        if v.size:
            am, gm = v.mean(), np.exp(np.mean(np.log(v)))
            out[name] = dict(am=float(am), gm=float(gm), t_saved_per_rotation=float(factor * np.log2(am / gm)))
    return out
