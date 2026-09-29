"""Per-qubit distance allocation under an observable error budget (D18 to D24)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .qec_model import LogicalErrorModel
from .sensitivity import SensitivityResult
from .synthesis import SynthesisAllocation, sensitivity_aware_precision, uniform_precision


@dataclass
class AllocationProblem:
    s_x: np.ndarray
    s_z: np.ndarray
    supports: list[tuple[int, ...]]
    n_t: np.ndarray
    model: LogicalErrorModel
    p_phys: float
    budget: float
    distances: tuple[int, ...] = tuple(range(3, 32, 2))
    ancilla_overhead: float = 0.5
    factory_cost_per_t: float = 1.9e5

    def __post_init__(self):
        self.K, self.n = self.s_x.shape
        self._px = {d: float(self.model.rates(d, self.p_phys)[0]) for d in self.distances}
        self._pz = {d: float(self.model.rates(d, self.p_phys)[1]) for d in self.distances}
        self._supp_mask = np.zeros((self.K, self.n), dtype=bool)
        for k, S in enumerate(self.supports):
            self._supp_mask[k, list(S)] = True


    def _tables(self, d: np.ndarray):
        d = np.asarray(d)
        dmin_k = np.array([d[list(S)].min() for S in self.supports])
        dmax_k = np.array([d[list(S)].max() for S in self.supports])
        d_eff = np.where(self._supp_mask, dmin_k[:, None], d[None, :])   # merge rule D19
        rounds = self.n_t * dmax_k                                      # duration D18
        px = np.vectorize(self._px.get)(d_eff)
        pz = np.vectorize(self._pz.get)(d_eff)
        return d_eff, rounds, px, pz

    def error(self, d) -> float:
        _, rounds, px, pz = self._tables(d)
        return float(np.sum(rounds[:, None] * (self.s_x * px + self.s_z * pz)))

    def error_breakdown(self, d) -> dict:
        _, rounds, px, pz = self._tables(d)
        per = rounds[:, None] * (self.s_x * px + self.s_z * pz)
        idle = per[~self._supp_mask].sum()
        active = per[self._supp_mask].sum()
        return dict(total=float(per.sum()), idle=float(idle), active=float(active),
                    per_qubit=per.sum(axis=0), x_part=float(np.sum(rounds[:, None] * self.s_x * px)),
                    z_part=float(np.sum(rounds[:, None] * self.s_z * pz)))

    def cost(self, d) -> dict:
        d = np.asarray(d)
        _, rounds, _, _ = self._tables(d)
        v_data = float(rounds.sum() * np.sum(d.astype(float) ** 2))
        n_t = int(self.n_t.sum())
        v_total = (1 + self.ancilla_overhead) * v_data + self.factory_cost_per_t * n_t
        return dict(v_data=v_data, v_total=v_total, rounds=int(rounds.sum()), n_t=n_t,
                    physical_data_qubits=float(np.sum(d.astype(float) ** 2)))

    def feasible(self, d) -> bool:
        return self.error(d) <= self.budget


    def uniform(self) -> np.ndarray | None:
        for d in self.distances:
            v = np.full(self.n, d)
            if self.feasible(v):
                return v
        return None


    def optimise(self, exhaustive_limit: int = 300_000) -> np.ndarray | None:
        """Minimise v_total subject to error <= budget. Exhaustive for small spaces, greedy otherwise."""
        start = self.uniform()
        if start is None:
            return None
        if len(self.distances) ** self.n <= exhaustive_limit:
            return self._exhaustive()
        return self._greedy(start)

    def _exhaustive(self) -> np.ndarray:
        best, best_cost = None, np.inf
        for combo in itertools.product(self.distances, repeat=self.n):
            d = np.array(combo)
            if not self.feasible(d):
                continue
            c = self.cost(d)["v_total"]
            if c < best_cost:
                best, best_cost = d, c
        return best

    def _greedy(self, start: np.ndarray) -> np.ndarray:
        d = start.copy()
        dmin, dmax = min(self.distances), max(self.distances)
        cur = self.cost(d)["v_total"]
        improved = True
        while improved:
            improved = False
            cands = []
            for i in range(self.n):
                if d[i] - 2 >= dmin:
                    v = d.copy(); v[i] -= 2; cands.append(v)
                for j in range(self.n):
                    if j != i and d[i] - 2 >= dmin and d[j] + 2 <= dmax:
                        v = d.copy(); v[i] -= 2; v[j] += 2; cands.append(v)
            best = None
            for v in cands:
                if self.feasible(v):
                    c = self.cost(v)["v_total"]
                    if c < cur - 1e-9:
                        cur, best = c, v
            if best is not None:
                d, improved = best, True
        return d


    def required_uniform_logical_rate(self) -> float:
        """Logical error rate per patch and per round that uniform protection would need."""
        rounds = self.n_t * 1.0
        w = np.sum(rounds[:, None] * (self.s_x + self.s_z))
        return float(self.budget / w)


def type_agnostic_problem(prob: AllocationProblem) -> AllocationProblem:
    """Control C2, both channels set to the maximum over Pauli types."""
    s_max = np.maximum(prob.s_x, prob.s_z)
    return AllocationProblem(s_max, s_max, prob.supports, prob.n_t, prob.model, prob.p_phys, prob.budget,
                             prob.distances, prob.ancilla_overhead, prob.factory_cost_per_t)


@dataclass
class CompilationReport:
    budget_total: float
    memory_fraction: float
    synthesis: SynthesisAllocation
    synthesis_uniform: SynthesisAllocation
    d_uniform: np.ndarray
    d_type_agnostic: np.ndarray
    d_aqrc: np.ndarray
    cost_uniform: dict
    cost_type_agnostic: dict
    cost_aqrc: dict
    err_uniform: dict
    err_aqrc: dict
    p_phys: float
    extras: dict = field(default_factory=dict)

    def summary(self) -> dict:
        cu, ct, cq = self.cost_uniform, self.cost_type_agnostic, self.cost_aqrc
        return dict(
            memory_fraction=self.memory_fraction,
            p_phys=self.p_phys,
            n_t_uniform_synthesis=self.synthesis_uniform.total_t,
            n_t_aqrc_synthesis=self.synthesis.total_t,
            d_uniform=int(self.d_uniform[0]),
            d_type_agnostic=self.d_type_agnostic.tolist(),
            d_aqrc=self.d_aqrc.tolist(),
            v_data_uniform=cu["v_data"], v_data_type_agnostic=ct["v_data"], v_data_aqrc=cq["v_data"],
            v_total_uniform=cu["v_total"], v_total_type_agnostic=ct["v_total"], v_total_aqrc=cq["v_total"],
            physical_data_qubits_uniform=cu["physical_data_qubits"],
            physical_data_qubits_aqrc=cq["physical_data_qubits"],
            saving_v_data_vs_uniform=1 - cq["v_data"] / cu["v_data"],
            saving_v_total_vs_uniform=1 - cq["v_total"] / cu["v_total"],
            saving_v_data_vs_type_agnostic=1 - cq["v_data"] / ct["v_data"],
            err_uniform=self.err_uniform["total"], err_aqrc=self.err_aqrc["total"],
            idle_share_aqrc=self.err_aqrc["idle"] / max(self.err_aqrc["total"], 1e-300),
        )


def magic_state_error(S: np.ndarray, supports, n_t: np.ndarray, p_T: float) -> float:
    """Magic-state term of the budget, delta_T = sum_k n_T(k) p_T max_{i in S_k, Q} s^Q_{k,i} (D21)."""
    tot = 0.0
    for k, Sk in enumerate(supports):
        tot += n_t[k] * p_T * float(S[k, list(Sk), :].max())
    return float(tot)


def compile_allocation(sens: SensitivityResult, supports, g, h, model: LogicalErrorModel, p_phys: float,
                       budget_total: float, memory_fractions=(0.3, 0.5, 0.7, 0.85, 0.95),
                       distances=tuple(range(3, 32, 2)), t_params=(3.0, 0.0), eps_max=1e-2,
                       ancilla_overhead=0.5, factory_cost_per_t=1.9e5, p_T: float = 4.5e-8,
                       during_mode: str = "max") -> CompilationReport:
    """Synthesis allocation, then distance allocation, scanned over the budget split f."""
    S = sens.during_step(during_mode)
    s_x, s_z = S[:, :, 0], S[:, :, 2]
    a, b = t_params
    best = None
    for f in memory_fractions:
        avail = budget_total
        for _ in range(2):
            syn = sensitivity_aware_precision(g, h, (1 - f) * avail, eps_max, a, b)
            syn_u = uniform_precision(g, h, (1 - f) * avail, eps_max, a, b)
            if syn.total_t > syn_u.total_t:
                syn = syn_u
            delta_T = magic_state_error(S, supports, syn.n_t, p_T)
            avail = budget_total - delta_T
        prob = AllocationProblem(s_x, s_z, supports, syn.n_t, model, p_phys, f * avail,
                                 distances, ancilla_overhead, factory_cost_per_t)
        d_q = prob.optimise()
        if d_q is None:
            continue
        prob_u = AllocationProblem(s_x, s_z, supports, syn_u.n_t, model, p_phys, f * avail,
                                   distances, ancilla_overhead, factory_cost_per_t)
        d_u = prob_u.uniform()
        prob_ta = type_agnostic_problem(prob)
        d_ta = prob_ta.optimise()
        if d_u is None or d_ta is None:
            continue
        rep = CompilationReport(budget_total, f, syn, syn_u, d_u, d_ta, d_q,
                                prob_u.cost(d_u), prob.cost(d_ta), prob.cost(d_q),
                                prob_u.error_breakdown(d_u), prob.error_breakdown(d_q), p_phys,
                                extras=dict(required_uniform_logical_rate=prob.required_uniform_logical_rate(),
                                            delta_T=delta_T, budget_available=avail, p_T=p_T, problem=prob))
        if best is None or rep.cost_aqrc["v_total"] < best.cost_aqrc["v_total"]:
            best = rep
    if best is None:
        raise RuntimeError("no feasible allocation for any budget split; enlarge the distance set")
    return best


def control_shuffled(rep: CompilationReport, n_perm: int = 20, seed: int = 0) -> dict:
    """Control C3, allocate on a qubit-permuted tensor and evaluate on the true one."""
    prob: AllocationProblem = rep.extras["problem"]
    rng = np.random.default_rng(seed)
    feasible, costs = [], []
    for _ in range(n_perm):
        perm = rng.permutation(prob.n)
        shuffled = AllocationProblem(prob.s_x[:, perm], prob.s_z[:, perm], prob.supports, prob.n_t, prob.model,
                                     prob.p_phys, prob.budget, prob.distances, prob.ancilla_overhead, prob.factory_cost_per_t)
        d = shuffled.optimise()
        if d is None:
            feasible.append(False); costs.append(np.nan); continue
        feasible.append(prob.feasible(d))
        costs.append(prob.cost(d)["v_total"])
    return dict(n_perm=n_perm, fraction_feasible_on_true_tensor=float(np.mean(feasible)),
                v_total_when_feasible=[c for c, f in zip(costs, feasible) if f])


def control_mean_field(rep: CompilationReport, orbital_energies: np.ndarray) -> dict:
    """Control C4, allocate from the mean-field proxy s_X = |eps_i|, s_Z = 0."""
    prob: AllocationProblem = rep.extras["problem"]
    sx = np.tile(np.abs(orbital_energies)[None, :], (prob.K, 1))
    proxy = AllocationProblem(sx, np.zeros_like(sx), prob.supports, prob.n_t, prob.model, prob.p_phys, prob.budget,
                              prob.distances, prob.ancilla_overhead, prob.factory_cost_per_t)
    d = proxy.optimise()
    if d is None:
        return dict(feasible_on_true_tensor=False)
    return dict(d=d.tolist(), feasible_on_true_tensor=bool(prob.feasible(d)), err_on_true_tensor=prob.error(d),
                v_total=prob.cost(d)["v_total"], v_total_aqrc=rep.cost_aqrc["v_total"])


def protection_spec(rep: CompilationReport, paulis, observable: str, ansatz_error: float,
                    fixed: dict | None = None) -> dict:
    """Protection specification, DEFINITIONS.md section 12."""
    syn = rep.synthesis
    return dict(
        logical_qubits=[dict(id=int(i), distance=int(d)) for i, d in enumerate(rep.d_aqrc)],
        rotations=[dict(index=int(k), pauli=str(P), epsilon=float(e), t_count=int(t))
                   for k, (P, e, t) in enumerate(zip(paulis, syn.eps, syn.n_t))],
        budget=dict(observable=observable, unit="Hartree", total=float(rep.budget_total + abs(ansatz_error)),
                    ansatz=float(abs(ansatz_error)), memory=float(rep.memory_fraction * rep.extras["budget_available"]),
                    synthesis=float((1 - rep.memory_fraction) * rep.extras["budget_available"]),
                    magic_state=float(rep.extras["delta_T"])),
        assumptions=dict(p_phys=rep.p_phys, p_T=rep.extras["p_T"], during_step="max", merge_rule="min over support",
                         duration_rule="max over support", logical_Y="omitted", **(fixed or {})),
    )
