import json

import numpy as np
import pandas as pd
import pytest

from aqrc.allocator import AllocationProblem, type_agnostic_problem
from aqrc.circuit import PauliRotationCircuit
from aqrc.pauli import Pauli, PauliSum, basis_state
from aqrc.qec_model import LogicalErrorFit, LogicalErrorModel, fit_threshold, make_memory_circuit
from aqrc.sensitivity import participation_eta, pauli_error_sensitivity
from aqrc.synthesis import sensitivity_aware_precision, t_count, uniform_precision


def test_pauli_products_and_phases():
    X, Y, Z = (Pauli.from_string(c) for c in "XYZ")
    assert (X * Y).to_string() == "Z" and np.isclose((X * Y).string_coeff(), 1j)
    assert (Y * Z).to_string() == "X" and np.isclose((Y * Z).string_coeff(), 1j)
    assert (Z * X).to_string() == "Y" and np.isclose((Z * X).string_coeff(), 1j)
    assert not X.commutes(Z) and Pauli.from_string("XX").commutes(Pauli.from_string("ZZ"))


def test_dense_matrix_matches_apply():
    H = PauliSum([("XZ", 0.3), ("YI", -0.7), ("ZZ", 1.1)])
    rng = np.random.default_rng(0)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    direct = sum(c * Pauli.from_string(s).apply(psi) for s, c in H.terms)
    assert np.allclose(H.dense() @ psi, direct)
    assert np.allclose(H.dense(), H.dense().conj().T)


def _toy():
    H = PauliSum([("ZI", -1.0), ("IZ", -0.5), ("XX", 0.3), ("YY", 0.3), ("ZZ", 0.2)])
    circ = PauliRotationCircuit(2, ["XY", "YX", "ZI"], np.array([0.31, -0.12, 0.4]), "10")
    return H, circ


def test_analytic_gradient_matches_finite_difference():
    H, circ = _toy()
    g = circ.gradient(H)
    eps = 1e-6
    for k in range(circ.depth):
        a = circ.angles.copy(); a[k] += eps
        ep = circ.energy(H, a)
        a[k] -= 2 * eps
        em = circ.energy(H, a)
        assert abs(g[k] - (ep - em) / (2 * eps)) < 1e-6


def test_json_roundtrip(tmp_path):
    H, circ = _toy()
    circ.to_json(tmp_path / "c.json", hamiltonian=H)
    c2, H2 = PauliRotationCircuit.from_json(tmp_path / "c.json")
    assert np.allclose(c2.run(), circ.run()) and abs(H2.expectation(c2.run()) - H.expectation(circ.run())) < 1e-12


def test_sensitivity_invariants():
    H, circ = _toy()
    res = pauli_error_sensitivity(circ, H, "energy")
    assert res.s.shape == (circ.depth + 1, 2, 3) and np.all(res.s >= 0)

    assert res.s[0, 0, 2] < 1e-12 and res.s[0, 1, 2] < 1e-12

    assert res.s[0, 0, 0] > 1.0

    ov = pauli_error_sensitivity(circ, H, "overlap")
    assert np.all(ov.s <= 1.0 + 1e-12)


def test_participation_eta():
    assert participation_eta(np.ones(10)) == pytest.approx(0.0)
    assert participation_eta(np.array([1.0, 0, 0, 0])) == pytest.approx(0.75)


def test_biased_circuit_has_pauli_channels():
    c = make_memory_circuit(3, 0.002, "z", "biased")
    names = {inst.name for inst in c.flattened()}
    assert "PAULI_CHANNEL_1" in names and "PAULI_CHANNEL_2" in names and "DEPOLARIZE1" not in names


def test_fit_recovers_synthetic_parameters():
    A, pth = 0.05, 0.007
    rows = []
    for d in (3, 5, 7):
        for p in (0.001, 0.002, 0.003):
            pr = A * (p / pth) ** ((d + 1) / 2)
            rows.append(dict(d=d, p=p, p_round=pr, errors=100, logical_type="X", model="synthetic"))
    fit = fit_threshold(pd.DataFrame(rows))
    assert fit.A == pytest.approx(A, rel=1e-6) and fit.p_th == pytest.approx(pth, rel=1e-6)
    assert fit.p_round(9, 0.001) < fit.p_round(7, 0.001) < fit.p_round(5, 0.001)


def test_synthesis_closed_form_first_order():
    g = np.array([1.0, 0.1, 0.01]); h = np.zeros(3); budget = 1e-3
    alloc = sensitivity_aware_precision(g, h, budget, eps_max=1.0)
    assert np.allclose(alloc.eps, budget / (3 * g), rtol=1e-6)
    uni = uniform_precision(g, h, budget, eps_max=1.0)
    assert alloc.total_t <= uni.total_t

    am, gm = g.mean(), np.exp(np.log(g).mean())
    expected = 3 * 3 * np.log2(am / gm)
    assert abs((uni.total_t - alloc.total_t) - expected) <= 3


def test_synthesis_second_order_scaling():
    g = np.zeros(3); h = np.array([4.0, 1.0, 0.25]); budget = 1e-4
    alloc = sensitivity_aware_precision(g, h, budget, eps_max=1.0)

    ratios = alloc.eps * np.sqrt(h)
    assert np.allclose(ratios, ratios[0], rtol=1e-6)
    assert alloc.budget_used == pytest.approx(budget, rel=1e-6)


def _synthetic_model():
    fx = LogicalErrorFit(0.05, 0.007, "X", "synthetic", 9, 0.0)
    fz = LogicalErrorFit(0.04, 0.0072, "Z", "synthetic", 9, 0.0)
    return LogicalErrorModel(fx, fz, "synthetic")


def test_allocator_baselines_and_savings():
    model = _synthetic_model()
    K, n = 5, 3
    s_x = np.array([[1.0, 0.1, 0.01]] * K)
    s_z = np.array([[0.1, 0.01, 0.001]] * K)
    supports = [(0, 1), (1, 2), (0,), (2,), (0, 1, 2)]
    n_t = np.full(K, 20)
    prob = AllocationProblem(s_x, s_z, supports, n_t, model, 1e-3, 1e-3, distances=tuple(range(3, 24, 2)))
    d_u = prob.uniform()
    d_q = prob.optimise()
    assert prob.feasible(d_u) and prob.feasible(d_q)
    assert prob.cost(d_q)["v_total"] <= prob.cost(d_u)["v_total"]

    assert d_q[2] <= d_q[0]

    d_ex = prob._exhaustive()
    d_gr = prob._greedy(d_u)
    assert prob.cost(d_ex)["v_total"] <= prob.cost(d_gr)["v_total"] + 1e-9

    d_ta = type_agnostic_problem(prob).optimise()
    assert prob.feasible(d_ta) and prob.cost(d_ta)["v_total"] >= prob.cost(d_q)["v_total"] - 1e-9


def test_idle_errors_are_counted():
    model = _synthetic_model()
    s_x = np.array([[1.0, 1.0]]); s_z = np.zeros((1, 2))
    prob = AllocationProblem(s_x, s_z, [(0,)], np.array([10]), model, 1e-3, 1.0)
    b = prob.error_breakdown(np.array([7, 7]))
    assert b["idle"] > 0 and b["active"] > 0 and b["idle"] == pytest.approx(b["active"])


def test_tcount_model():
    assert t_count(np.array([1e-3]))[0] == int(np.ceil(3 * np.log2(1e3)))
