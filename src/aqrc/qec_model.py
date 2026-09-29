"""Surface-code memory experiments and the per-round logical error model (D14 to D17)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import stim

NOISE_MODELS = ("uniform", "gate", "measurement", "biased")


def _channel_rates(model: str, p: float) -> dict[str, float]:
    if model in ("uniform", "biased"):
        return dict(after_clifford_depolarization=p, before_round_data_depolarization=p,
                    before_measure_flip_probability=p, after_reset_flip_probability=p)
    if model == "gate":
        return dict(after_clifford_depolarization=p, before_round_data_depolarization=p / 10,
                    before_measure_flip_probability=p / 10, after_reset_flip_probability=p / 10)
    if model == "measurement":
        return dict(after_clifford_depolarization=p / 5, before_round_data_depolarization=p / 5,
                    before_measure_flip_probability=5 * p, after_reset_flip_probability=5 * p)
    raise ValueError(model)


def _bias_channels(circuit: stim.Circuit, eta: float) -> stim.Circuit:
    """Replace DEPOLARIZE1/2 by Z-biased Pauli channels of the same total rate."""
    out = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name == "DEPOLARIZE1":
            p = inst.gate_args_copy()[0]
            pz = p * eta / (1 + eta)
            pxy = p / (1 + eta) / 2
            out.append("PAULI_CHANNEL_1", inst.targets_copy(), [pxy, pxy, pz])
        elif inst.name == "DEPOLARIZE2":
            p = inst.gate_args_copy()[0]

            names = ["IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
            ztype = [nm in ("IZ", "ZI", "ZZ") for nm in names]
            pz_tot = p * eta / (1 + eta)
            pother_tot = p / (1 + eta)
            probs = [pz_tot / 3 if zt else pother_tot / 12 for zt in ztype]
            out.append("PAULI_CHANNEL_2", inst.targets_copy(), probs)
        else:
            out.append(inst)
    return out


def make_memory_circuit(d: int, p: float, basis: str = "z", model: str = "uniform",
                        rounds: int | None = None, bias: float = 10.0) -> stim.Circuit:
    rounds = d if rounds is None else rounds
    c = stim.Circuit.generated(f"surface_code:rotated_memory_{basis}", distance=d, rounds=rounds,
                               **_channel_rates(model, p))
    if model == "biased":
        c = _bias_channels(c, bias)
    return c


def run_sweep(ds, ps, bases=("z", "x"), model="uniform", max_shots=1_000_000, max_errors=400,
              num_workers=1, bias=10.0, out_csv: str | Path | None = None,
              skip=lambda d, p: False, verbose=True) -> pd.DataFrame:
    """Memory experiments at r = d, one row per (basis, d, p)."""
    import sinter

    tasks = []
    for basis in bases:
        for d in ds:
            for p in ps:
                if skip(d, p):
                    continue
                tasks.append(sinter.Task(
                    circuit=make_memory_circuit(d, p, basis, model, bias=bias),
                    json_metadata={"d": d, "p": p, "basis": basis, "model": model, "rounds": d}))
    stats = sinter.collect(num_workers=num_workers, tasks=tasks, decoders=["pymatching"],
                           max_shots=max_shots, max_errors=max_errors, print_progress=verbose)
    rows = []
    for s in stats:
        m = s.json_metadata
        p_shot = s.errors / s.shots
        rounds = m["rounds"]
        p_round = 1.0 - (1.0 - p_shot) ** (1.0 / rounds) if p_shot < 1 else 1.0
        rows.append(dict(model=m["model"], basis=m["basis"], d=m["d"], p=m["p"], rounds=rounds,
                         shots=s.shots, errors=s.errors, p_shot=p_shot, p_round=p_round,
                         logical_type="X" if m["basis"] == "z" else "Z"))
    df = pd.DataFrame(rows).sort_values(["model", "basis", "d", "p"]).reset_index(drop=True)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


@dataclass
class LogicalErrorFit:
    A: float
    p_th: float
    logical_type: str
    model: str
    n_points: int
    rmse_log10: float

    def p_round(self, d, p) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        return self.A * (p / self.p_th) ** ((d + 1) / 2)

    def to_dict(self):
        return dict(A=self.A, p_th=self.p_th, logical_type=self.logical_type, model=self.model,
                    n_points=self.n_points, rmse_log10=self.rmse_log10)


def fit_threshold(df: pd.DataFrame, min_errors: int = 10) -> LogicalErrorFit:
    """Least-squares fit of log p_round = log A + (d+1)/2 (log p - log p_th) (D17)."""
    use = df[(df.errors >= min_errors) & (df.p_round > 0)]
    if len(use) < 3:
        raise ValueError("not enough well-sampled points to fit")
    y = np.log(use.p_round.values)
    x1 = (use.d.values + 1) / 2 * np.log(use.p.values)
    x2 = (use.d.values + 1) / 2

    Xd = np.column_stack([np.ones_like(x2), -x2])
    coef, *_ = np.linalg.lstsq(Xd, y - x1, rcond=None)
    logA, logpth = coef
    fit = LogicalErrorFit(float(np.exp(logA)), float(np.exp(logpth)),
                          use.logical_type.iloc[0], use.model.iloc[0], len(use), 0.0)
    pred = np.log10(fit.p_round(use.d.values, use.p.values))
    fit.rmse_log10 = float(np.sqrt(np.mean((pred - np.log10(use.p_round.values)) ** 2)))
    return fit


class LogicalErrorModel:
    """Per-round logical X and Z error rates for one noise structure."""

    def __init__(self, fit_x: LogicalErrorFit, fit_z: LogicalErrorFit, model: str):
        self.fit_x, self.fit_z, self.model = fit_x, fit_z, model

    @classmethod
    def from_csv(cls, csv: str | Path, model: str = "uniform", min_errors: int = 10) -> "LogicalErrorModel":
        df = pd.read_csv(csv)
        df = df[df.model == model]
        fx = fit_threshold(df[df.logical_type == "X"], min_errors)
        fz = fit_threshold(df[df.logical_type == "Z"], min_errors)
        return cls(fx, fz, model)

    def rates(self, d, p) -> tuple[np.ndarray, np.ndarray]:
        """(p_LX, p_LZ) per round at distance d and physical rate p."""
        return self.fit_x.p_round(d, p), self.fit_z.p_round(d, p)

    def to_dict(self):
        return dict(model=self.model, X=self.fit_x.to_dict(), Z=self.fit_z.to_dict())

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path) -> "LogicalErrorModel":
        d = json.loads(Path(path).read_text())
        return cls(LogicalErrorFit(**d["X"]), LogicalErrorFit(**d["Z"]), d["model"])


def per_round_from_rounds(rounds, p_shot):
    """Per-round rate from several round counts, P(r) = (1 - (1 - 2 eps_b)(1 - 2 eps)^r)/2 (D16)."""
    rounds = np.asarray(rounds, float)
    y = np.log(1.0 - 2.0 * np.asarray(p_shot, float))
    slope, intercept = np.polyfit(rounds, y, 1)
    eps = 0.5 * (1.0 - np.exp(slope))
    eps_b = 0.5 * (1.0 - np.exp(intercept))
    return float(eps), float(eps_b)


def run_round_sweep(ds, ps, bases=("z", "x"), model="uniform", round_multipliers=(1, 2, 3),
                    max_shots=600_000, max_errors=150, num_workers=1, bias=10.0, out_csv=None,
                    skip=lambda d, p: False, verbose=False) -> pd.DataFrame:
    """Memory experiments at r = m d, one row per (basis, d, p, r)."""
    import sinter
    rows = []
    for basis in bases:
        for d in ds:
            for p in ps:
                if skip(d, p):
                    continue
                for m in round_multipliers:
                    r = m * d
                    task = sinter.Task(circuit=make_memory_circuit(d, p, basis, model, rounds=r, bias=bias),
                                       json_metadata={"d": d, "p": p, "basis": basis, "model": model, "rounds": r})
                    st = sinter.collect(num_workers=num_workers, tasks=[task], decoders=["pymatching"],
                                        max_shots=max_shots, max_errors=max_errors, print_progress=verbose)[0]
                    rows.append(dict(model=model, basis=basis, d=d, p=p, rounds=r, shots=st.shots, errors=st.errors,
                                     p_shot=st.errors / st.shots, logical_type="X" if basis == "z" else "Z"))
                    if out_csv is not None:
                        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


def per_round_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, basis, d, p): eps from the slope, eps_b, and the single-experiment value for comparison."""
    out = []
    for (model, basis, d, p), g in df.groupby(["model", "basis", "d", "p"]):
        g = g.sort_values("rounds")
        if (g.errors < 5).any() or len(g) < 2:
            continue
        eps, eps_b = per_round_from_rounds(g.rounds.values, g.p_shot.values)
        single = g[g.rounds == d]
        p_single = 1.0 - (1.0 - single.p_shot.values[0]) ** (1.0 / d) if len(single) else np.nan
        out.append(dict(model=model, basis=basis, d=d, p=p, logical_type="X" if basis == "z" else "Z",
                        p_round=eps, eps_boundary=eps_b, p_round_single=p_single,
                        errors=int(g.errors.min()), rounds_used=list(map(int, g.rounds.values))))
    return pd.DataFrame(out)
