"""Physical layer. Modes: rounds, fit, variants. Run as a file, sinter spawns worker processes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import numpy as np

from aqrc.qec_model import LogicalErrorModel, fit_threshold, per_round_table, run_round_sweep, run_sweep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CSV = DATA / "qec_sweep.csv"


def _merge(df_new: pd.DataFrame) -> pd.DataFrame:
    if CSV.exists():
        old = pd.read_csv(CSV)
        old = old[~old.set_index(["model", "basis", "d", "p"]).index.isin(
            df_new.set_index(["model", "basis", "d", "p"]).index)]
        df_new = pd.concat([old, df_new], ignore_index=True)
    df_new = df_new.sort_values(["model", "basis", "d", "p"]).reset_index(drop=True)
    df_new.to_csv(CSV, index=False)
    return df_new


def sweep_rounds(model="uniform"):
    """Both bases, d = 3, 5, 7, p = 0.1% to 0.5%, r = d, 2d, 3d (D16)."""
    frames = []
    for basis in ("z", "x"):
        frames.append(run_round_sweep(ds=[3, 5, 7], ps=[0.001, 0.002, 0.003, 0.005], bases=(basis,), model=model,
                                      round_multipliers=(1, 2, 3), max_shots=600_000, max_errors=150,
                                      out_csv=DATA / f"round_sweep_{basis}.csv"))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(DATA / "round_sweep.csv", index=False)
    print(df.to_string())


def sweep_uniform():
    df = run_sweep(ds=[3, 5, 7, 9], ps=[0.001, 0.002, 0.003, 0.005], model="uniform",
                   max_shots=1_500_000, max_errors=500,
                   skip=lambda d, p: (d == 9 and p < 0.002))
    _merge(df)
    print(df.to_string())


def sweep_variants():
    for model in ("gate", "measurement", "biased"):
        df = run_sweep(ds=[3, 5, 7], ps=[0.002, 0.004], model=model, max_shots=400_000, max_errors=300)
        _merge(df)
        print(df.to_string())


def fit_all(p_max=0.003):
    """Per-round rates by the slope method, fit and hold-out (D17)."""
    tab = per_round_table(pd.read_csv(DATA / "round_sweep.csv"))
    tab.to_csv(DATA / "qec_per_round.csv", index=False)
    df = tab[tab.p <= p_max]
    report = {}
    for model in sorted(df.model.unique()):
        sub = df[df.model == model]
        m = LogicalErrorModel(fit_threshold(sub[sub.logical_type == "X"]), fit_threshold(sub[sub.logical_type == "Z"]), model)
        m.save(DATA / f"qec_fit_{model}.json")
        entry = m.to_dict()

        val = {}
        for lt in ("X", "Z"):
            s = sub[sub.logical_type == lt]
            train = s[s.d <= 5]
            test = s[(s.d >= 7) & (s.errors >= 10)]
            if len(train) >= 3 and len(test):
                f = fit_threshold(train)
                pred = f.p_round(test.d.values, test.p.values)
                val[lt] = dict(train_d="<=5", n_test=int(len(test)),
                               max_abs_log10_error=float(max(abs(np.log10(pred / test.p_round.values)))),
                               points=[dict(d=int(d), p=float(p), measured=float(m_), predicted=float(pr))
                                       for d, p, m_, pr in zip(test.d, test.p, test.p_round, pred)])
        entry["holdout_validation"] = val
        report[model] = entry
        print(f"[{model}] X: A={m.fit_x.A:.3g} p_th={m.fit_x.p_th:.3%} rmse(log10)={m.fit_x.rmse_log10:.3f} | "
              f"Z: A={m.fit_z.A:.3g} p_th={m.fit_z.p_th:.3%} rmse(log10)={m.fit_z.rmse_log10:.3f}")
        for lt, v in val.items():
            print(f"    hold-out {lt}: max |log10 error| over d>=7 = {v['max_abs_log10_error']:.2f}")
    (DATA / "qec_fit_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "rounds"
    {"rounds": sweep_rounds, "uniform": sweep_uniform, "variants": sweep_variants, "fit": fit_all}[what]()
