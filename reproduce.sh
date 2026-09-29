#!/usr/bin/env bash
# Regenerates everything in data/ and figures/ from a clean checkout. ~10 min on one core.
set -e
cd "$(dirname "$0")"
python scripts/make_molecule.py            # D1 to D5: inputs (needs pyscf)
python scripts/run_qec_sweep.py rounds     # D14 to D16: memory experiments
python scripts/run_qec_sweep.py fit        # D17: per-round rates, fit, hold-out
python scripts/run_pipeline.py lih h4      # D7 to D24: sensitivities, allocation, controls, protection specs
python scripts/check_prop5.py              # Proposition 5 check
python scripts/make_figures.py             # fig1 to fig5
pytest -q
