# aqrc

Application-aware quantum resource compiler. Given a molecular Hamiltonian, a state-preparation circuit written as Pauli product rotations, and an accuracy target on an observable, aqrc computes the sensitivity of the observable to a Pauli error at every (time slice, logical qubit, Pauli type) and uses it to choose a surface-code distance per logical qubit and a synthesis precision per rotation. The output is a JSON protection specification for downstream compilers and resource estimators. Layout, scheduling and gate synthesis are out of scope.

Version 0.1, research code.

## Install

```
conda create -n aqrc python=3.12 -y
conda activate aqrc
pip install -e ".[dev]"        # add chem for pyscf, needed only to regenerate the inputs
pytest -q
```

Run the scripts as files. sinter spawns worker processes that re-import `__main__`, so the physical layer hangs when started from a notebook or from stdin.

## Reproduce

`bash reproduce.sh` regenerates `data/` and `figures/` in about 10 minutes on one core. The steps are

1. `scripts/make_molecule.py`, active-space Hamiltonians (PySCF, Jordan-Wigner) and ADAPT-VQE circuits for LiH and H4, written to `data/*.json`
2. `scripts/run_qec_sweep.py rounds` and `scripts/run_qec_sweep.py fit`, surface-code memory experiments with Stim and PyMatching, per-round logical error rates and their fit
3. `scripts/run_pipeline.py lih h4`, sensitivity tensors, allocation with controls, protection specifications
4. `scripts/check_prop5.py`, numerical check of the coherent-error bound
5. `scripts/check_fig2.py`, the two checks behind Figure 2 (transients against the string angle, spin-orbital ordering)
6. `scripts/make_figures.py`, `figures/fig1`, `fig2`, `fig3`, `fig5` and `figS1`, PNG and PDF

## Contents

```
src/aqrc/    pauli, circuit, chemistry, sensitivity, qec_model, synthesis, allocator
scripts/     see above
data/        inputs, simulation results, fits, sensitivity tensors, protection specs
docs/        DEFINITIONS.md
figures/     fig1 physical layer, fig2 tensor structure, fig3 structure propositions, fig5 coherent errors, figS1 concentration
tests/       unit tests
aqrc_note.pdf   research note, version 0.4
```

## Notes

Every quantity is defined in `docs/DEFINITIONS.md` together with the function that implements it. The research note `aqrc_note.pdf` gives the derivations and the results.

Test systems are LiH (2e, 3o) and an H4 chain (4e, 4o), both STO-3G. Results are in the note, Section 8, and in `figures/`. On these two systems the allocation returns uniform distances at chemical accuracy and p = 1e-3.

Related tools: Qualtran, Azure Quantum Resource Estimator, pyLIQTR, BenchQ (resource estimation under a global error budget); Stim, PyMatching (physical layer). Instruction set and time model follow Pauli-based computation with lattice surgery (Litinski, Quantum 3, 128, 2019).

## License

Apache-2.0. Citation information in `CITATION.cff`.
