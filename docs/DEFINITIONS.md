# AQRC definitions

Version 1.1, 28 September 2026. Every quantity used in the project is defined here once. The code follows this file. When the two disagree, the code is wrong. Each definition names the function that implements it.

D0 (what AQRC is). AQRC, application-aware quantum resource compiler, is a compilation pass. Its input is a program in the Pauli-rotation instruction set (D4) and an observable (D8). Its output is a protection specification (Section 12), the distance of every patch and the synthesis precision of every rotation, together with the error budget they realise. It does not perform layout, scheduling, or gate synthesis. Those belong to the compilers and estimators that consume the specification. The word compiler is used in the sense in which register allocation is part of a classical compiler.

Notation. Ha is Hartree. A logical qubit and its surface-code patch share the index i. Rotations are indexed by k = 1, ..., K. Time slices are indexed by t = 0, ..., K, where t is the number of rotations already executed.

## 1. Electronic structure

D1 (system). A molecule, a Gaussian basis, and an active space given by a list of spatial orbitals. Orbitals are canonical restricted Hartree--Fock (RHF) orbitals of the full molecule. The default is the full orbital space of the basis. A frozen core is used only when stated and only when the full space is out of reach. `chemistry.build_active_space_hamiltonian`

D2 (spin-orbitals and qubits). Spatial orbital p gives two spin-orbitals, 2p (alpha) and 2p+1 (beta); the option `spin_order="ba"` puts beta on the even qubit and is used only for the ordering check of the note, Section 8.2. Spin-orbital j is mapped to qubit j by the Jordan--Wigner transformation with a_j^dag = (X_j - i Y_j)/2 times Z on qubits 0 to j-1. A Pauli string is a string whose character j acts on qubit j. Bit j of a basis-state index is the occupation of spin-orbital j. `pauli.Pauli`, `chemistry._ladder`

D3 (Hamiltonian and reference energies). H = E_core + sum_pq h'_pq a^dag_p a_q + (1/2) sum_pqrs (pq|rs) a^dag_p a^dag_r a_s a_q, with (pq|rs) chemists' notation and h' the one-body integrals with the frozen core folded in. Written as a Pauli sum with real coefficients. E_HF is the energy of the reference determinant. E_exact is the lowest eigenvalue of H in the sector of N electrons and S_z = 0. Validation: E_HF equals the RHF total energy and E_exact equals the PySCF CASCI energy, both to 1e-8 Ha. `chemistry.build_active_space_hamiltonian`, `pauli.PauliSum.ground_state`, `chemistry.casci_reference`

Note on D3. The S_z = 0 restriction is implemented (`n_alpha`). For LiH and H4 the lowest state of the N sector is already the S_z = 0 state and E_exact agrees with CASCI to 1e-8 Ha with and without the restriction. Closed 25 Sept 2026.

## 2. Circuit

D4 (circuit). U = U_K ... U_1 with U_k = exp(-i theta_k P_k), P_k a Pauli string, theta_k real, applied to the reference determinant. The rotations are the flattened exponentiated generators of an ADAPT-VQE ansatz with the fermionic singles-and-doubles pool, spin-conserving. ADAPT stops when the largest pool gradient is below 1e-4 Ha or the energy error is below 1e-7 Ha. After ADAPT the angles are fixed. Everything downstream is a property of the fixed circuit. `chemistry.adapt_vqe`, `circuit.PauliRotationCircuit`

D5 (ansatz error). delta_ans = E_circuit - E_exact, with E_circuit the energy of the ideal circuit output. Reported for every circuit.

## 3. Errors and sensitivity

D6 (error location and error model). A location is l = (t, i, Q) with t the time slice, i the logical qubit, Q in {X, Y, Z}. The error is the Pauli Q applied to qubit i after t rotations, with probability p_l. Locations fail independently.

D7 (sensitivity). For the ideal output |psi> and the corrupted output |psi'_l>,
s_l(O) = | <psi'_l| O |psi'_l> - <psi| O |psi> |,
and Delta_l(O) is the same quantity with its sign. The expectation of O under a single error location is <O> + p_l Delta_l exactly. Under many independent locations it is <O> + sum_l p_l Delta_l + O(p^2). `sensitivity.pauli_error_sensitivity`

D8 (observables). Energy, O = H. Overlap, O = |psi_0><psi_0| with |psi_0> the exact ground state of D3. Both tensors are computed for every circuit.

D9 (during-step sensitivity). For an error that occurs while rotation k executes, s^Q_{k,i} = max(s_{k-1,i,Q}, s_{k,i,Q}). This is a convention chosen as an upper bound of the two endpoint values. The mean of the two endpoints is computed as well and the difference between the two conventions is reported. `sensitivity.SensitivityResult.during_step`

## 4. Reduced density matrices and structural quantities

D10 (occupations and 1-RDM). gamma_pq = <psi| a^dag_p a_q |psi> on the ideal output, n_i = gamma_ii. `sensitivity.one_rdm`

D11 (weight of occupation-changing terms). W_i is the sum of |c| over the fermionic terms of H in which spin-orbital i appears exactly once. `chemistry.occupation_changing_weight`

D12 (anticommuting part). For a Pauli string Q, H = H_c + H_a with H_a the terms of H that anticommute with Q. At t = K, Delta = -2 <psi| H_a |psi>. `sensitivity.anticommuting_part`. Check: at t = K the identity agrees with the numerical tensor to 6e-15 Ha on LiH.

## 5. Concentration

D13 (concentration). For a non-negative vector v of length N, eta = 1 - N_eff / N with N_eff = (sum v)^2 / sum v^2. Evaluated along each axis of the tensor. eta = 0 means all entries equal. `sensitivity.participation_eta`

## 6. Physical layer

D14 (memory experiment). Rotated surface code of distance d, r rounds, generated by Stim as `surface_code:rotated_memory_z` or `_x`, decoded by PyMatching. Circuit-level noise with the four Stim channels after_clifford_depolarization, before_round_data_depolarization, before_measure_flip_probability, after_reset_flip_probability. The uniform structure sets all four to p. `qec_model.make_memory_circuit`

D15 (logical error types). The `_z` experiment counts logical X errors, the `_x` experiment counts logical Z errors. The two are always run separately.

D16 (per-round logical error rate). The per-shot failure probability of an r-round experiment is modelled as P(r) = (1/2)[1 - (1 - 2 eps_b)(1 - 2 eps)^r], with eps the per-round logical error rate and eps_b a boundary term from initialisation and readout. eps is the slope of ln(1 - 2P) against r, fitted over r in {d, 2d, 3d}. The single-experiment conversion 1 - (1 - P(d))^(1/d) is retained only as a comparison. `qec_model.run_round_sweep`, `qec_model.per_round_from_rounds`

D17 (logical error model). p_L(d, p) = A (p / p_th)^((d+1)/2), fitted by least squares on ln p_L with the two unknowns ln A and ln p_th, separately for X and Z, using only points with at least 10 observed errors in every round count, and only p <= 0.003 (p / p_th below 0.4, the below-threshold regime the formula describes). Including p = 0.005 changes p_th by less than 1%. Validation: fit on d <= 5, predict d = 7, report the largest deviation in log10. `qec_model.fit_threshold`, `qec_model.LogicalErrorModel`

## 7. Time model

D18 (duration). Rotation k takes r_k = n_T(eps_k) d_time(k) code cycles with d_time(k) = max over j in S_k of d_j. S_k is the support of P_k.

D19 (effective distance). d_eff(k, i) = min over j in S_k of d_j if i in S_k, and d_i otherwise.

D20 (memory error functional). Err_mem = sum_k r_k sum_i [ s^X_{k,i} p_LX(d_eff) + s^Z_{k,i} p_LZ(d_eff) ]. Logical Y errors are omitted. `allocator.AllocationProblem.error`

## 8. Error budget

D21 (budget). delta_O = delta_ans + delta_mem + delta_syn + delta_T. delta_ans is D5. delta_mem bounds D20. delta_syn bounds the synthesis error of D23. delta_T bounds the magic-state error, delta_T = sum_k n_T(eps_k) p_T max_{i in S_k, Q} s^Q_{k,i}, with p_T the output error of the factory. The split between delta_mem and delta_syn is optimised. delta_T is computed and reported, not optimised. `allocator.magic_state_error`, `allocator.compile_allocation`

D22 (T count). n_T(eps) = ceil(a log2(1/eps) + b), a = 3, b = 0. `synthesis.t_count`

D23 (synthesis error model). Delta E_k(eps_k) = g_k eps_k + (1/2) h_k eps_k^2 with g_k = max(|dE/dtheta_k|, 2 sigma_H) and h_k = |d^2 E / dtheta_k^2|, sigma_H the energy standard deviation of the ideal output. eps_k <= 1e-2. `sensitivity.coherent_angle_sensitivity`, `sensitivity.energy_std`, `synthesis.sensitivity_aware_precision`

## 9. Cost

D24 (volumes). V_data = sum_k r_k sum_i d_i^2 in qubit-rounds. V_total = (1 + alpha) V_data + c_T N_T with N_T = sum_k n_T(eps_k). `allocator.AllocationProblem.cost`

## 10. Optimised, fixed, measured

Optimised: d_i for every logical qubit, eps_k for every rotation, the split f = delta_mem / (delta_mem + delta_syn).

Fixed: p = 1e-3, D = {3, 5, ..., 33}, alpha = 0.5, c_T = 1.9e5 qubit-rounds, p_T = 4.5e-8, a = 3, b = 0, eps_max = 1e-2, delta_O = 1.6e-3 Ha, the noise structure, the merge rule D19, the convention D9.

Measured: A, p_th for X and Z (D16, D17). The sensitivity tensor (D7). sigma_H, g_k, h_k (D23). n_i, W_i (D10, D11).

## 11. Controls

C1 uniform. All d_i equal, smallest feasible value.
C2 type-agnostic. Per-qubit allocation with both channels set to max_Q s.
C3 shuffled. The qubit labels of the tensor are randomly permuted, the allocation is computed on the permuted tensor, and its feasibility and cost are evaluated on the true tensor. Repeated over permutations.
C4 mean-field proxy. Allocation computed from s_X,i = |epsilon_i| and s_Z,i = 0, evaluated on the true tensor.

## 12. Protection specification

A JSON document with four blocks. `logical_qubits`: list of {id, distance}. `rotations`: list of {index, pauli, epsilon, t_count}. `budget`: {observable, total, ansatz, memory, synthesis, magic_state, unit}. `assumptions`: the fixed quantities of Section 10 and the conventions D9, D19. `allocator.protection_spec`, written to `data/spec_<system>_<model>_<observable>.json`

## 13. Text-to-code map

| definition | function | status |
|---|---|---|
| D1, D3 | chemistry.build_active_space_hamiltonian | done |
| D2 | pauli.Pauli, chemistry._ladder | done |
| D4 | chemistry.adapt_vqe | done |
| D7, D8 | sensitivity.pauli_error_sensitivity | done |
| D9 | SensitivityResult.during_step | done (max, mean) |
| D10, D11, D12 | sensitivity.one_rdm, chemistry.occupation_changing_weight, sensitivity.anticommuting_part | done |
| D13 | sensitivity.participation_eta | done |
| D14, D15 | qec_model.make_memory_circuit | done |
| D16 | qec_model.run_round_sweep, per_round_from_rounds, per_round_table | done |
| D17 | qec_model.fit_threshold | done |
| D18 to D20 | allocator.AllocationProblem | done |
| D21 | allocator.magic_state_error, compile_allocation | done |
| D22, D23 | synthesis, sensitivity | done |
| D24 | AllocationProblem.cost | done |
| C1, C2 | allocator | done |
| C3, C4 | allocator.control_shuffled, control_mean_field | done |
| Section 12 | allocator.protection_spec | done |

## 13a. Planned interface (not yet in code)

D25 (architecture interface). The allocation layer will depend only on an Architecture object with four methods and one attribute. `logical_rates(d, p)` returns the per-round X and Z logical error rates. `duration(k, d, n_T)` returns the number of code cycles rotation k occupies. `effective_protection(k, i, d)` returns the distance protecting qubit i during rotation k. `cost(d, N_T)` returns the volume. `distances` is the allowed set. The surface-code lattice-surgery model (D18, D19, D24) is the first implementation. A neutral-atom model with transversal gates and atom transport is the second. The sensitivity tensor is outside the interface and is shared by all implementations. Status: planned, the allocator still hard-codes the first implementation.

## 14. Step 1 log, 25 September 2026

Renamed from QuARC to AQRC on 28 September 2026 because QuArC is the name of the Quantum Architectures and Computation group at Microsoft Research. Package, scripts, documents and the note were renamed together.

What changed when the code was made to follow the definitions.

1. D16 replaced the single-experiment conversion by the slope over r in {d, 2d, 3d}. At d = 7 the single-experiment conversion underestimates the per-round rate by 10 to 40% (ratios 0.58 to 0.89), because the time boundaries of a d-round experiment protect better than the bulk. With the d = 7, p = 0.1% experiments extended to 67 to 78 errors (29 Sept 2026) the model (D17) is A = 0.0175, p_th = 0.869% for X and A = 0.0176, p_th = 0.831% for Z, against 0.0199, 0.932% and 0.0324, 1.040% from the single-experiment conversion. At p = 1e-3 and d = 13 the per-round rates are 4.7e-9 (X) and 6.4e-9 (Z). Hold-out deviation at d = 7 is 0.11 (X) and 0.13 (Z) in log10.

2. With the corrected physical layer the LiH (2e, 3o) allocation at chemical accuracy no longer affords the d = 13 to 11 step on the LUMO qubits. AQRC returns the uniform distances. The 9.5% data-block saving of note v0.3 does not hold with the corrected physical layer. The gap rule already put the step at the margin (gaps 1.2 and 1.8 for a step of 2 at Lambda = 8.5).

3. D21 added the magic-state term. For LiH, delta_T = 2.0e-5 Ha, 1.2% of the budget.

4. C4 (mean-field proxy) is infeasible on the true tensor for LiH at this budget (error 5.3e-4 Ha against a memory budget of 4.7e-4 Ha). The Z sensitivities, which the proxy sets to zero, are needed. C3 (shuffled) is uninformative while the allocation is uniform; it becomes a test only when structure is exploited.

5. D12 identity checked against the numerical tensor at t = K to 6e-15 Ha. The occupation-fluctuation bound s_Z,i <= 2 W_i min(sqrt(n_i), sqrt(1 - n_i)) holds at every qubit of LiH, with the ratio s_Z,i / sqrt(n_i (1 - n_i)) between 0.14 and 0.51 Ha.

6. With the corrected physical layer the H4 uniform distance moves from 13 to 15 and the type-agnostic allocation moves with it, so the 31% over-provisioning of note v0.3 does not hold either. Both numbers depended on where the budget sat relative to a distance step.

7. Figures fig1, fig2, fig3, fig5 and figS1 generated on 29 September 2026 from the definitions-driven data, Morandi palette, PNG and PDF. They are the step-3 sanity plots for Propositions 2, 3 and 5 and for the physical layer.

8. Two checks behind Figure 2 (`scripts/check_fig2.py`, results in `data/fig2_checks.json`). Interior slices of the H4 circuit: excess of the summed sensitivity over the enclosing generator boundaries has correlation 0.91 with sin^2 of the string angle, slope 6.2 Ha, mean excess 2.2 Ha inside single excitations and -0.04 Ha inside double excitations. Ordering: rebuilding H4 with beta first leaves every sensitivity unchanged qubit by qubit and the energy unchanged to 1e-15 Ha, so the alpha/beta asymmetry of the X row belongs to the qubit position (the Jordan-Wigner string), not to the spin.

Open after step 1: the general expression of s_X,i in terms of the 1- and 2-RDM (Proposition 3, general form); the bond-length scans; the full-space 12-qubit LiH; the refined merge rule; D25.
