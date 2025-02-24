# A Low Rank Approach to Minimize Output Feedback Multi-Agent System

## Setup

 From the base directory of this repository, install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

To run the code solving the optimization problems for the nuclear norm, sensor norm and actuator norm cases and reproducing the results and figures in section "Numerical Demonstrations", run the following command:

```bash
python3 simulation.py
```

The figures and the file `simulationT20.pickle` containing the simulation data is saved in `simulation_results`.

To run the code only reproducing the figures using the previously saved simulation data in `simulation_results/simulationT20.pickle`, run the following command:

```bash
python3 plots.py
```



## Functions

`simulation_mutraul_detection.py` is the experiment for two UWV where one agent can see the x positions of two agents and the other can only see the two y positions

`simulation_double_drone.py` is the experiment where one agent is with little control while the other have more.



In `simulation_mutraul_detection.py`,

- `offdiag_phiuy_only_data` is the optimization data with only `rank off-diag(phi_uy)` is optimized without other constraints.

- `offdiag_phixx_constrained_data` is the optimization data with `phi_Xx` constrained to be off-diag sparse and optimize `rank off-diag(phi_uy)`.

- `offdiag_three_phis_data` is the optimization data with `phi_Xx` constrained and optimize `rank off-diag(phi_uy) + rank off-diag(phi_ux) + rank off-diag(phi_xy)`.

- `no_comm_data` is the data with `phi_xx, phi_xy, phi_ux, phi_uy` constrained to be off-diag sparse.



In `SLSFinite.py`,

- `extract_Phi_subcom_mat` and `extract_offdiag_expr` are used to get off-diag parts of phi matrix and K matrix.
- `row_factorization_causal` is the function for calculating causal factorization for extracted sub-communication matrix.



In `Functions.py`

- `diagonal_identity_block_constraints_xx` and other diagonal constraints are constraints that kills off-diag part of phi matrix.



## Appendix

The following additional scripts are used by `simulation.py` and `plots.py`.

1. `SLSFinite.py` defines a class `SLSFinite` storing the optimization variables and parameters of the optimization problems. Methods of `SLSFinite` compute system level synthesis constraint and the causal factorization of the optimal controller.

2. `Polytope.py` defines a class `Polytope` that allows taking products and powers of polytopes, which facilitates defining polytope containment constraints.

3. `functions.py` defines the functions solving the respective optimization problems in steps 1 and 2.