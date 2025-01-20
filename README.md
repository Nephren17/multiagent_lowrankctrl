# A Low Rank Approach to Minimize Sensor-to-Actuator Communication in Finite Horizon Output Feedback

The code accompaning our joint ACC 2024 and IEEE L-CSS paper submission.

Paper in IEEE L-CSS: https://ieeexplore.ieee.org/abstract/document/10336872

**Authors:** Antoine Aspeel, Jakob Nylof, Jing Shuang (Lisa) Li and Necmiye Ozay

The code reproduces the results in the section "Numerical Demonstrations" of the paper. In particular, the code:
1. Solves the nuclear norm minimization problem and computes a causal factorization of the optimal controller.
2. Solves the actuator and sensor norm minimization problems.
3. Plots the figures in the paper and prints out results of steps 1 and 2.

## Setup
From the base directory of this repository, install dependencies with:
~~~~
pip install -r requirements.txt
~~~~

## Run
To run the code solving the optimization problems for the nuclear norm, sensor norm and actuator norm cases and reproducing the results and figures in section "Numerical Demonstrations", run the following command:
~~~~
python3 simulation.py
~~~~

The figures and the file `simulationT20.pickle` containing the simulation data is saved in `simulation_results`.

To run the code only reproducing the figures using the previously saved simulation data in `simulation_results/simulationT20.pickle`, run the following command:
~~~~
python3 plots.py
~~~~

## Appendix

The following additional scripts are used by `simulation.py` and `plots.py`.
1. `SLSFinite.py` defines a class `SLSFinite` storing the optimization variables and parameters of the optimization problems. Methods of `SLSFinite` compute system level synthesis constraint and the causal factorization of the optimal controller.
2. `Polytope.py` defines a class `Polytope` that allows taking products and powers of polytopes, which facilitates defining polytope containment constraints.
3. `functions.py` defines the functions solving the respective optimization problems in steps 1 and 2.


