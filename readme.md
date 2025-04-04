# Safe Control of Multi-Agent Systems with Minimal Communication

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

## Appendix

The following additional scripts are used by `simulation.py` and `plots.py`.

1. `SLSFinite.py` defines a class `SLSFinite` storing the optimization variables and parameters of the optimization problems. Methods of `SLSFinite` compute system level synthesis constraint and the causal factorization of the optimal controller.

2. `Polytope.py` defines a class `Polytope` that allows taking products and powers of polytopes, which facilitates defining polytope containment constraints.

3. `functions.py` defines the functions solving the respective optimization problems in steps 1 and 2.