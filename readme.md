# Safe Control of Multi-Agent Systems with Minimal Communication

This repository contains the code accompanying our conference paper "Safe Control of Multi-Agent Systems with Minimal Communication" at IEEE CDC 2025 (Paper link: [https://web.eecs.umich.edu/~necmiye/pubs/YangYO_cdc25.pdf](https://web.eecs.umich.edu/~necmiye/pubs/YangYO_cdc25.pdf)). The code is based on the respository [(https://github.com/aaspeel/lowRankControl)](https://github.com/aaspeel/lowRankControl) by Antoine Aspeel.

The provided scripts reproduce the results presented in the Numerical Evaluation section of the paper. In particular, the code implements four experiments in which vehicles are required to start from an initial region (“start box”), remain within the safety constraints, and reach a designated terminal region (“end box”):
- two vehicles, where one operates with small process noise while the other has large noise and limited control authority,
- two vehicles, each having partial knowledge of its own state and partial information about the other vehicle,
- two vehicles, where one vehicle has access to its absolute position while the other can only sense relative position,
- four vehicles, where only one vehicle has absolute position information and the remaining three rely solely on relative sensing.

In all scenarios, communication delays can be introduced and analyzed to study their effects on coordination and performance.


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
