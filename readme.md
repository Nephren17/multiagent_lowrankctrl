# Safe Control of Multi-Agent Systems with Minimal Communication

This repository contains the code accompanying our conference paper "Safe Control of Multi-Agent Systems with Minimal Communication" at IEEE CDC 2025 (Paper link: [[[https://web.eecs.umich.edu/~necmiye/pubs/YangYO_cdc25.pdf](https://arxiv.org/abs/2512.13021)](https://arxiv.org/abs/2512.13021)](https://web.eecs.umich.edu/~necmiye/pubs/YangYO_cdc25.pdf)). The code is based on the respository [(https://github.com/aaspeel/lowRankControl)](https://github.com/aaspeel/lowRankControl) by Antoine Aspeel.

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

To run the code solving the optimization problems for the nuclear norm, sensor norm and actuator norm cases and reproducing the results and figures in section "Numerical Evaluation", run the following command:

```bash
python3 simulation_four_relative.py
```

Other experiment can also be run in this way. The results and plots would be found in the corresponding folder.

## Appendix

The following additional scripts are used by `simulation.py` and `plots.py`.

1. `SLSFinite.py` and `Polytope.py` is directly from Aspeel's code.

2. `functions.py` defines the contraints and optimization of the problems.

3.  `functions_swarm.py` defines the functions that is used for cases with more than 2 agents.

4.  `simulation_xxx.py` are the experiments we designed.
