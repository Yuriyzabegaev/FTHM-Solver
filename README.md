# Preconditioners for Thermo-PoroMechanics with Frictional Contact Mechanics of Fractures 

This repository contains the source code of the algorithms for two publications:
* for the isothermal problem, see the [pre-print](https://arxiv.org/abs/2501.07441). The code is located in `main` branch ([switch to it](https://github.com/Yuriyzabegaev/FTHM-Solver/tree/main)).
* for the non-isothermal problem, see the pre-print (to be done). The code is located in the `thermal` branch (you are here now).

The experiments are based on [PorePy](https://github.com/pmgbergen/porepy) and [PETSc](https://petsc.org/).

# Reproduce the non-isothermal experiments

A Docker image with the full environment is available on Zenodo (TODO). Download the image and run these commands ([Docker](https://www.docker.com/) should be installed):
```
docker load -i fthm_solver.tar.gz
docker run -it --name fthm_solver fthm_solver:latest
docker exec -it fthm_solver /bin/bash
```
Please don't forget to pull the recent changes with `git pull`.

The code is tested and works with porepy commit `65199b1a609af269d3a44204a06f8c97f3891d65`:

(do `git checkout 65199b1a609af269d3a44204a06f8c97f3891d65` in the porepy repo).

In the container, run the experiments: `python thermal/thermal_model_4.py` corresponds to the 2D experiment, while `python thermal/thermal_model_5.py` corresponds for the 3D experiment. The experiment parameters can be adjusted in the corresponding files. Results can be visualized in jupyter notebooks in the same folder, I use VSCode for it.

# Understand the code

See the [tutorial](tutorial.ipynb).