# Preconditioners for Thermo-PoroMechanics with Frictional Contact Mechanics of Fractures 

This repository contains the source code of the algorithms for two publications:
* for the isothermal problem, see the [pre-print](https://arxiv.org/abs/2501.07441). The code is located in `main` branch (you are here now).
* for the non-isothermal problem, see the pre-print (to be done). The code is located in the `thermal` branch ([switch to it](https://github.com/Yuriyzabegaev/FTHM-Solver/tree/thermal)).

The experiments are based on [PorePy](https://github.com/pmgbergen/porepy) and [PETSc](https://petsc.org/).

# Reproduce the isothermal experiments

A Docker image with the full environment is available on Zenodo ([here](https://zenodo.org/records/14609885)). Download the image and run these commands ([Docker](https://www.docker.com/) should be installed):
```
docker load -i fhm_solver.tar.gz
docker run -it --name fhm_solver fhm_solver:latest
docker exec -it fhm_solver /bin/bash
```
Please don't forget to pull the recent changes with `git pull`.

The code is tested and works with porepy commit `9befd80f7c22c4818518d0714452b44502306c4b`: 

(do `git checkout 9befd80f7c22c4818518d0714452b44502306c4b` in the porepy repo).

In the container, run the [experiments](experiments/) with `python`. Their results can be visualized in jupyter notebooks in the same folder, I use VSCode for it.

# Understand the code

See the [tutorial](tutorial.ipynb).
