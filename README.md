# Preconditioners and solver selection for thermo-hydro-mechanics in porous media with fractures, governed by frictional contact mechanics 

This repository contains the source code of the algorithms for three publications:
* (isothermal preconditioner) [**An efficient preconditioner for mixed-dimensional contact poromechanics based on the fixed stress splitting scheme**](https://arxiv.org/abs/2501.07441). The code is located in `main` branch (you are here now).
* (non-isothermal preconditioner) [**A block preconditioner for thermo-poromechanics with frictional deformation of fractures**](
https://doi.org/10.48550/arXiv.2505.04247). The code is located in the `thermal` branch ([switch to it](https://github.com/Yuriyzabegaev/FTHM-Solver/tree/thermal)).
* (machine learning for solver selection) [**Data-driven linear solver selection and performance tuning for multiphysics simulations in porous media**](
https://doi.org/10.48550/arXiv.2510.04920). The code is located in the `solver_selection` branch ([switch to it](https://github.com/Yuriyzabegaev/FTHM-Solver/tree/solver_selection)).

The implementation is based on [PorePy](https://github.com/pmgbergen/porepy) and [PETSc](https://petsc.org/).

# Reproduce the experiments

## Step 1. Installation

To reproduce the experiments, PETSc, PorePy and all their dependencies must be installey. To skip this tedious process, it is highly recommended to use [Docker](https://www.docker.com/). We provide a *docker image*, which has all the needed dependencies packed into it and ensures reproducibility of the experiments. You need to install Docker following [the official instruction](https://www.docker.com/get-started/). The instructions below are *hopefully* agnostic to the operation system, but were tested on Windows 11 with Windows Subsystem for Linux 2 (WSL 2) installed.


When Docker is installed and running, download [the image of this repository](https://doi.org/10.5281/zenodo.14609885). Note that it is a few gigabytes. 
To start a *docker container* based on this image, navigate to the folder you've downloaded it and run these terminal commands:
```
docker load -i fhm_solver.tar.gz
docker run -dit --name fhm_solver fhm_solver:latest
docker exec -it fhm_solver /bin/bash
```

Please don't forget to pull the recent changes with `git pull`.

For the graphical user interface, it is recommended to use [VSCode](https://code.visualstudio.com/) and attach to the running container, following [the official instruction](https://code.visualstudio.com/docs/devcontainers/attach-container).

The numerical experiments are tested using PorePy commit `9befd80f7c22c4818518d0714452b44502306c4b` (do `git checkout 9befd80f7c22c4818518d0714452b44502306c4b` in the porepy folder). All the other dependencies are listed in [requirements.txt](requirements.txt). The *docker container* has the correct versions pre-installed, so you don't need to do anything else.

## Step 2. Reproducing the numerical experiments

In the container, run the [experiments](experiments/) with `python`. Their results can be visualized in jupyter notebooks in the same folder.

# Acknowledgement

This project has received funding from the VISTA program, The Norwegian Academy of Science and Letters and Equinor, from the Research Council of Norway (grant 308733), and from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 101002507).
