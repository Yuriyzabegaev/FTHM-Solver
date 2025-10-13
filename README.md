# Preconditioners and solver selection for thermo-hydro-mechanics in porous media with fractures, governed by frictional contact mechanics 

This repository contains the source code of the algorithms for three publications:
* (isothermal preconditioner) [**An efficient preconditioner for mixed-dimensional contact poromechanics based on the fixed stress splitting scheme**](https://arxiv.org/abs/2501.07441). The code is located in `main` branch ([switch to it](https://github.com/Yuriyzabegaev/FTHM-Solver/tree/main)).
* (non-isothermal preconditioner) [**A block preconditioner for thermo-poromechanics with frictional deformation of fractures**](
https://doi.org/10.48550/arXiv.2505.04247). The code is located in the `thermal` branch (you are here now).
* (machine learning for solver selection) [**Data-driven linear solver selection and performance tuning for multiphysics simulations in porous media**](
https://doi.org/10.48550/arXiv.2510.04920). The code is located in the `solver_selection` branch ([switch to it](https://github.com/Yuriyzabegaev/FTHM-Solver/tree/solver_selection)).


The code in this branch implements a preconditioner for the thermo-hydro-mechanics problem in porous media with fractures, governed by frictional contact mechanics. The numerical experiments consider:
- the 2D and 3D model problems;
- grid refinement;
- various friction coefficients;
- various Peclet numbers;

The implementation is based on [PorePy](https://github.com/pmgbergen/porepy) and [PETSc](https://petsc.org/).

# Reproduce the experiments

Here you can find the information on how to:
- [Prepare the environment](README.md#step-1-installation)
- [Analyse and visualize the performance data of our numerical experiments](README.md#step-2-accessing-the-data-of-our-numerical-experiments)
- [Reproduce the numerical experiements](README.md#step-3-re-running-the-numerical-experiments)

## Step 1. Installation

To reproduce the experiments, PETSc, PorePy and all their dependencies must be installey. To skip this tedious process, it is highly recommended to use [Docker](https://www.docker.com/). We provide a *docker image*, which has all the needed dependencies packed into it and ensures reproducibility of the experiments. You need to install Docker following [the official instruction](https://www.docker.com/get-started/). The instructions below are *hopefully* agnostic to the operation system, but were tested on Windows 11 with Windows Subsystem for Linux 2 (WSL 2) installed.


When Docker is installed and running, download [the image of this repository](https://doi.org/10.5281/zenodo.15350993). Note that it is a few gigabytes. 
To start a *docker container* based on this image, navigate to the folder you've downloaded it and run these terminal commands:
```
docker load -i fthm_solver.tar.gz   # This may take a few minutes
docker run -dit --name fthm_solver sfthm_solver:latest
docker exec -it fthm_solver /bin/bash
```

Please don't forget to pull the recent changes with `git pull`.

For the graphical user interface, it is recommended to use [VSCode](https://code.visualstudio.com/) and attach to the running container, following [the official instruction](https://code.visualstudio.com/docs/devcontainers/attach-container).

The repository root folder is located at `/home/porepy/solver_selection_thm` in the *docker container* file system.

The numerical experiments are tested using `Python 3.11.7`, PorePy commit `65199b1a609af269d3a44204a06f8c97f3891d65` and PETSc commit `bff66efa9044f546ae447ed195723e21295eb6dd`. All the other dependencies are listed in [requirements.txt](requirements.txt). The *docker container* has the correct versions pre-installed, so you don't need to do anything else.

## Step 2. Reproducing the numerical experiments

To run all the simulations from the paper, use the following command from the root folder of this repository:
```
python thermal/thermal_runscript_4.py friction    # 2D model, varying friction coefficients
python thermal/thermal_runscript_4.py peclet      # 2D model, varying Peclet number
python thermal/thermal_runscript_4.py refinement  # 2D model, grid refinement
python thermal/thermal_runscript_4.py direct      # 2D model, grid refinement with a direct solver

python thermal/thermal_runscript_5.py refinement  # 3D model, grid refinement
python thermal/thermal_runscript_5.py direct      # 3D model, grid refinement with a direct solver
```
Executing all these experiments takes about a real world week on our .


When the experiments are complete, you can use the jupyter notebooks to visualize and analyse the data:
- [thermal/experiments_thermal_4_results.ipynb](thermal/experiments_thermal_4_results.ipynb) - 2D grid refinement;
- [thermal/variations_4.ipynb](thermal/variations_4.ipynb) - 2D friction coefficient and Peclet;
- [thermal/experiments_thermal_5_results.ipynb](thermal/experiments_thermal_5_results.ipynb) - 3D grid refinement.


# Acknowledgement

This project has received funding from the VISTA program, The Norwegian Academy of Science and Letters and Equinor, from the Research Council of Norway (grant 308733), and from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 101002507).