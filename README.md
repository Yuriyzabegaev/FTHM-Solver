# Preconditioners and solver selection for thermo-hydro-mechanics in porous media with fractures, governed by frictional contact mechanics 

This repository contains the source code of the algorithms for three publications:
* (isothermal preconditioner) [**An efficient preconditioner for mixed-dimensional contact poromechanics based on the fixed stress splitting scheme**](https://arxiv.org/abs/2501.07441). The code is located in `main` branch ([switch to it](https://github.com/Yuriyzabegaev/FTHM-Solver/tree/main)).
* (non-isothermal preconditioner) [**A block preconditioner for thermo-poromechanics with frictional deformation of fractures**](
https://doi.org/10.48550/arXiv.2505.04247). The code is located in the `thermal` branch ([switch to it](https://github.com/Yuriyzabegaev/FTHM-Solver/tree/thermal)).
* (machine learning for solver selection) [**Data-driven linear solver selection and performance tuning for multiphysics simulations in porous media**](
https://doi.org/10.48550/arXiv.2510.04920). The code is located in the `solver_selection` branch (you are here now).


The code in this branch implements a machine-learning-based preconditioned linear solver selection algorithm for multphysics simulations. Two model problems are considered:
- Coupled flow and heat transport in porous media with heterogeneous permeability.
- Coupled thermo-hydro-mechanics in porous media with fractures, governed by frictional contact mechanics.

The implementation is based on [PorePy](https://github.com/pmgbergen/porepy) and [PETSc](https://petsc.org/).

# Reproduce the experiments

Here you can find the information on how to:
- [Prepare the environment](README.md#step-1-installation)
- [Analyse and visualize the performance data of our numerical experiments](README.md#step-2-accessing-the-data-of-our-numerical-experiments)
- [Reproduce the numerical experiements](README.md#step-3-re-running-the-numerical-experiments)

## Step 1. Installation

To reproduce the experiments, PETSc, PorePy and all their dependencies must be installey. To skip this tedious process, it is highly recommended to use [Docker](https://www.docker.com/). We provide a *docker image*, which has all the needed dependencies packed into it and ensures reproducibility of the experiments. You need to install Docker following [the official instruction](https://www.docker.com/get-started/).

When Docker is installed and running, download [the image of this repository](todo). Note that it is a few GBs. 
To start a *docker container* based on this image, navigate to the folder you've downloaded it and run these terminal commands:
```
docker load -i fhm_solver.tar.gz
docker run -it --name fhm_solver fhm_solver:latest
docker exec -it fhm_solver /bin/bash
```

Please don't forget to pull the recent changes with `git pull`.

For the graphical user interface, it is recommended to use [VSCode](https://code.visualstudio.com/) and attach to the running container, following [the official instruction](https://code.visualstudio.com/docs/devcontainers/attach-container).

## Step 2. Accessing the data of our numerical experiments

The data of the experiments, used to generate figures and tables for the publications are commited to the [stats/](stats/) folder. You can see the datasets in Pandas and re-generate the figures and the tables using these jupyter notebooks:
- Sequence A - coupled flow and heat transport: [solver_selection_thm/spe_results.ipynb](solver_selection_thm/spe_results.ipynb);
- Sequence B - contact-THM: [solver_selection_thm/thm_results.ipynb](solver_selection_thm/thm_results.ipynb).

## Step 3. Re-running the numerical experiments

If you want re-run the simulations, we recommend you to delete the contents of the `stats/` folder to avoid mixing our data with yours. You can always restore the original data using git.

To run all the simulations from the paper, use the following command from the root folder of this repository:
```
sh solver_selection_thm/runscript.sh
```
It will run the following experiments:
- Random selection experiment (Section 6.1 in the paper);
- Solver selection experiment (Section 6.2 in the paper);
- Expert selection experiment (Section 6.3 in the paper).

Each experiment will run for both Sequence A and Sequence B. Each experiment will be repeated 5 times to account for randomness. Each of 5 repetitions is assigned a `run_id` within `[200, 201, 202, 203, 204]` (successively). These `run_ids` are used in the file names and in the visualization notebooks. 

**Warning**: Executing this command takes about **a real world week** on our machine, so you might not want to wait for it to complete.

You can instead run the experiments selectively. To run the experiments for Sequence A with only one repetition, use the following commands from the root folder of this repository:
```
python solver_selection_thm/spe_runscript.py 0 random            # section 6.1 experiment
python solver_selection_thm/spe_runscript.py 0 solver_selection  # section 6.2 experiment
```
The first command-line argument takes a value from 0 to 4 to account for the repetition index. This takes about 2.5 hours on our machine.

For Sequence B, use these commands from the root folder of this repository:
 ```
python solver_selection_thm/thm_runscript.py 0 random            # section 6.1 experiment
python solver_selection_thm/thm_runscript.py 0 solver_selection  # section 6.2 experiment
```
Sequence B takes about 10 hours on our machine and its peak memory consumption is ~20 GBs.

The experiments in Section 6.3 are based on the data, previously collected in Sections 6.1 and 6.2, so you need to restore our data with git, if you have deleted it before. Use the following commands from the root folder of this repository to run them:
```
python solver_selection_thm/spe_runscript.py 0 expert  # Sequence A
python solver_selection_thm/thm_runscript.py 0 expert  # Sequence B
```

When the experiments are complete, you can return to [Step 2](README.md#step-2-accessing-the-data-of-our-numerical-experiments) to visualize and analyse the data.