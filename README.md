# An Effificent Preconditioner For Mixed-Dimensional Contact Poromechanics Based On The Fixed Stress Splitting Scheme

This repository contains the source code of the algorithm from the publication (to be done). This implements an iterative linear solver to address the contact poromechanics problem. The implementation is based on [PorePy](https://github.com/pmgbergen/porepy) and [PETSc](https://petsc.org/).

# Reproduce the experiments

A Docker image with the full environment is available on Zenodo ([here](https://zenodo.org/records/14609885)). Download the image and run these commands ([Docker](https://www.docker.com/) should be installed):
```
docker load -i fhm_solver.tar.gz
docker run -it --name fhm_solver fhm_solver:latest
docker exec -it fhm_solver /bin/bash
```
Please don't forget to pull the recent changes with `git pull`.

In the container, run the [experiments](experiments/) with `python`. Their results can be visualized in jupyter notebooks in the same folder, I use VSCode for it.

# Understand the code

See the [tutorial](tutorial.ipynb).
