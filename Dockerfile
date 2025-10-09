FROM porepy-petsc:latest

RUN sudo apt update && sudo apt upgrade -y

RUN git -C ${HOME}/porepy fetch && \
    git -C ${HOME}/porepy checkout 65199b1a609af269d3a44204a06f8c97f3891d65

RUN git clone -b solver_selection https://github.com/Yuriyzabegaev/FTHM-Solver.git solver_selection_thm && \
    pip install --no-cache-dir -r solver_selection_thm/requirements.txt

ENV PYTHONPATH=${PYTHONPATH}:${HOME}/solver_selection_thm

WORKDIR ${HOME}/solver_selection_thm

ENTRYPOINT [ "/bin/bash" ]