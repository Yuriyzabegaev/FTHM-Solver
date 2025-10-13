FROM porepy-petsc:latest

RUN sudo apt update && sudo apt upgrade -y

RUN git -C ${HOME}/porepy fetch && \
    git -C ${HOME}/porepy checkout 65199b1a609af269d3a44204a06f8c97f3891d65

RUN git clone -b thermal https://github.com/Yuriyzabegaev/FTHM-Solver.git fthm_solver && \
    pip install --no-cache-dir -r fthm_solver/requirements.txt

ENV PYTHONPATH=${PYTHONPATH}:${HOME}/fthm_solver

WORKDIR ${HOME}/fthm_solver

ENTRYPOINT [ "/bin/bash" ]