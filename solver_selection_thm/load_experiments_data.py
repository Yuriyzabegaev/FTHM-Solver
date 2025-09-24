import pickle
from copy import copy
from itertools import count
from typing import Literal

import numpy as np

from plot_utils import load_data
from solver_selection_thm.performance_predictor import (
    PerformancePredictorEpsGreedy,
    PerformancePredictorPassiveAgressive,
)
from solver_selection_thm.pp_binding import KNOWN_SOLVER_COMPONENTS_THM
from solver_selection_thm.selector import SolverSelector
from solver_selection_thm.solver_space import (
    CategoricalChoices,
    NumericalChoices,
    SolverSpace,
)
from solver_selection_thm.spe_physics import X_SLICES, Z_SLICES
from solver_selection_thm.spe_physics import simulation_name as simulation_name_spe
from solver_selection_thm.thm_physics import (
    ModelTHM,
    initialize,
    inlet_placements,
    outlet_placements,
    params,
    run,
)
from solver_selection_thm.thm_physics import (
    simulation_name as simulation_name_thm,
)


def load_experiments_data_thm(
    runs: list[int],
    case: Literal["expert", "random", "solver_selection"],
    dir="../stats/",
):
    data_simulations_common = []
    solver_selection_history_common = []
    solver_selector = None

    for run_idx in runs:
        with open(f"{dir}thm_solver_space_scheme_run_{run_idx}.pkl", "rb") as f:
            solver_space_scheme = pickle.load(f)

        # Load permutations
        with open(f"{dir}thm_permutations_{run_idx}.pkl", "rb") as f:
            permutations = pickle.load(f)
            permutations_inlet = permutations["inlet"]
            permutations_outlet = permutations["outlet"]

        solver_space = SolverSpace(
            solver_space_scheme=solver_space_scheme,
            solver_scheme_builders=KNOWN_SOLVER_COMPONENTS_THM,
        )
        num_solvers = len(solver_space.all_decisions_encoding)
        # print("Num solvers:", num_solvers)

        performance_predictor = PerformancePredictorPassiveAgressive(
            num_solvers=num_solvers,
        )
        solver_selector = SolverSelector(
            solver_space=solver_space,
            performance_predictor=performance_predictor,
        )

        data_simulations = []
        solver_selection_history = []
        data_simulations_common.append(data_simulations)
        solver_selection_history_common.append(solver_selection_history)

        inlet_placements_ = np.array(inlet_placements)
        outlet_placements_ = np.array(outlet_placements)

        for inlet_placement in inlet_placements_[permutations_inlet]:
            data_row = []
            data_simulations.append(data_row)
            for outlet_placement in outlet_placements_[permutations_outlet]:
                params["inlet_placement"] = inlet_placement
                params["outlet_placement"] = outlet_placement
                sim_name = f"run_{run_idx}_{simulation_name_thm(params)}"
                if case == "random":
                    sim_name = f"RANDOM_{sim_name}"
                elif case == "solver_selection":
                    pass
                elif case == "expert":
                    sim_name = f"EXPERT_{sim_name}"
                else:
                    raise ValueError(case)
                try:
                    data = load_data(f"{dir}{sim_name}.json")
                    data_row.append(data)
                    solver_selector.history.load(
                        f"{dir}solver_selection_history_{sim_name}.npy"
                    )
                    solver_selection_history.append(copy(solver_selector.history))
                except FileNotFoundError:
                    print("failed to load", sim_name)

    return data_simulations_common, solver_selection_history_common, solver_selector


def load_experiments_data_spe(
    runs: list[int],
    case: Literal["expert", "random", "solver_selection"],
    dir="../stats/",
):
    data_simulations_common = []
    solver_selection_history_common = []

    for run_idx in runs:
        with open(f"{dir}spe_solver_space_scheme_run_{run_idx}.pkl", "rb") as f:
            solver_space_scheme = pickle.load(f)

        # Load permutations
        with open(f"{dir}spe_permutations_{run_idx}.pkl", "rb") as f:
            permutations = pickle.load(f)
            permutations_x = permutations["x"]
            permutations_z = permutations["z"]

        solver_space = SolverSpace(
            solver_space_scheme=solver_space_scheme,
            solver_scheme_builders=KNOWN_SOLVER_COMPONENTS_THM,
        )
        num_solvers = len(solver_space.all_decisions_encoding)
        # print("Num solvers:", num_solvers)

        performance_predictor = PerformancePredictorPassiveAgressive(
            num_solvers=num_solvers,
        )
        solver_selector = SolverSelector(
            solver_space=solver_space,
            performance_predictor=performance_predictor,
        )

        data_simulations = []
        solver_selection_history = []
        data_simulations_common.append(data_simulations)
        solver_selection_history_common.append(solver_selection_history)

        Z_SLICES_ = np.array(Z_SLICES)
        X_SLICES_ = np.array(X_SLICES)

        for z_slice in Z_SLICES_[permutations_z]:
            data_row = []
            data_simulations.append(data_row)
            for x_slice in X_SLICES_[permutations_x]:
                params["x_slice"] = x_slice
                params["z_slice"] = z_slice
                sim_name = f"run_{run_idx}_{simulation_name_spe(params)}"
                if case == "random":
                    sim_name = f"RANDOM_{sim_name}"
                elif case == "solver_selection":
                    pass
                elif case == "expert":
                    sim_name = f"EXPERT_{sim_name}"
                else:
                    raise ValueError(case)
                try:
                    data = load_data(f"{dir}{sim_name}.json")
                    data_row.append(data)
                    solver_selector.history.load(
                        f"{dir}solver_selection_history_{sim_name}.npy"
                    )
                    solver_selection_history.append(copy(solver_selector.history))
                except FileNotFoundError:
                    print("failed to load", sim_name)

    return data_simulations_common, solver_selection_history_common, solver_selector
