from solver_selection_thm.thm_physics import ModelTHM, initialize, run, params
from solver_selection_thm.selector import SolverSelector
from solver_selection_thm.solver_space import CategoricalChoices, NumericalChoices
from solver_selection_thm.performance_predictor import (
    PerformancePredictorPassiveAgressive,
    PerformancePredictorEpsGreedy,
)
from solver_selection_thm.solver_space import SolverSpace
from solver_selection_thm.pp_binding import KNOWN_SOLVER_COMPONENTS_THM
import numpy as np

import pickle
from itertools import count
from copy import copy
from plot_utils import load_data
from solver_selection_thm.thm_physics import (
    simulation_name,
    inlet_placements,
    outlet_placements,
)


def load_experiments_data_thm(runs: list[int], random_selection: bool):

    data_simulations_common = []
    solver_selection_history_common = []

    for run_idx in runs:

        with open(f"../stats/thm_solver_space_scheme_run_{run_idx}.pkl", "rb") as f:
            solver_space_scheme = pickle.load(f)

        # Load permutations
        with open(f"../stats/thm_permutations_{run_idx}.pkl", "rb") as f:
            permutations = pickle.load(f)
            permutations_inlet = permutations["inlet"]
            permutations_outlet = permutations["outlet"]

        solver_space = SolverSpace(
            solver_space_scheme=solver_space_scheme,
            solver_scheme_builders=KNOWN_SOLVER_COMPONENTS_THM,
        )
        num_solvers = len(solver_space.all_decisions_encoding)
        print("Num solvers:", num_solvers)

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
                sim_name = f"run_{run_idx}_{simulation_name(params)}"
                if random_selection:
                    sim_name = f"RANDOM_{sim_name}"
                try:
                    data = load_data(f"../stats/{sim_name}.json")
                    data_row.append(data)
                    solver_selector.history.load(
                        f"../stats/solver_selection_history_{sim_name}.npy"
                    )
                    solver_selection_history.append(copy(solver_selector.history))
                except FileNotFoundError:
                    print("failed to load", sim_name)

    return data_simulations_common, solver_selection_history_common, solver_selector
