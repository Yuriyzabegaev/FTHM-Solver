import pickle
from collections import defaultdict
from copy import copy
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plot_utils import load_data
from solver_selection_thm.performance_predictor import (
    EpsGreedyExplorationModel,
    IncrementalRefitModel,
    InitialExplorationEstimator,
    TwoEstimators,
)
from solver_selection_thm.pp_binding import KNOWN_SOLVER_COMPONENTS_THM
from solver_selection_thm.selector import SolverSelector
from solver_selection_thm.solver_space import SolverSpace
from solver_selection_thm.spe_physics import X_SLICES, Z_SLICES
from solver_selection_thm.spe_physics import simulation_name as simulation_name_spe
from solver_selection_thm.thm_physics import inlet_placements, outlet_placements, params
from solver_selection_thm.thm_physics import simulation_name as simulation_name_thm


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
        # num_solvers = len(solver_space.all_decisions_encoding)
        # print("Num solvers:", num_solvers)

        performance_predictor = InitialExplorationEstimator(
            num_initial_exploration=64,
            batch_size=64,
            model=EpsGreedyExplorationModel(
                eps=0,
                eps1=0.9,
                model=TwoEstimators(
                    classifier=IncrementalRefitModel(
                        model=make_pipeline(StandardScaler(), RidgeClassifier())
                    ),
                    regressor=IncrementalRefitModel(
                        model=GradientBoostingRegressor(random_state=run_idx)
                    ),
                ),
            ),
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
        try:
            with open(f"{dir}spe_solver_space_scheme_run_{run_idx}.pkl", "rb") as f:
                solver_space_scheme = pickle.load(f)
        except FileNotFoundError:
            print(f"Was not able to load experiment {run_idx}, skipping it.")
            continue
        # Load permutations
        with open(f"{dir}spe_permutations_{run_idx}.pkl", "rb") as f:
            permutations = pickle.load(f)
            permutations_x = permutations["x"]
            permutations_z = permutations["z"]

        solver_space = SolverSpace(
            solver_space_scheme=solver_space_scheme,
            solver_scheme_builders=KNOWN_SOLVER_COMPONENTS_THM,
        )
        # num_solvers = len(solver_space.all_decisions_encoding)
        # print("Num solvers:", num_solvers)

        performance_predictor = InitialExplorationEstimator(
            num_initial_exploration=64,
            batch_size=64,
            model=EpsGreedyExplorationModel(
                eps=0,
                eps1=0.9,
                model=TwoEstimators(
                    classifier=IncrementalRefitModel(
                        model=make_pipeline(StandardScaler(), RidgeClassifier())
                    ),
                    regressor=IncrementalRefitModel(
                        model=GradientBoostingRegressor(random_state=run_idx)
                    ),
                ),
            ),
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


def make_pandas(sim_data, perf_data, seq_ids):
    sim_data_dict = defaultdict(lambda: [])
    perf_data_dict = defaultdict(lambda: [])
    for seq_id, data_simulations, solver_selection_history_seq in zip(
        seq_ids, sim_data, perf_data
    ):
        sim_idx = -1
        for data_row in data_simulations:
            for data in data_row:
                sim_idx += 1

                for ts_idx, ts in enumerate(data):
                    for ls_idx, ls in enumerate(ts.linear_solves):
                        sim_data_dict["seq_id"].append(seq_id)
                        sim_data_dict["sim_idx"].append(sim_idx)
                        sim_data_dict["ts_idx"].append(ts_idx)
                        sim_data_dict["ls_idx"].append(ls_idx)
                        sim_data_dict["real_solve_time"].append(ls.linear_solve_time)
                        sim_data_dict["krylov_iters"].append(ls.krylov_iters)
                        sim_data_dict["petsc_converged_reason"].append(
                            ls.petsc_converged_reason
                        )
                        sim_data_dict["cfl"].append(ls.cfl)
                        sim_data_dict["simulation_dt"].append(ls.simulation_dt)
                        sim_data_dict["enthalpy_max"].append(ls.enthalpy_max)
                        sim_data_dict["enthalpy_mean"].append(ls.enthalpy_mean)
                        sim_data_dict["fourier_max"].append(ls.fourier_max)
                        sim_data_dict["fourier_mean"].append(ls.fourier_mean)

        # they store the data incrementally (second has what first has and more)
        offset = 0

        for sim_idx, solver_selection_history in enumerate(
            solver_selection_history_seq
        ):
            num_data = len(solver_selection_history.reward) - offset
            perf_data_dict["seq_id"].extend([seq_id] * num_data)
            perf_data_dict["sim_idx"].extend([sim_idx] * num_data)
            perf_data_dict["reward"].extend(solver_selection_history.reward[-num_data:])
            perf_data_dict["expectation"].extend(
                solver_selection_history.expectation[-num_data:]
            )
            perf_data_dict["decision_idx"].extend(
                solver_selection_history.decision_idx[-num_data:]
            )
            perf_data_dict["features"].extend(
                solver_selection_history.features[-num_data:]
            )
            perf_data_dict["predict_time"].extend(
                solver_selection_history.predict_time[-num_data:]
            )
            perf_data_dict["fit_time"].extend(
                solver_selection_history.fit_time[-num_data:]
            )
            offset += num_data

    return pd.DataFrame(data=sim_data_dict), pd.DataFrame(data=perf_data_dict)
