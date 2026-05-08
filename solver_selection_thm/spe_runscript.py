import sys
import traceback
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from solver_selection_thm.load_experiments_data import (
    load_experiments_data_spe,
    make_pandas,
)
from solver_selection_thm.performance_predictor import (
    EpsGreedyExplorationModel,
    IncrementalRefitModel,
    InitialExplorationEstimator,
    PerformancePredictorRandom,
    TwoEstimators,
)
from solver_selection_thm.pp_binding import (
    KNOWN_SOLVER_COMPONENTS_THM,
    SolverSelectionMixinTH,
)
from solver_selection_thm.selector import SolverSelector
from solver_selection_thm.solver_space import (
    CategoricalChoices,
    NumericalChoices,
    SolverSpace,
)
from solver_selection_thm.spe_physics import (
    X_SLICES,
    Z_SLICES,
    SPE10Model,
    params,
    run,
    simulation_name,
)
from stats import StatisticsSavingMixin


class ModelSPEWithSelector(StatisticsSavingMixin, SolverSelectionMixinTH, SPE10Model):
    """The class defines the simulation for Sequence A."""

    def data_to_export(self):
        """Among others, export to file porosity and permeability fields."""
        data = super().data_to_export()
        sds = self.mdg.subdomains()
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        q = self._evaluate_and_scale(sds, "porosity", "m^3")
        perm = self._spe10_perm.ravel(order="f")
        for id, sd in enumerate(sds):
            data.append(
                (
                    sd,
                    "porosity",
                    q[cell_offsets[id] : cell_offsets[id + 1]],
                )
            )
            data.append(
                (
                    sd,
                    "permeability",
                    perm[cell_offsets[id] : cell_offsets[id + 1]],
                )
            )
        return data


def make_solver_space_scheme_hm(nd: int):
    """Describe the range of available options for solver selection in Sequence A."""
    flow = [0]
    temp = [1]
    SYSTEM_AMG_OR_ILU = {
        "block_type": "PetscFieldSplitScheme",
        "groups": flow + temp,
        "python_pc": {
            "block_type": "PcPythonPermutation",
            "permutation_type": "pt_permutation",
            "p_groups": flow,
            "t_groups": temp,
            "block_size": 2,
        },
        "elim_options": CategoricalChoices(
            [
                {
                    "python_pc_type": "ilu",
                    "python_pc_factor_levels": NumericalChoices([0, 1, 2]),
                },
                {"python_pc_type": "sor"},
                {"python_pc_type": "pbjacobi"},
                {
                    "python_pc_type": "hypre",
                    "python_pc_hypre_type": "boomeramg",
                    "python_pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                        [0.5, 0.6, 0.7, 0.8, 0.9]
                    ),
                    "python_pc_hypre_boomeramg_P_max": 16,
                    "python_pc_hypre_boomeramg_agg_nl": NumericalChoices([0, 1, 2]),
                    "python_pc_hypre_boomeramg_relax_type_all": CategoricalChoices(
                        [
                            "symmetric-SOR/Jacobi",
                            "l1scaled-Jacobi",
                            "SOR/Jacobi",
                            "Jacobi",
                        ]
                    ),
                    "python_pc_hypre_boomeramg_cycle_type": CategoricalChoices(
                        ["V", "W"]
                    ),
                    "python_pc_hypre_boomeramg_grid_sweeps_all": NumericalChoices(
                        [1, 2, 3]
                    ),
                },
                {
                    "python_pc_type": "gamg",
                    "python_pc_gamg_threshold": NumericalChoices([0, 0.01, 0.05, 0.1]),
                    "python_pc_gamg_agg_nsmooths": NumericalChoices([0, 1]),
                    "python_pc_gamg_aggressive_coarsening": NumericalChoices([1, 2]),
                    "python_pc_mg_cycle_type": CategoricalChoices(["V", "W"]),
                    "python_mg_levels_ksp_max_it": NumericalChoices([1, 2, 4]),
                    "python_mg_levels_pc_type": CategoricalChoices(["sor", "pbjacobi"]),
                },
            ]
        ),
    }
    CPR = {
        "block_type": "PetscCompositeScheme",
        "groups": flow + temp,
        "solvers": {
            0: {
                "block_type": "PetscFieldSplitScheme",
                "groups": flow,
                "fieldsplit_options": {
                    "pc_fieldsplit_type": "additive",
                },
                "elim_options": CategoricalChoices(
                    [
                        {
                            "pc_type": "hypre",
                            "pc_hypre_type": "boomeramg",
                            "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                                [0.5, 0.6, 0.7, 0.8, 0.9]
                            ),
                            "pc_hypre_boomeramg_P_max": 16,
                            "pc_hypre_boomeramg_agg_nl": NumericalChoices([0, 1, 2]),
                            "pc_hypre_boomeramg_relax_type_all": CategoricalChoices(
                                [
                                    "symmetric-SOR/Jacobi",
                                    "l1scaled-Jacobi",
                                    "SOR/Jacobi",
                                    "Jacobi",
                                ]
                            ),
                            "pc_hypre_boomeramg_cycle_type": CategoricalChoices(
                                ["V", "W"]
                            ),
                            "pc_hypre_boomeramg_grid_sweeps_all": NumericalChoices(
                                [1, 2, 3]
                            ),
                        },
                        {
                            "pc_type": "gamg",
                            "pc_gamg_threshold": NumericalChoices([0, 0.01, 0.05, 0.1]),
                            "pc_gamg_agg_nsmooths": NumericalChoices([0, 1]),
                            "pc_gamg_aggressive_coarsening": NumericalChoices([1, 2]),
                            "pc_mg_cycle_type": CategoricalChoices(["V", "W"]),
                            "mg_levels_ksp_max_it": NumericalChoices([1, 2, 4]),
                            "mg_levels_pc_type": CategoricalChoices(
                                ["sor", "pbjacobi"]
                            ),
                        },
                    ]
                ),
                "complement": {
                    "block_type": "PetscFieldSplitScheme",
                    "groups": temp,
                    "elim_options": {
                        "pc_type": CategoricalChoices(
                            [
                                "sor",
                                "jacobi",
                                "none",
                            ]
                        ),
                    },
                },
            },
            1: {
                "block_type": "PetscFieldSplitScheme",
                "groups": flow + temp,
                "python_pc": {
                    "block_type": "PcPythonPermutation",
                    "permutation_type": "pt_permutation",
                    "p_groups": flow,
                    "t_groups": temp,
                    "block_size": 2,
                },
                "elim_options": {
                    "python_pc_type": CategoricalChoices(
                        [
                            "ilu",
                            "sor",
                            "pbjacobi",
                        ]
                    ),
                },
            },
        },
    }

    return {
        "block_type": "PetscKSPScheme",
        "petsc_options": {
            "ksp_monitor": None,
            "ksp_rtol": 1e-12,
            "ksp_gmres_restart": NumericalChoices(
                [
                    30,
                    50,
                    100,
                ]
            ),
        },
        "compute_eigenvalues": False,
        "preconditioner": {
            "block_type": CategoricalChoices(
                [
                    SYSTEM_AMG_OR_ILU,
                    CPR,
                ]
            ),
        },
    }


if __name__ == "__main__":
    Path("stats/").mkdir(exist_ok=True)

    import pickle

    NUM_RUNS = 5
    IDX_START = 200
    if len(sys.argv) == 3:
        run_idx = IDX_START + int(sys.argv[1])
        CASE: Literal["solver_selection", "random", "expert", "tmp"] = sys.argv[2]
    else:
        print(
            'Command line arguments: run_index (int), case ("solver_selection", "random", "expert")'
        )
        run_idx = IDX_START
        CASE = "tmp"

    # Generate and save to file the geometries for all experiments in Sequence A.

    np.random.seed(run_idx)
    Z_SLICES = np.array(Z_SLICES)
    X_SLICES = np.array(X_SLICES)
    permutations_z = [np.random.permutation(len(Z_SLICES)) for i in range(5)]
    permutations_x = [np.random.permutation(len(X_SLICES)) for i in range(5)]
    solver_space_scheme = make_solver_space_scheme_hm(nd=3)

    print("Starting run", run_idx, CASE)

    with open(f"stats/spe_solver_space_scheme_run_{run_idx}.pkl", "wb") as f:
        pickle.dump(solver_space_scheme, f)
    with open(f"stats/spe_permutations_{run_idx}.pkl", "wb") as f:
        pickle.dump(
            {
                "x": permutations_x[run_idx - IDX_START],
                "z": permutations_z[run_idx - IDX_START],
            },
            f,
        )

    solver_space = SolverSpace(
        solver_space_scheme=solver_space_scheme,
        solver_scheme_builders=KNOWN_SOLVER_COMPONENTS_THM,
    )
    num_solvers = solver_space.all_decisions_encoding.shape[0]
    print(solver_space.decision_tree)
    print("Num solvers:", num_solvers)

    if CASE == "random":
        # Create a random performance predictor.
        performance_predictor = PerformancePredictorRandom(num_solvers=num_solvers)
    elif CASE == "solver_selection":
        # Create the ML-based performance predictor.
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
    elif CASE in ["expert", "tmp"]:
        # Create the ML-based performance predictor and load the data from all the past
        # experiments.
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
        offline_runs = [200, 201, 202, 203, 204]
        sim_data_random, perf_data_random, _ = load_experiments_data_spe(
            runs=offline_runs, case="random", dir="./stats/"
        )
        sim_data, perf_data, _ = load_experiments_data_spe(
            runs=offline_runs, case="solver_selection", dir="./stats/"
        )
        df_sim_rand, df_perf_rand = make_pandas(
            sim_data=sim_data_random,
            perf_data=perf_data_random,
            seq_ids=offline_runs,
        )
        df_sim, df_perf = make_pandas(
            sim_data=sim_data,
            perf_data=perf_data,
            seq_ids=offline_runs,
        )

        df_perf = pd.concat([df_perf_rand, df_perf], axis=0)
        X = np.stack(df_perf.features)
        y = np.array(df_perf.reward)
        performance_predictor.model.fit(X, y)
        performance_predictor.X_history = X.tolist()
        performance_predictor.y_history = y.tolist()
        performance_predictor.is_ready_to_predict = True

    else:
        raise ValueError(CASE)

    solver_selector = SolverSelector(
        solver_space=solver_space,
        performance_predictor=performance_predictor,
    )
    params["setup"]["linear_solver_selector"] = solver_selector

    # Run the experiments in Sequence A.

    for z_slice in Z_SLICES[permutations_z[run_idx - IDX_START]]:
        for x_slice in X_SLICES[permutations_x[run_idx - IDX_START]]:
            params["x_slice"] = x_slice
            params["z_slice"] = z_slice
            sim_name = f"run_{run_idx}_{simulation_name(params)}"
            if CASE == "random":
                sim_name = f"RANDOM_{sim_name}"
            elif CASE == "solver_selection":
                pass  # do nothing
            elif CASE == "expert":
                sim_name = f"EXPERT_{sim_name}"
            elif CASE == "tmp":
                sim_name = f"TMP_{sim_name}"
            else:
                raise ValueError(CASE)
            params["folder_name"] = sim_name
            model = ModelSPEWithSelector(params)
            print(model.simulation_name())
            model.prepare_simulation()
            print("Running")
            try:
                run(model, params)
            except Exception as e:
                traceback.print_exc()
            solver_selector.history.save(
                f"./stats/solver_selection_history_{sim_name}.npy"
            )
            print("Done")
