from itertools import count
import traceback
from solver_selection_thm.thm_physics import (
    ModelTHM,
    initialize,
    run,
    params,
    inlet_placements,
    outlet_placements,
    simulation_name,
)
from solver_selection_thm.selector import SolverSelector
from solver_selection_thm.solver_space import CategoricalChoices, NumericalChoices
from solver_selection_thm.performance_predictor import (
    PerformancePredictorPassiveAgressive,
    PerformancePredictorEpsGreedy,
    RewardEstimator,
    PerformancePredictorRandom,
    Estimator,
)
from solver_selection_thm.solver_space import SolverSpace
from solver_selection_thm.pp_binding import (
    KNOWN_SOLVER_COMPONENTS_THM,
    SolverSelectionMixinTHM,
)
from stats import StatisticsSavingMixin


class ModelTHMWithSelector(StatisticsSavingMixin, SolverSelectionMixinTHM, ModelTHM):
    pass


contact = [0]
intf = [1, 2]
mech = [3, 4]
flow = [5, 6, 7]
temp = [8, 9, 10]


def make_solver_space_scheme_fthm(nd: int):
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
                # {
                #     "python_pc_type": "ilu",
                #     "python_pc_factor_levels": NumericalChoices([0, 1, 2]),
                # },
                # {
                #     "python_pc_type": "sor",
                # },
                # {
                #     "python_pc_type": "pbjacobi",
                # },
                {
                    "python_pc_type": "hypre",
                    "python_pc_hypre_type": "boomeramg",
                    "python_pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                        [
                            # 0.5,
                            # 0.7,
                            0.9,
                        ]
                    ),
                    "python_pc_hypre_boomeramg_P_max": 16,
                    "python_pc_hypre_boomeramg_agg_nl": NumericalChoices(
                        [
                            # 0,
                            # 1,
                            2,
                        ]
                    ),
                    "python_pc_hypre_boomeramg_relax_type_all": CategoricalChoices(
                        [
                            "symmetric-SOR/Jacobi",
                            # "l1scaled-Jacobi",
                            # "SOR/Jacobi",
                            # "Jacobi",
                        ]
                    ),
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
                "elim_options": {
                    "pc_type": "hypre",
                    "pc_hypre_type": "boomeramg",
                    "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                        [
                            # 0.5,
                            # 0.7,
                            0.9,
                        ]
                    ),
                    "pc_hypre_boomeramg_agg_nl": NumericalChoices(
                        [
                            # 0,
                            # 1,
                            2,
                        ]
                    ),
                    "pc_hypre_boomeramg_relax_type_all": CategoricalChoices(
                        [
                            "symmetric-SOR/Jacobi",
                            # "l1scaled-Jacobi",
                            # "SOR/Jacobi",
                            # "Jacobi",
                        ]
                    ),
                },
                "complement": {
                    "block_type": "PetscFieldSplitScheme",
                    "groups": temp,
                    "elim_options": {
                        "pc_type": CategoricalChoices(
                            [
                                "sor",
                                # "jacobi",
                                # "none",
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
        "block_type": "LinearTransformedScheme",
        "scale_energy_balance": True,
        "Qright": True,
        "inner": {
            "block_type": "PetscKSPScheme",
            "petsc_options": {
                "ksp_monitor": None,
                "ksp_rtol": 1e-12,
                "ksp_gmres_restart": NumericalChoices(
                    [
                        # 10,
                        30,
                        50,
                    ]
                ),
            },
            "compute_eigenvalues": False,
            "preconditioner": {
                "block_type": "PetscFieldSplitScheme",
                "groups": contact,
                "block_size": nd,
                "fieldsplit_options": {
                    "pc_fieldsplit_schur_precondition": "selfp",
                },
                "elim_options": {
                    "pc_type": "pbjacobi",
                },
                "keep_options": {
                    "mat_schur_complement_ainv_type": "blockdiag",
                },
                "complement": {
                    "block_type": "PetscFieldSplitScheme",
                    "groups": intf,
                    "elim_options": {
                        "pc_type": "ilu",
                    },
                    "fieldsplit_options": {
                        "pc_fieldsplit_schur_precondition": "selfp",
                    },
                    "complement": {
                        "block_type": "PetscFieldSplitScheme",
                        "groups": mech,
                        "elim_options": CategoricalChoices(
                            [
                                {
                                    "pc_type": "hypre",
                                    "pc_hypre_type": "boomeramg",
                                    "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                                        [
                                            0.5,
                                            0.7,
                                            0.9,
                                        ]
                                    ),
                                    "pc_hypre_boomeramg_agg_nl": NumericalChoices(
                                        [
                                            0,
                                            1,
                                            2,
                                        ]
                                    ),
                                    "pc_hypre_boomeramg_relax_type_all": CategoricalChoices(
                                        [
                                            "symmetric-SOR/Jacobi",
                                            "l1scaled-Jacobi",
                                            "SOR/Jacobi",
                                            "Jacobi",
                                        ]
                                    ),
                                },
                                {
                                    "pc_type": "gamg",
                                    "pc_gamg_threshold": NumericalChoices(
                                        [-1, 0, 0.2, 0.5, 0.7]
                                    ),
                                    "pc_gamg_agg_nsmooths": NumericalChoices([1, 2]),
                                    "pc_gamg_aggressive_coarsening": NumericalChoices(
                                        [1, 2]
                                    ),
                                },
                            ]
                        ),
                        "block_size": nd,
                        "invert": {
                            "block_type": "fs_analytical_slow_new",
                            "p_mat_group": 5,
                            "p_frac_group": 6,
                            "groups": flow + temp,
                        },
                        "complement": {
                            "block_type": CategoricalChoices(
                                [
                                    SYSTEM_AMG_OR_ILU,
                                    CPR,
                                ]
                            ),
                        },
                    },
                },
            },
        },
    }


import numpy as np

RANDOM_SELECTION = False

if __name__ == "__main__":
    import pickle

    NUM_RUNS = 5
    np.random.seed(42)

    inlet_placements = np.array(inlet_placements)
    outlet_placements = np.array(outlet_placements)

    permutations_inlet = [
        np.random.permutation(len(inlet_placements)) for i in range(NUM_RUNS)
    ]
    permutations_outlet = [
        np.random.permutation(len(outlet_placements)) for i in range(NUM_RUNS)
    ]

    IDX_START = 20
    solver_space_scheme = make_solver_space_scheme_fthm(nd=3)

    counter = 0
    for run_idx in range(IDX_START, IDX_START + NUM_RUNS):
        counter += 1
        if counter <= 3:
            continue

        print("Starting run", run_idx)

        with open(f"stats/thm_solver_space_scheme_run_{run_idx}.pkl", "wb") as f:
            pickle.dump(solver_space_scheme, f)
        with open(f"stats/thm_permutations_{run_idx}.pkl", "wb") as f:
            pickle.dump(
                {
                    "inlet": permutations_inlet[run_idx - IDX_START],
                    "outlet": permutations_outlet[run_idx - IDX_START],
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

        if RANDOM_SELECTION:
            performance_predictor = PerformancePredictorRandom(num_solvers=num_solvers)
        else:
            performance_predictor = Estimator(num_solvers=num_solvers)

        solver_selector = SolverSelector(
            solver_space=solver_space,
            performance_predictor=performance_predictor,
        )

        params["setup"]["linear_solver_selector"] = solver_selector

        for inlet_placement in inlet_placements[
            permutations_inlet[run_idx - IDX_START]
        ]:
            for outlet_placement in outlet_placements[
                permutations_outlet[run_idx - IDX_START]
            ]:
                params["inlet_placement"] = inlet_placement
                params["outlet_placement"] = outlet_placement
                sim_name = f"run_{run_idx}_{simulation_name(params)}"
                if RANDOM_SELECTION:
                    sim_name = f"RANDOM_{sim_name}"
                params["folder_name"] = sim_name

                try:
                    model = ModelTHMWithSelector(params)
                    model.prepare_simulation()

                    print("Initialising")
                    initialize(model, params)
                    print("Running")
                    run(model, params)
                except Exception as e:
                    traceback.print_exc()

                solver_selector.history.save(
                    f"./stats/solver_selection_history_{sim_name}.npy"
                )
                print("Done")
