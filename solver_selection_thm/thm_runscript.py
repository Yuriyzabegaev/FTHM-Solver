from itertools import count
import traceback
from solver_selection_thm.physics import (
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
    return {
        "block_type": "LinearTransformedScheme",
        "scale_energy_balance": True,
        "Qright": True,
        "inner": {
            "block_type": "PetscKSPScheme",
            "petsc_options": {
                "ksp_monitor": None,
                "ksp_rtol": 1e-12,
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
                                    "pc_type": "hmg",
                                    "hmg_inner_pc_type": "hypre",
                                    "hmg_inner_pc_hypre_type": "boomeramg",
                                    "hmg_inner_pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                                        [0.5, 0.6, 0.7, 0.8]
                                        # [0.5]
                                    ),
                                    "mg_levels_ksp_type": CategoricalChoices(
                                        [
                                            "chebyshev",
                                            "richardson",
                                        ]
                                    ),
                                    "mg_levels_ksp_max_it": NumericalChoices(
                                        [1, 2, 4, 8]
                                        # [1]
                                    ),
                                    "mg_levels_pc_type": CategoricalChoices(
                                        [
                                            "ilu",
                                            # "sor",  # very bad
                                        ]
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
                                    {
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
                                            "python_pc_type": "hypre",
                                            "python_pc_hypre_type": "boomeramg",
                                            "python_pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                                                # [0.5, 0.6, 0.7, 0.8]
                                                [0.5]
                                            ),
                                            "python_pc_hypre_boomeramg_P_max": 16,
                                        },
                                    },
                                    # {
                                    #     "block_type": "PetscCompositeScheme",
                                    #     "groups": flow + temp,
                                    #     "solvers": {
                                    #         0: {
                                    #             "block_type": "PetscFieldSplitScheme",
                                    #             "groups": flow,
                                    #             "fieldsplit_options": {
                                    #                 "pc_fieldsplit_type": "additive",
                                    #             },
                                    #             "elim_options": {
                                    #                 "pc_type": "hypre",
                                    #                 "pc_hypre_type": "boomeramg",
                                    #                 "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                                    #                     # [0.5, 0.6, 0.7, 0.8]
                                    #                     [0.5]
                                    #                 ),
                                    #             },
                                    #             "complement": {
                                    #                 "block_type": "PetscFieldSplitScheme",
                                    #                 "groups": temp,
                                    #                 "elim_options": {
                                    #                     "pc_type": "none",
                                    #                 },
                                    #             },
                                    #         },
                                    #         1: {
                                    #             "block_type": "PetscFieldSplitScheme",
                                    #             "groups": flow + temp,
                                    #             "python_pc": {
                                    #                 "block_type": "PcPythonPermutation",
                                    #                 "permutation_type": "pt_permutation",
                                    #                 "p_groups": flow,
                                    #                 "t_groups": temp,
                                    #                 "block_size": 2,
                                    #             },
                                    #             "elim_options": {
                                    #                 "python_pc_type": "ilu",
                                    #             },
                                    #         },
                                    #     },
                                    # },
                                ]
                            ),
                        },
                    },
                },
            },
        },
    }


if __name__ == "__main__":
    solver_space = SolverSpace(
        solver_space_scheme=make_solver_space_scheme_fthm(nd=3),
        solver_scheme_builders=KNOWN_SOLVER_COMPONENTS_THM,
    )
    num_solvers = solver_space.all_decisions_encoding.shape[0]
    performance_predictor = PerformancePredictorPassiveAgressive(
        num_solvers=num_solvers,
    )
    solver_selector = SolverSelector(
        reward_estimator=RewardEstimator(),
        solver_space=solver_space,
        performance_predictor=performance_predictor,
    )
    print(solver_space.decision_tree)

    params["setup"]["linear_solver_selector"] = solver_selector

    counter = count()
    for inlet_placement in inlet_placements:
        for outlet_placement in outlet_placements:
            i = next(counter)
            params["inlet_placement"] = inlet_placement
            params["outlet_placement"] = outlet_placement
            params["folder_name"] = f"{i}_{simulation_name(params)}"

            try:
                model = ModelTHMWithSelector(params)
                model.prepare_simulation()

                # try:
                print("Initialising")
                initialize(model, params)
                print("Running")
                run(model, params)
            except Exception as e:
                traceback.print_exc()
