from itertools import count
import numpy as np
import traceback
from solver_selection_thm.spe_physics import (
    SPE10Model,
    run,
    params,
    Z_SLICES,
    X_SLICES,
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
    SolverSelectionMixinTH,
)
from stats import StatisticsSavingMixin


class ModelTHMWithSelector(StatisticsSavingMixin, SolverSelectionMixinTH, SPE10Model):
    def data_to_export(self):
        data = super().data_to_export()
        sds = self.mdg.subdomains()
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        q = self.evaluate_and_scale(sds, "porosity", "m^3")
        for id, sd in enumerate(sds):
            data.append(
                (
                    sd,
                    "permeability",
                    q[cell_offsets[id] : cell_offsets[id + 1]],
                )
            )
        return data


def make_solver_space_scheme_hm(nd: int):
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
                {
                    "python_pc_type": "sor",
                },
                {
                    "python_pc_type": "pbjacobi",
                },
                {
                    "python_pc_type": "hypre",
                    "python_pc_hypre_type": "boomeramg",
                    "python_pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                        [0.5, 0.7, 0.9]
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
                        [0.5, 0.7, 0.9]
                    ),
                    "pc_hypre_boomeramg_agg_nl": NumericalChoices([0, 1, 2]),
                    "pc_hypre_boomeramg_relax_type_all": CategoricalChoices(
                        [
                            "symmetric-SOR/Jacobi",
                            "l1scaled-Jacobi",
                            "SOR/Jacobi",
                            "Jacobi",
                        ]
                    ),
                },
                "complement": {
                    "block_type": "PetscFieldSplitScheme",
                    "groups": temp,
                    "elim_options": {
                        "pc_type": "none",
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
            "ksp_gmres_restart": NumericalChoices([10, 30, 50]),
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
    solver_space = SolverSpace(
        solver_space_scheme=make_solver_space_scheme_hm(nd=3),
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

    np.random.seed(42)

    for run_idx in range(5):
        np.random.shuffle(Z_SLICES)
        np.random.shuffle(X_SLICES)

        for z_slice in Z_SLICES:
            for x_slice in X_SLICES:
                params["x_slice"] = x_slice
                params["z_slice"] = z_slice
                sim_name = f'run_{run_idx}_{simulation_name(params)}'
                params["folder_name"] = sim_name
                model = ModelTHMWithSelector(params)
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
