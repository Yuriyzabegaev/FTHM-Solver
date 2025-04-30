import pytest
import numpy as np
from solver_selection_thm.performance_predictor import (
    PerformancePredictorEpsGreedy,
    RewardEstimator,
)
from solver_selection_thm.selector import SolverSelector
from solver_selection_thm.solver_space import (
    CategoricalChoices,
    NumericalChoices,
    SolverSpace,
)


nd = 2
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
                                    "hmg_inner_pc_hypre_boomeramg_strong_threshold": 0.7,
                                    "mg_levels_ksp_type": "richardson",
                                    "mg_levels_ksp_max_it": 2,
                                    "mg_levels_pc_type": "sor",
                                },
                                {
                                    "pc_type": "gamg",
                                    "pc_gamg_threshold": NumericalChoices(
                                        [0.01, 0.001, 0.0001]
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
                                                    "pc_hypre_boomeramg_strong_threshold": 0.7,
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
                                                    "python_pc_type": "ilu",
                                                },
                                            },
                                        },
                                    },
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
                                            "python_pc_hypre_boomeramg_strong_threshold": 0.7,
                                            "python_pc_hypre_boomeramg_P_max": 16,
                                        },
                                    },
                                ]
                            ),
                        },
                    },
                },
            },
        },
    }


@pytest.fixture
def porepy_model():
    from porepy.models.thermoporomechanics import Thermoporomechanics
    from thermal.thm_solver import THMSolver

    class Model(THMSolver, Thermoporomechanics):
        pass

    model = Model()
    model.prepare_simulation()
    return model


@pytest.mark.parametrize(
    "params",
    [
        # No configurable options
        {
            "solver_space_scheme": {
                "block_type": "PetscFieldSplitScheme",
                "groups": mech,
                "elim_options": {"pc_type": "hypre"},
            },
            "num_possible_solvers": 1,
            "expected_printing": "block_type: PetscFieldSplitScheme",
            "expected_config": {
                "block_type": "PetscFieldSplitScheme",
                "groups": mech,
                "elim_options": {"pc_type": "hypre"},
            },
        },
        # CategoricalChoices with not dict inside
        dict(
            solver_space_scheme={
                "block_type": "PetscFieldSplitScheme",
                "a": CategoricalChoices(["b", "c"]),
            },
            num_possible_solvers=2,
            expected_printing=(
                "block_type: PetscFieldSplitScheme\n"
                "| a (fork with 2 branches):\n"
                "| | b\n"
                "| | c"
            ),
            expected_config={
                "block_type": "PetscFieldSplitScheme",
                "a": "c",
            },
        ),
        # Some nontrivial combination of numerical and categorical
        dict(
            solver_space_scheme={
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
                            ),
                            "mg_levels_ksp_type": CategoricalChoices(
                                ["chebyshev", "richardson"]
                            ),
                            "mg_levels_ksp_max_it": NumericalChoices([1, 2, 4, 8]),
                            "mg_levels_pc_type": CategoricalChoices(["ilu", "sor"]),
                        },
                    ]
                ),
            },
            num_possible_solvers=64,
            expected_printing=(
                "block_type: PetscFieldSplitScheme\n"
                "| elim_options (fork with 1 branches):\n"
                "| | pc_type: hmg\n"
                "| | | hmg_inner_pc_hypre_boomeramg_strong_threshold: Choices from 0.5 to 0.8, len = 4\n"
                "| | | mg_levels_ksp_max_it: Choices from 1 to 8, len = 4\n"
                "| | | mg_levels_ksp_type (fork with 2 branches):\n"
                "| | | | chebyshev\n"
                "| | | | richardson\n"
                "| | | mg_levels_pc_type (fork with 2 branches):\n"
                "| | | | ilu\n"
                "| | | | sor"
            ),
            expected_config={
                "block_type": "PetscFieldSplitScheme",
                "groups": mech,
                "elim_options": {
                    "pc_type": "hmg",
                    "hmg_inner_pc_type": "hypre",
                    "hmg_inner_pc_hypre_type": "boomeramg",
                    "hmg_inner_pc_hypre_boomeramg_strong_threshold": 0.8,
                    "mg_levels_ksp_type": "richardson",
                    "mg_levels_ksp_max_it": 8,
                    "mg_levels_pc_type": "sor",
                },
            },
        ),
        # CategoricalChoices with inner block_type
        {
            "solver_space_scheme": {
                "block_type": "PetscFieldSplitScheme",
                "a": CategoricalChoices(
                    [
                        {"block_type": "PetscFieldSplitScheme", "value": "b"},
                        {"block_type": "PetscFieldSplitScheme", "value": "c"},
                    ]
                ),
            },
            "num_possible_solvers": 2,
            "expected_printing": (
                "block_type: PetscFieldSplitScheme\n"
                "| a (fork with 2 branches):\n"
                "| | block_type: PetscFieldSplitScheme\n"
                "| | block_type: PetscFieldSplitScheme"
            ),
            "expected_config": {
                "block_type": "PetscFieldSplitScheme",
                "a": {"block_type": "PetscFieldSplitScheme", "value": "c"},
            },
        },
        # CategoricalChoices with no inner block_type
        {
            "solver_space_scheme": {
                "block_type": "PetscFieldSplitScheme",
                "a": CategoricalChoices(
                    [
                        {"value": "b"},
                        {"value": "c"},
                    ]
                ),
            },
            "num_possible_solvers": 2,
            "expected_printing": (
                "block_type: PetscFieldSplitScheme\n"
                "| a (fork with 2 branches):\n"
                "| | value: b\n"
                "| | value: c"
            ),
            "expected_config": {
                "block_type": "PetscFieldSplitScheme",
                "a": {"value": "c"},
            },
        },
        # CategoricalChoices in outer block_type with inner block_type
        {
            "solver_space_scheme": {
                "block_type": CategoricalChoices(
                    [
                        {"block_type": "PetscFieldSplitScheme", "value": "b"},
                        {"block_type": "PetscCompositeScheme", "value": "c"},
                    ]
                ),
            },
            "num_possible_solvers": 2,
            "expected_printing": (
                "block_type: CategoricalChoices\n"
                "| block_type (fork with 2 branches):\n"
                "| | block_type: PetscFieldSplitScheme\n"
                "| | block_type: PetscCompositeScheme"
            ),
            "expected_config": {"block_type": "PetscCompositeScheme", "value": "c"},
        },
        # CategoricalChoices in outer block_type with no inner block type
        {
            "solver_space_scheme": {
                "block_type": CategoricalChoices(
                    [
                        {"value": "b"},
                        {"value": "c"},
                    ]
                ),
            },
            "num_possible_solvers": 2,
            "expected_printing": (
                "block_type: CategoricalChoices\n"
                "| block_type (fork with 2 branches):\n"
                "| | value: b\n"
                "| | value: c"
            ),
            "expected_config": {"value": "c"},
        },
        # Two numerical parameters within a categorical choice
        dict(
            solver_space_scheme={
                "block_type": "PetscFieldSplitScheme",
                "elim_options": CategoricalChoices(
                    [
                        {
                            "pc_type": "hypre",
                            "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                                [0.05, 0.005, 0.0005]
                            ),
                        },
                        {
                            "pc_type": "gamg",
                            "pc_gamg_threshold": NumericalChoices(
                                [0.01, 0.001, 0.0001]
                            ),
                        },
                    ]
                ),
            },
            num_possible_solvers=6,
            expected_printing=(
                "block_type: PetscFieldSplitScheme\n"
                "| elim_options (fork with 2 branches):\n"
                "| | pc_type: hypre\n"
                "| | | pc_hypre_boomeramg_strong_threshold: Choices from 0.0005 to 0.05, len = 3\n"
                "| | pc_type: gamg\n"
                "| | | pc_gamg_threshold: Choices from 0.0001 to 0.01, len = 3"
            ),
            expected_config={
                "block_type": "PetscFieldSplitScheme",
                "elim_options": {"pc_type": "gamg", "pc_gamg_threshold": 0.0001},
            },
        ),
        # CPR or SAMG
        dict(
            solver_space_scheme={
                "block_type": CategoricalChoices(
                    [
                        {
                            "block_type": "PetscCompositeScheme",
                            "groups": [flow, temp],
                            "solvers": {
                                0: {
                                    "block_type": "PetscFieldSplitScheme",
                                    "fieldsplit_options": {
                                        "pc_fieldsplit_type": "additive",
                                    },
                                    "elim_options": {
                                        "pc_type": "hypre",
                                        "pc_hypre_type": "boomeramg",
                                        "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                                            [0.5, 0.6, 0.7, 0.8]
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
                                    "groups": [flow, temp],
                                    "python_pc": {
                                        "block_type": "PcPythonPermutation",
                                        "permutation_type": "pt_permutation",
                                        "p_groups": flow,
                                        "t_groups": temp,
                                        "block_size": NumericalChoices([1, 2, 4, 8]),
                                    },
                                    "elim_options": {
                                        "python_pc_type": "ilu",
                                    },
                                },
                            },
                        },
                        {
                            "block_type": "PetscFieldSplitScheme",
                            "groups": [flow, temp],
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
                                    [0.5, 0.6, 0.7, 0.8]
                                ),
                                "python_pc_hypre_boomeramg_P_max": 16,
                            },
                        },
                    ]
                ),
            },
            num_possible_solvers=20,
            expected_printing=(
                "block_type: CategoricalChoices\n"
                "| block_type (fork with 2 branches):\n"
                "| | block_type: PetscCompositeScheme\n"
                "| | | pc_hypre_boomeramg_strong_threshold: Choices from 0.5 to 0.8, len = 4\n"
                "| | | block_size: Choices from 1 to 8, len = 4\n"
                "| | block_type: PetscFieldSplitScheme\n"
                "| | | python_pc_hypre_boomeramg_strong_threshold: Choices from 0.5 to 0.8, len = 4"
            ),
            expected_config={
                "block_type": "PetscFieldSplitScheme",
                "groups": [flow, temp],
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
                    "python_pc_hypre_boomeramg_strong_threshold": 0.8,
                    "python_pc_hypre_boomeramg_P_max": 16,
                },
            },
        ),
        # CPR with subsolver options
        dict(
            solver_space_scheme={
                "block_type": "PetscCompositeScheme",
                "groups": [flow, temp],
                "solvers": {
                    0: CategoricalChoices(
                        [
                            {"block_type": "PetscFieldSplitScheme", "choice": "a"},
                            {"block_type": "PetscFieldSplitScheme", "choice": "b"},
                            {"block_type": "PetscFieldSplitScheme", "choice": "c"},
                        ],
                    ),
                    1: CategoricalChoices(
                        [
                            {"block_type": "PetscFieldSplitScheme", "choice": "d"},
                            {"block_type": "PetscFieldSplitScheme", "choice": "e"},
                        ],
                    ),
                },
            },
            num_possible_solvers=6,
            expected_printing=(
                "block_type: PetscCompositeScheme\n"
                "| 0 (fork with 3 branches):\n"
                "| | block_type: PetscFieldSplitScheme\n"
                "| | block_type: PetscFieldSplitScheme\n"
                "| | block_type: PetscFieldSplitScheme\n"
                "| 1 (fork with 2 branches):\n"
                "| | block_type: PetscFieldSplitScheme\n"
                "| | block_type: PetscFieldSplitScheme"
            ),
            expected_config={
                "block_type": "PetscCompositeScheme",
                "groups": [flow, temp],
                "solvers": {
                    0: {"block_type": "PetscFieldSplitScheme", "choice": "c"},
                    1: {"block_type": "PetscFieldSplitScheme", "choice": "e"},
                },
            },
        ),
        # Same parameter names for two independent parameters
        dict(
            solver_space_scheme={
                "block_type": "PetscFieldSplitScheme",
                "groups": mech,
                "elim_options": {
                    "pc_type": "hypre",
                    "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                        [0.09, 0.009, 0.0009]
                    ),
                },
                "complement": {
                    "block_type": "PetscFieldSplitScheme",
                    "groups": flow,
                    "elim_options": {
                        "pc_type": "hypre",
                        "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                            [0.01, 0.001, 0.0001, 0.00005]
                        ),
                    },
                },
            },
            num_possible_solvers=12,
            expected_printing=(
                "block_type: PetscFieldSplitScheme\n"
                "| pc_hypre_boomeramg_strong_threshold: Choices from 0.0009 to 0.09, len = 3\n"
                "| pc_hypre_boomeramg_strong_threshold: Choices from 5e-05 to 0.01, len = 4"
            ),
            expected_config={
                "block_type": "PetscFieldSplitScheme",
                "groups": mech,
                "elim_options": {
                    "pc_type": "hypre",
                    "pc_hypre_boomeramg_strong_threshold": 0.0009,
                },
                "complement": {
                    "block_type": "PetscFieldSplitScheme",
                    "groups": flow,
                    "elim_options": {
                        "pc_type": "hypre",
                        "pc_hypre_boomeramg_strong_threshold": 0.00005,
                    },
                },
            },
        ),
        # Big boy (full FTHM)
        dict(
            solver_space_scheme=make_solver_space_scheme_fthm(nd=2),
            num_possible_solvers=8,
            expected_printing=(
                "block_type: LinearTransformedScheme\n"
                "| elim_options (fork with 2 branches):\n"
                "| | pc_type: hmg\n"
                "| | pc_type: gamg\n"
                "| | | pc_gamg_threshold: Choices from 0.0001 to 0.01, len = 3\n"
                "| block_type (fork with 2 branches):\n"
                "| | block_type: PetscCompositeScheme\n"
                "| | block_type: PetscFieldSplitScheme"
            ),
            expected_config={
                "block_type": "LinearTransformedScheme",
                "Qright": True,
                "scale_energy_balance": True,
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
                                "elim_options": {
                                    "pc_type": "gamg",
                                    "pc_gamg_threshold": 0.0001,
                                },
                                "block_size": nd,
                                "invert": {
                                    "block_type": "fs_analytical_slow_new",
                                    "p_mat_group": 5,
                                    "p_frac_group": 6,
                                    "groups": flow + temp,
                                },
                                "complement": {
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
                                        "python_pc_hypre_boomeramg_strong_threshold": 0.7,
                                        "python_pc_hypre_boomeramg_P_max": 16,
                                    },
                                },
                            },
                        },
                    },
                },
            },
        ),
    ],
)
def test_solver_space(params):
    solver_space_scheme: dict = params["solver_space_scheme"]
    num_possible_solvers: int = params["num_possible_solvers"]
    expected_printing: str = params["expected_printing"]
    expected_config: dict = params["expected_config"]

    solver_space = SolverSpace(solver_space_scheme, solver_scheme_builders={})
    assert str(solver_space.decision_tree) == expected_printing

    assert solver_space.all_decisions_encoding.shape[0] == num_possible_solvers

    decision_taken = solver_space.all_decisions_encoding[-1]
    config = solver_space.config_from_decision(decision_taken)
    assert config == expected_config


class MockSolverSchemeBuilder:
    def build_solver_scheme_from_config(
        self,
        config: dict,
        build_inner_solver_scheme: callable,
        porepy_model,
    ):
        return "my_scheme"


def test_solver_selector():
    np.random.seed(42)

    solver_space = SolverSpace(
        solver_space_scheme=make_solver_space_scheme_fthm(nd=2),
        solver_scheme_builders={"LinearTransformedScheme": MockSolverSchemeBuilder()},
    )
    num_solvers = solver_space.all_decisions_encoding.shape[0]
    performance_predictor = PerformancePredictorEpsGreedy(
        num_solvers=num_solvers,
        exploration=0.5,
        exploration_decrease_rate=0.9,
    )
    solver_selector = SolverSelector(
        reward_estimator=RewardEstimator(),
        solver_space=solver_space,
        performance_predictor=performance_predictor,
    )
    solver_in_use_idx: None | int = None

    for time_step in range(100):
        characteristics = np.array([5, 6, 7.0]) + (time_step % 4)
        scheme, choice = solver_selector.select_linear_solver_scheme(
            characteristics=characteristics,
            porepy_model=None,
            active_solver_idx=solver_in_use_idx,
        )
        assert scheme == "my_scheme"

        solve_time = 0.5 + 0.5 * np.cos(choice / (num_solvers - 1) * 2 * np.pi)
        if solver_in_use_idx == choice:
            construct_time = 1e-2
        else:
            construct_time = 1e-2 + 0.5 * np.sin(choice / (num_solvers - 1) * np.pi)
        solver_in_use_idx = choice
        success = choice != 4

        solver_selector.provide_performance_feedback(
            solve_time=solve_time, construct_time=construct_time, success=success
        )

    # assert np.allclose(np.sum(rewards_history), 115.617)
    # assert np.allclose(np.sum(expectation_list), 3421.107)
    assert np.sum(solver_selector.history.greedy) == 67
    assert np.median(solver_selector.history.decision_idx) == 3


@pytest.mark.parametrize("i", range(8))
def test_scheme_builders(i: int, porepy_model):
    from pp_binding import KNOWN_SOLVER_COMPONENTS_THM

    porepy_model.before_nonlinear_loop()
    porepy_model.before_nonlinear_iteration()
    porepy_model.assemble_linear_system()

    solver_space = SolverSpace(
        solver_space_scheme=make_solver_space_scheme_fthm(nd=2),
        solver_scheme_builders=KNOWN_SOLVER_COMPONENTS_THM,
    )
    decision = solver_space.all_decisions_encoding[i]
    scheme = solver_space.build_solver_scheme(
        solver_space.config_from_decision(decision=decision), porepy_model=porepy_model
    )
    scheme.make_solver(porepy_model.bmat)
