import pytest

from solver_selection_thm.solver_space import CategoricalChoices, NumericalChoices, SolverSpace


nd = 3
contact = [0]
intf = [1, 2]
mech = [3, 4]
flow = [5, 6, 7]
temp = [8, 9, 10]


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
            solver_space_scheme={
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
            num_possible_solvers=8,
            expected_printing=(
                "block_type: PetscKSPScheme\n"
                "| elim_options (fork with 2 branches):\n"
                "| | pc_type: hmg\n"
                "| | pc_type: gamg\n"
                "| | | pc_gamg_threshold: Choices from 0.0001 to 0.01, len = 3\n"
                "| block_type (fork with 2 branches):\n"
                "| | block_type: PetscCompositeScheme\n"
                "| | block_type: PetscFieldSplitScheme"
            ),
            expected_config={
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
