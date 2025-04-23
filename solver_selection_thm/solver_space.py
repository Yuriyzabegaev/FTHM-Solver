from itertools import count, product
from typing import Optional
import numpy as np

from thermal.thm_solver import make_pt_permutation

nd = 3
contact = [0]
intf = [1, 2]
mech = [3, 4]
flow = [5, 6, 7]
temp = [8, 9, 10]


class CategoricalChoices:
    def __init__(self, choices: list[dict]):
        self.choices = choices

    def __repr__(self):
        return f"CategoricalChoices({self.choices})"


class NumericalChoices:
    def __init__(self, choices: np.ndarray):
        self.choices: np.ndarray = np.array(choices)

    def __repr__(self):
        return f"NumericalChoices({self.choices})"

    def __str__(self) -> str:
        return f"Choices from {min(self.choices)} to {max(self.choices)}, len = {len(self.choices)}"


solver_space_scheme = {
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
                                np.array([0.01, 0.001, 0.0001])
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
}

# solver_space_scheme = {
#     "block_type": CategoricalChoices(
#         [
#             {
#                 "block_type": "PetscFieldSplitScheme",
#                 "groups": mech + flow,
#                 "elim_options": {"pc_type": "lu"},
#             },
#             {
#                 "block_type": "PetscFieldSplitScheme",
#                 "groups": mech,
#                 "elim_options": CategoricalChoices(
#                     [
#                         {
#                             "pc_type": "hmg",
#                         },
#                         {
#                             "pc_type": "gamg",
#                             "pc_gamg_threshold": NumericalChoices(
#                                 np.array([0.09, 0.009, 0.0009])
#                             ),
#                         },
#                     ]
#                 ),
#                 "complement": {
#                     "block_type": "PetscFieldSplitScheme",
#                     "groups": flow,
#                     "elim_options": CategoricalChoices(
#                         [
#                             {
#                                 "pc_type": "hypre",
#                                 "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
#                                     np.array([0.01, 0.001, 0.0001, 0.00005])
#                                 ),
#                             },
#                             {
#                                 "pc_type": "gamg",
#                                 "pc_gamg_threshold": NumericalChoices(
#                                     np.array([0.01, 0.001, 0.0001])
#                                 ),
#                             },
#                         ]
#                     ),
#                 },
#             },
#         ],
#     )
# }

# solver_space_scheme = {
#     "block_type": "PetscFieldSplitScheme",
#     "groups": mech,
#     "elim_options": {
#         "pc_type": "gamg",
#         "pc_gamg_threshold": NumericalChoices(np.array([0.09, 0.009, 0.0009])),
#     },
#     "complement": {
#         "block_type": "PetscFieldSplitScheme",
#         "groups": flow,
#         "elim_options": {
#             "pc_type": "hypre",
#             "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
#                 np.array([0.01, 0.001, 0.0001, 0.00005])
#             ),
#         },
#     },
# }

# solver_space_scheme = {
#     "block_type": CategoricalChoices(
#         [
#             {
#                 "block_type": "PetscCompositeScheme",
#                 "groups": flow + temp,
#                 "solvers": [
#                     {
#                         "block_type": "PetscFieldSplitScheme",
#                         "fieldsplit_options": {
#                             "pc_fieldsplit_type": "additive",
#                         },
#                         "elim_options": {
#                             "pc_type": "hypre",
#                             "pc_hypre_type": "boomeramg",
#                             "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
#                                 np.array([0.09, 0.009, 0.0009])
#                             ),
#                         },
#                         "complement": {
#                             "block_type": "PetscFieldSplitScheme",
#                             "groups": temp,
#                             "elim_options": {
#                                 "pc_type": "none",
#                             },
#                         },
#                     },
#                     {
#                         "block_type": "PetscFieldSplitScheme",
#                         "groups": flow + temp,
#                         "python_pc": {
#                             "block_type": "PcPythonPermutation",
#                             "permutation_type": "pt_permutation",
#                             "p_groups": flow,
#                             "t_groups": temp,
#                             "block_size": NumericalChoices(np.array([1, 2, 4])),
#                         },
#                         "elim_options": {
#                             "python_pc_type": "ilu",
#                         },
#                     },
#                 ],
#             },
#             {
#                 "block_type": "PetscFieldSplitScheme",
#                 "groups": flow + temp,
#                 "python_pc": {
#                     "block_type": "PcPythonPermutation",
#                     "permutation_type": "pt_permutation",
#                     "p_groups": flow,
#                     "t_groups": temp,
#                     "block_size": 2,
#                 },
#                 "elim_options": {
#                     "python_pc_type": "hypre",
#                     "python_pc_hypre_type": "boomeramg",
#                     "python_pc_hypre_boomeramg_strong_threshold": 0.7,
#                     "python_pc_hypre_boomeramg_P_max": 16,
#                 },
#             },
#         ]
#     ),
# }


class FlatSolverDecision:
    def __init__(
        self,
        categorical: Optional[set["DecisionNode"]] = None,
        numerical: Optional[set[NumericalChoices]] = None,
    ):
        self.categorical: set[DecisionNode] = categorical or set()
        self.numerical: set[NumericalChoices] = numerical or set()

    def __repr__(self):
        return f"FlatSolverConfig({self.categorical}, {self.numerical})"


class DecisionNode:
    def __init__(self, solver_space_scheme: dict):
        self.solver_space_scheme: dict = solver_space_scheme
        self.children: list[ForkNode] = []
        self.numerical_choices: dict[str, NumericalChoices] = {}

    def _str(self, prefix="") -> str:
        k, v = next(iter(self.solver_space_scheme.items()))
        if isinstance(v, CategoricalChoices):
            v = 'CategoricalChoices'
        repr = f"{prefix}{k}: {v}"
        child_prefix = f"{prefix}| "
        if len(self.numerical_choices) > 0:
            numerical_repr = [
                f"{child_prefix}{k}: {v}" for k, v in self.numerical_choices.items()
            ]
            tmp = "\n".join(numerical_repr)
            repr = f"{repr}\n{tmp}"
        if len(self.children) > 0:
            child_repr = [child._str(prefix=child_prefix) for child in self.children]
            tmp = "\n".join(child_repr)
            repr = f"{repr}\n{tmp}"
        return repr

    def __repr__(self) -> str:
        k, v = next(iter(self.solver_space_scheme.items()))
        return f"DecisionNode({k}, {v})"

    def __str__(self) -> str:
        return self._str()

    def list_possible_solvers(self) -> list[FlatSolverDecision]:
        if len(self.children) == 0:
            flat_config = FlatSolverDecision()
            for numerical_choice in self.numerical_choices.values():
                flat_config.numerical.add(numerical_choice)
            return [flat_config]

        children_solver_spaces = [c.encode_solver_space() for c in self.children]

        merged_results = []
        for tuple_of_decisions in list(product(*children_solver_spaces)):
            cat = set(x for conf in tuple_of_decisions for x in conf.categorical)
            num = set(x for conf in tuple_of_decisions for x in conf.numerical)
            merged_results.append(FlatSolverDecision(categorical=cat, numerical=num))
        return merged_results


class ForkNode:
    def __init__(self, categorical_choices: CategoricalChoices, options_key: str):
        self.options_key: str = options_key
        self.categorical_choices: CategoricalChoices = categorical_choices
        self.children: list[DecisionNode] = []

    def __repr__(self):
        return f"ForkNode({self.options_key})"

    def __str__(self):
        return self._str()

    def _str(self, prefix="") -> str:
        num = len(self.categorical_choices.choices)
        repr = f"{prefix}{self.options_key} (fork with {num} branches):"
        child_prefix = f"{prefix}| "
        if len(self.children) > 0:
            child_repr = [child._str(prefix=child_prefix) for child in self.children]
            tmp = "\n".join(child_repr)
            repr = f"{repr}\n{tmp}"
        return repr

    def encode_solver_space(self) -> list[FlatSolverDecision]:
        if len(self.children) == 0:
            assert False, "Why Fork node with no options?"

        solver_space = []
        for child in self.children:
            child_solver_space = child.list_possible_solvers()
            for solver in child_solver_space:
                solver.categorical.add(child)
            solver_space.extend(child_solver_space)
        return solver_space


class SolverSpace:
    def __init__(self, solver_space_scheme: dict):
        self.solver_space_scheme: dict = solver_space_scheme
        self.decision_tree = DecisionNode(solver_space_scheme)
        build_solver_space(solver_space_scheme, self.decision_tree, options_key="")


class SolverComponentBuilder:
    @staticmethod
    def build_solver_scheme_from_config(config: dict, build_inner_solver: callable):
        pass


from full_petsc_solver import (
    PetscKSPScheme,
    PetscFieldSplitScheme,
    PetscCompositeScheme,
    PcPythonPermutation,
)
from fixed_stress import make_fs_analytical_slow_new


class PetscKSPSchemeBuilder(SolverComponentBuilder):
    @staticmethod
    def build_solver_scheme_from_config(config: dict, build_inner_solver: callable):
        config = config.copy()
        del config["block_type"]
        pc_config = config.pop("preconditioner", None)
        pc = build_inner_solver(pc_config) if pc_config is not None else None
        return PetscKSPScheme(preconditioner=pc, **config)


class PetscFieldSplitSchemeBuilder(SolverComponentBuilder):
    @staticmethod
    def build_solver_scheme_from_config(config: dict, build_inner_solver: callable):
        config = config.copy()
        del config["block_type"]
        complement_config = config.pop("complement", None)
        complement = (
            build_inner_solver(complement_config)
            if complement_config is not None
            else None
        )
        return PetscFieldSplitScheme(complement=complement, **config)


class fs_analytical_slow_new_Builder(SolverComponentBuilder):
    @staticmethod
    def build_solver_scheme_from_config(config: dict, build_inner_solver: callable):
        config = config.copy()
        del config["block_type"]
        return lambda bmat: make_fs_analytical_slow_new(**config)


class PetscCompositeSchemeBuilder(SolverComponentBuilder):
    @staticmethod
    def build_solver_scheme_from_config(config: dict, build_inner_solver: callable):
        config = config.copy()
        del config["block_type"]
        return PetscCompositeScheme(
            solvers=[
                build_inner_solver(
                    config["solvers"][i] for i in range(len(config["solvers"]))
                )
            ],
            **config,
        )


class PcPythonPermutationBuilder(SolverComponentBuilder):
    @staticmethod
    def build_solver_scheme_from_config(config: dict, build_inner_solver: callable):
        config = config.copy()
        del config["block_type"]
        assert config["permutation_type"] == "pt_permutation"

        return lambda bmat: PcPythonPermutation(
            make_pt_permutation(
                bmat, p_groups=config["p_groups"], t_groups=config["t_groupd"]
            ),
            block_size=config["block_type"],
        )


KNOWN_SOLVER_COMPONENTS: dict[str, SolverComponentBuilder] = {
    "PetscKSPScheme": PetscKSPSchemeBuilder(),
    "PetscFieldSplitScheme": PetscFieldSplitSchemeBuilder(),
    "fs_analytical_slow_new": fs_analytical_slow_new_Builder(),
    "PetscCompositeScheme": PetscCompositeSchemeBuilder(),
    "PcPythonPermutation": PcPythonPermutationBuilder(),
}


def build_solver_space(
    solver_space_scheme, current_decision_node: DecisionNode, options_key: str
):
    if isinstance(solver_space_scheme, dict):
        for key, value in solver_space_scheme.items():
            build_solver_space(
                solver_space_scheme=value,
                current_decision_node=current_decision_node,
                options_key=key,
            )
    elif isinstance(solver_space_scheme, CategoricalChoices):
        fork_node = ForkNode(solver_space_scheme, options_key=options_key)
        current_decision_node.children.append(fork_node)
        for decision in solver_space_scheme.choices:
            new_decision_node = DecisionNode(decision)
            fork_node.children.append(new_decision_node)
            build_solver_space(
                solver_space_scheme=decision,
                current_decision_node=new_decision_node,
                options_key=options_key,
            )
    elif isinstance(solver_space_scheme, NumericalChoices):
        current_decision_node.numerical_choices[options_key] = solver_space_scheme


solver_space = SolverSpace(solver_space_scheme)
print(solver_space.decision_tree)
x = solver_space.decision_tree.list_possible_solvers()


print(x)
print(len(x))


def make_all_possible_decisions(solver_space: list[FlatSolverDecision]):
    category_choices_map: dict[int, int] = dict()
    numerical_choices_map: dict[int, int] = dict()
    # all_category_choices: dict[int, DecisionNode] = dict()
    # all_numerical_choices: dict[int, dict[str, NumericalChoices]] = dict()
    category_choices_counter = count()
    numerical_choices_counter = count()

    for solver in solver_space:
        for choice in solver.categorical:
            choice_id = id(choice)
            if category_choices_map.get(choice_id) is not None:
                continue
            # all_category_choices[choice_id] = choice
            category_idx = next(category_choices_counter)
            category_choices_map[choice_id] = category_idx
        for numerical_choice in solver.numerical:
            choice_id = id(numerical_choice)
            if numerical_choices_map.get(choice_id) is not None:
                continue

            numerical_idx = next(numerical_choices_counter)
            numerical_choices_map[choice_id] = numerical_idx
            # all_numerical_choices[numerical_idx] = numerical_choice

    num_category_choices = next(category_choices_counter)
    num_numerical_choices = next(numerical_choices_counter)

    all_possible_decisions = []
    for solver in solver_space:
        categorical_decision = []
        numerical_decision: dict[int, NumericalChoices] = {}
        for choice in solver.categorical:
            categorical_decision.append(category_choices_map[id(choice)])
        for numerical_choice in solver.numerical:
            choice_idx = numerical_choices_map[id(numerical_choice)]
            numerical_decision[choice_idx] = numerical_choice

        categorical_encoding = np.zeros((1, num_category_choices))
        categorical_encoding[:, categorical_decision] = 1

        numerical_encoding = [np.arange(1) for _ in range(num_numerical_choices)]
        for i, choice in numerical_decision.items():
            numerical_encoding[i] = choice.choices
        x = np.atleast_2d(np.meshgrid(*numerical_encoding, indexing="ij"))
        if x.size != 0:
            x = x.reshape(num_numerical_choices, -1).T
        categorical_encoding = np.broadcast_to(
            categorical_encoding, (x.shape[0], num_category_choices)
        )
        encoding = np.concatenate([categorical_encoding, x], axis=1)
        all_possible_decisions.append(encoding)
    all_possible_decisions = np.concatenate(all_possible_decisions, axis=0)
    return all_possible_decisions, category_choices_map, numerical_choices_map


all_possible_decisions, category_choices_map, numerical_choices_map = (
    make_all_possible_decisions(x)
)

decision_taken = all_possible_decisions[-1]


def config_from_decision(
    decision: np.ndarray,
    decision_tree: DecisionNode,
    solver_space_scheme: dict,
    category_choices_map: dict[int, int],
    numerical_choices_map: dict[int, int],
):
    config = {}
    num_category_choices = len(category_choices_map)
    for key, value in solver_space_scheme.items():
        if isinstance(value, dict):
            config[key] = config_from_decision(
                decision=decision,
                decision_tree=decision_tree,
                solver_space_scheme=value,
                category_choices_map=category_choices_map,
                numerical_choices_map=numerical_choices_map,
            )

        elif isinstance(value, NumericalChoices):
            choice_idx = numerical_choices_map[id(value)] + num_category_choices
            decision_value = decision[choice_idx]
            config[key] = decision_value

        elif isinstance(value, CategoricalChoices):
            is_chosen = False
            try:
                fork = next(c for c in decision_tree.children if c.options_key == key)
            except StopIteration:
                assert False, "This should never happen"
            for choice in fork.children:
                choice_idx = category_choices_map[id(choice)]
                is_chosen = decision[choice_idx] == 1  # Assuming it can be only 0 or 1.
                if is_chosen:
                    child_config = config_from_decision(
                        decision=decision,
                        decision_tree=choice,
                        solver_space_scheme=choice.solver_space_scheme,
                        category_choices_map=category_choices_map,
                        numerical_choices_map=numerical_choices_map,
                    )
                    # Not sure:
                    if key != "block_type":
                        config[key] = child_config
                    else:
                        config |= child_config
                    break
            assert is_chosen

        else:
            config[key] = value
    return config


config = config_from_decision(
    decision_taken,
    decision_tree=solver_space.decision_tree,
    solver_space_scheme=solver_space.decision_tree.solver_space_scheme,
    category_choices_map=category_choices_map,
    numerical_choices_map=numerical_choices_map,
)
print(config)


def build_solver_scheme_from_config(config: dict):
    return KNOWN_SOLVER_COMPONENTS[
        config["block_type"]
    ].build_solver_scheme_from_config(
        config=config, build_inner_solver=build_solver_scheme_from_config
    )


scheme = build_solver_scheme_from_config(config)
pass
