from itertools import count, product
from typing import Any, Callable, Optional, Protocol
import numpy as np


class CategoricalChoices:
    def __init__(self, choices: list[dict | Any]):
        self.choices = choices

    def __repr__(self):
        return f"CategoricalChoices({self.choices})"


class NumericalChoices:
    def __init__(
        self,
        choices: np.ndarray,
        tag: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
    ):
        self.choices: np.ndarray = np.array(choices)
        self.dtype: np.dtype = dtype or self.choices.dtype
        self.tag: str | None = tag

    def __repr__(self):
        return f"NumericalChoices({self.tag}: {self.choices})"

    def __str__(self) -> str:
        return f"{self.tag}: Choices from {min(self.choices)} to {max(self.choices)}, len = {len(self.choices)}"


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
    def __init__(self, solver_space_scheme: dict | Any):
        self.solver_space_scheme: dict | Any = solver_space_scheme
        self.children: list[ForkNode] = []
        self.numerical_choices: list[NumericalChoices] = []

    def _str(self, prefix="") -> str:
        if not isinstance(self.solver_space_scheme, dict):
            return f"{prefix}{self.solver_space_scheme}"
        k, v = next(iter(self.solver_space_scheme.items()))
        if isinstance(v, CategoricalChoices):
            v = "CategoricalChoices"
        repr = f"{prefix}{k}: {v}"
        child_prefix = f"{prefix}| "
        if len(self.numerical_choices) > 0:
            numerical_repr = [f"{child_prefix}{v}" for v in self.numerical_choices]
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
        my_numerical_choices = set(self.numerical_choices)
        if len(self.children) == 0:
            flat_config = FlatSolverDecision(numerical=my_numerical_choices)
            return [flat_config]

        children_solver_spaces = [c.list_possible_solvers() for c in self.children]

        merged_results = []
        for tuple_of_decisions in list(product(*children_solver_spaces)):
            cat = set(x for conf in tuple_of_decisions for x in conf.categorical)
            num = set(x for conf in tuple_of_decisions for x in conf.numerical)
            num |= my_numerical_choices
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

    def list_possible_solvers(self) -> list[FlatSolverDecision]:
        if len(self.children) == 0:
            assert False, "Why Fork node with no options?"

        solver_space = []
        for child in self.children:
            child_solver_space = child.list_possible_solvers()
            for solver in child_solver_space:
                solver.categorical.add(child)
            solver_space.extend(child_solver_space)
        return solver_space


class SolverSchemeProtocol(Protocol):
    def make_solver(self) -> Any:
        pass


class SolverSchemeBuilder:
    def build_solver_scheme_from_config(
        self,
        config: dict,
        build_inner_solver_scheme: Callable,
        porepy_model,
    ) -> SolverSchemeProtocol:
        raise NotImplementedError


def build_decision_tree(
    solver_space_scheme, current_decision_node: DecisionNode, options_key: str
):
    match solver_space_scheme:
        case solver_space_scheme if isinstance(solver_space_scheme, dict):
            for key, value in solver_space_scheme.items():
                build_decision_tree(
                    solver_space_scheme=value,
                    current_decision_node=current_decision_node,
                    options_key=key,
                )
        case categorical_choice if isinstance(categorical_choice, CategoricalChoices):
            fork_node = ForkNode(categorical_choice, options_key=options_key)
            current_decision_node.children.append(fork_node)
            for decision in categorical_choice.choices:
                new_decision_node = DecisionNode(decision)
                fork_node.children.append(new_decision_node)
                build_decision_tree(
                    solver_space_scheme=decision,
                    current_decision_node=new_decision_node,
                    options_key=options_key,
                )
        case numerical_choices if isinstance(numerical_choices, NumericalChoices):
            if numerical_choices.tag is None:
                numerical_choices.tag = options_key
            current_decision_node.numerical_choices.append(numerical_choices)


def make_choices_map(
    solver_space: list[FlatSolverDecision],
) -> tuple[dict[int, int], dict[int, int]]:
    category_choices_map: dict[int, int] = dict()
    numerical_choices_map: dict[int, int] = dict()
    category_choices_counter = count()
    numerical_choices_counter = count()

    for solver in solver_space:
        for choice in solver.categorical:
            choice_id = id(choice)
            if category_choices_map.get(choice_id) is not None:
                continue
            category_idx = next(category_choices_counter)
            category_choices_map[choice_id] = category_idx
        for numerical_choice in solver.numerical:
            choice_id = id(numerical_choice)
            if numerical_choices_map.get(choice_id) is not None:
                continue

            numerical_idx = next(numerical_choices_counter)
            numerical_choices_map[choice_id] = numerical_idx
    return category_choices_map, numerical_choices_map


def make_all_decisions_encoding(
    solver_space: list[FlatSolverDecision],
    category_choices_map: dict[int, int],
    numerical_choices_map: dict[int, int],
) -> np.ndarray:
    num_category_choices = len(category_choices_map)
    num_numerical_choices = len(numerical_choices_map)

    all_possible_decisions: list[np.ndarray] = []
    for solver in solver_space:
        categorical_decision = [
            category_choices_map[id(choice)] for choice in solver.categorical
        ]
        numerical_decision: dict[int, NumericalChoices] = {
            numerical_choices_map[id(numerical_choice)]: numerical_choice
            for numerical_choice in solver.numerical
        }

        categorical_encoding = np.zeros((1, num_category_choices))
        categorical_encoding[:, categorical_decision] = 1

        numerical_encoding = [np.zeros(1) for _ in range(num_numerical_choices)]
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
    return np.concatenate(all_possible_decisions, axis=0)


class SolverSpace:
    def __init__(
        self,
        solver_space_scheme: dict,
        solver_scheme_builders: dict[str, SolverSchemeBuilder],
    ):
        self.solver_scheme_builders: dict[str, SolverSchemeBuilder] = (
            solver_scheme_builders
        )
        self.solver_space_scheme: dict = solver_space_scheme
        self.decision_tree = DecisionNode(solver_space_scheme)
        build_decision_tree(solver_space_scheme, self.decision_tree, options_key="")

        self.flat_solver_decisions: list[FlatSolverDecision] = (
            self.decision_tree.list_possible_solvers()
        )

        category_choices_map, numerical_choices_map = make_choices_map(
            self.flat_solver_decisions
        )
        self.category_choices_map: dict[int, int] = category_choices_map
        self.numerical_choices_map: dict[int, int] = numerical_choices_map
        self.all_decisions_encoding: np.ndarray = make_all_decisions_encoding(
            solver_space=self.flat_solver_decisions,
            category_choices_map=self.category_choices_map,
            numerical_choices_map=self.numerical_choices_map,
        )

    def config_from_decision(
        self,
        decision: np.ndarray,
        decision_tree: Optional[DecisionNode] = None,
        solver_space_scheme: Optional[dict | Any] = None,
    ):
        if solver_space_scheme is None:
            # Starting recursion
            solver_space_scheme = self.solver_space_scheme
        if decision_tree is None:
            # Starting recursion
            decision_tree = self.decision_tree
        if not isinstance(solver_space_scheme, dict):
            # Ending recursion
            return solver_space_scheme

        numerical_choices_map = self.numerical_choices_map
        category_choices_map = self.category_choices_map

        config = {}
        num_category_choices = len(self.category_choices_map)
        for key, value in solver_space_scheme.items():
            if isinstance(value, dict):
                config[key] = self.config_from_decision(
                    decision=decision,
                    decision_tree=decision_tree,
                    solver_space_scheme=value,
                )

            elif isinstance(value, NumericalChoices):
                choice_idx = numerical_choices_map[id(value)] + num_category_choices
                decision_value = decision[choice_idx]
                config[key] = decision_value.astype(value.dtype)

            elif isinstance(value, CategoricalChoices):
                is_chosen = False
                try:
                    fork = next(
                        c for c in decision_tree.children if c.options_key == key
                    )
                except StopIteration:
                    assert False, "This should never happen"
                for choice in fork.children:
                    choice_idx = category_choices_map[id(choice)]
                    is_chosen = (
                        decision[choice_idx] == 1
                    )  # Assuming it can be only 0 or 1.
                    if is_chosen:
                        child_config = self.config_from_decision(
                            decision=decision,
                            decision_tree=choice,
                            solver_space_scheme=choice.solver_space_scheme,
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

    def build_solver_scheme(self, config: dict, porepy_model):
        return self.solver_scheme_builders[
            config["block_type"]
        ].build_solver_scheme_from_config(
            config=config,
            build_inner_solver_scheme=self.build_solver_scheme,
            porepy_model=porepy_model,
        )
