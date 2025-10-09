from functools import cached_property

from block_matrix import FieldSplitScheme
from iterative_solver import (
    IterativeLinearSolver,
    get_equations_group_ids,
    get_variables_group_ids,
)


class THSolver(IterativeLinearSolver):
    def group_row_names(self) -> list[str]:
        return [
            "Flow mat.",
            "Energy mat.",
        ]

    def group_col_names(self) -> list[str]:
        return [
            "$p_{3D}$",
            "$T_{3D}$",
        ]

    @cached_property
    def variable_groups(self) -> list[list[int]]:
        return get_variables_group_ids(
            model=self,
            md_variables_groups=[
                [self.pressure(self.mdg.subdomains())],
                [self.temperature(self.mdg.subdomains())],
            ],
        )

    @cached_property
    def equation_groups(self) -> list[list[int]]:
        subdomains = self.mdg.subdomains()
        return get_equations_group_ids(
            model=self,
            equations_group_order=[
                [("mass_balance_equation", subdomains)],
                [("energy_balance_equation", subdomains)],
            ],
        )

    def make_solver_scheme(self) -> FieldSplitScheme:
        raise NotImplementedError
