from full_petsc_solver import (
    PetscKSPScheme,
    PetscFieldSplitScheme,
    PetscCompositeScheme,
    PcPythonPermutation,
)
from fixed_stress import make_fs_analytical_slow_new
from thermal.thm_solver import make_pt_permutation

from solver_selection_thm.solver_space import SolverSchemeBuilder


class PetscKSPSchemeBuilder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self, config: dict, build_inner_solver: callable
    ):
        config = config.copy()
        del config["block_type"]
        pc_config = config.pop("preconditioner", None)
        pc = build_inner_solver(pc_config) if pc_config is not None else None
        return PetscKSPScheme(preconditioner=pc, **config)


class PetscFieldSplitSchemeBuilder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self, config: dict, build_inner_solver: callable
    ):
        config = config.copy()
        del config["block_type"]
        complement_config = config.pop("complement", None)
        complement = (
            build_inner_solver(complement_config)
            if complement_config is not None
            else None
        )
        return PetscFieldSplitScheme(complement=complement, **config)


class fs_analytical_slow_new_Builder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self, config: dict, build_inner_solver: callable
    ):
        config = config.copy()
        del config["block_type"]
        return lambda bmat: make_fs_analytical_slow_new(**config)


class PetscCompositeSchemeBuilder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self, config: dict, build_inner_solver: callable
    ):
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


class PcPythonPermutationBuilder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self, config: dict, build_inner_solver: callable
    ):
        config = config.copy()
        del config["block_type"]
        assert config["permutation_type"] == "pt_permutation"

        return lambda bmat: PcPythonPermutation(
            make_pt_permutation(
                bmat, p_groups=config["p_groups"], t_groups=config["t_groupd"]
            ),
            block_size=config["block_type"],
        )


KNOWN_SOLVER_COMPONENTS_THM: dict[str, SolverSchemeBuilder] = {
    "PetscKSPScheme": PetscKSPSchemeBuilder(),
    "PetscFieldSplitScheme": PetscFieldSplitSchemeBuilder(),
    "fs_analytical_slow_new": fs_analytical_slow_new_Builder(),
    "PetscCompositeScheme": PetscCompositeSchemeBuilder(),
    "PcPythonPermutation": PcPythonPermutationBuilder(),
}
