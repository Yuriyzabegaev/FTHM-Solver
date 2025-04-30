from functools import cached_property
from full_petsc_solver import (
    LinearTransformedScheme,
    PetscKSPScheme,
    PetscFieldSplitScheme,
    PetscCompositeScheme,
    PcPythonPermutation,
)
from fixed_stress import make_fs_analytical_slow_new
from iterative_solver import IterativeLinearSolver
from mat_utils import csr_to_petsc
from solver_selection_thm.selector import SolverSelector
from thermal.thm_solver import THMSolver, make_pt_permutation
import porepy as pp
import numpy as np

from solver_selection_thm.solver_space import SolverSchemeBuilder


class PetscKSPSchemeBuilder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self,
        config: dict,
        build_inner_solver_scheme: callable,
        porepy_model: pp.PorePyModel,
    ):
        config = config.copy()
        del config["block_type"]
        pc_config = config.pop("preconditioner", None)
        pc = (
            build_inner_solver_scheme(config=pc_config, porepy_model=porepy_model)
            if pc_config is not None
            else None
        )
        return PetscKSPScheme(preconditioner=pc, **config)


class PetscFieldSplitSchemeBuilder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self,
        config: dict,
        build_inner_solver_scheme: callable,
        porepy_model: pp.PorePyModel,
    ):
        config = config.copy()
        del config["block_type"]
        complement_config = config.pop("complement", None)
        complement = (
            build_inner_solver_scheme(
                config=complement_config, porepy_model=porepy_model
            )
            if complement_config is not None
            else None
        )
        invert_config = config.pop("invert", None)
        invert = (
            build_inner_solver_scheme(config=invert_config, porepy_model=porepy_model)
            if invert_config is not None
            else None
        )
        python_pc_config = config.pop("python_pc", None)
        python_pc = (
            build_inner_solver_scheme(
                config=python_pc_config, porepy_model=porepy_model
            )
            if python_pc_config is not None
            else None
        )
        return PetscFieldSplitScheme(
            complement=complement, invert=invert, python_pc=python_pc, **config
        )


class FSAnalyticalSlowNewBuilder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self,
        config: dict,
        build_inner_solver_scheme: callable,
        porepy_model: pp.PorePyModel,
    ):
        config = config.copy()
        del config["block_type"]
        return lambda bmat: csr_to_petsc(
            make_fs_analytical_slow_new(model=porepy_model, J=bmat, **config).mat,
            bsize=1,
        )


class PetscCompositeSchemeBuilder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self,
        config: dict,
        build_inner_solver_scheme: callable,
        porepy_model: pp.PorePyModel,
    ):
        config = config.copy()
        del config["block_type"]
        solvers_configs = config.pop("solvers")
        return PetscCompositeScheme(
            solvers=[
                build_inner_solver_scheme(
                    config=solvers_configs[i], porepy_model=porepy_model
                )
                for i in range(len(solvers_configs))
            ],
            **config,
        )


class PcPythonPermutationBuilder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self,
        config: dict,
        build_inner_solver_scheme: callable,
        porepy_model: pp.PorePyModel,
    ):
        config = config.copy()
        del config["block_type"]
        assert config["permutation_type"] == "pt_permutation"

        return lambda bmat: PcPythonPermutation(
            make_pt_permutation(
                bmat, p_groups=config["p_groups"], t_groups=config["t_groups"]
            ),
            block_size=config["block_size"],
        )


class LinearTransformedSchemeBuilder(SolverSchemeBuilder):
    def build_solver_scheme_from_config(
        self,
        config: dict,
        build_inner_solver_scheme: callable,
        porepy_model: pp.PorePyModel,
    ):
        config = config.copy()
        del config["block_type"]

        return LinearTransformedScheme(
            right_transformations=[
                lambda bmat: porepy_model.Qright(contact_group=0, u_intf_group=4)
            ]
            if config.get("Qright", False)
            else [],
            left_transformations=[
                lambda bmat: porepy_model.scale_energy_balance(bmat),
            ]
            if config.get("scale_energy_balance", False)
            else [],
            inner=build_inner_solver_scheme(
                config=config["inner"], porepy_model=porepy_model
            ),
        )


KNOWN_SOLVER_COMPONENTS_THM: dict[str, SolverSchemeBuilder] = {
    "PetscKSPScheme": PetscKSPSchemeBuilder(),
    "PetscFieldSplitScheme": PetscFieldSplitSchemeBuilder(),
    "fs_analytical_slow_new": FSAnalyticalSlowNewBuilder(),
    "PetscCompositeScheme": PetscCompositeSchemeBuilder(),
    "PcPythonPermutation": PcPythonPermutationBuilder(),
    "LinearTransformedScheme": LinearTransformedSchemeBuilder(),
}


class SolverSelectionMixin(IterativeLinearSolver):
    def collect_characteristics_for_linear_solver_selection(self) -> np.ndarray:
        raise NotImplementedError

    @cached_property
    def solver_selector(self) -> SolverSelector:
        return self.params["setup"]["linear_solver_selector"]

    def make_solver_scheme(self):
        characteristics = self.collect_characteristics_for_linear_solver_selection()
        try:
            solver_idx = getattr(self, "_solver_id")
        except AttributeError:
            solver_idx = None
        scheme, solver_idx = self.solver_selector.select_linear_solver_scheme(
            characteristics=characteristics,
            porepy_model=self,
            active_solver_idx=solver_idx,
        )
        self._solver_id = solver_idx
        return scheme

    def solve_linear_system(self):
        sol = super().solve_linear_system()
        self.solver_selector.provide_performance_feedback(
            solve_time=self._solve_time,
            construct_time=self._construction_time,
            success=self._linear_solve_stats.petsc_converged_reason > 0,
        )
        return sol


class SolverSelectionMixinTHM(SolverSelectionMixin, THMSolver):
    def collect_characteristics_for_linear_solver_selection(self) -> np.ndarray:
        return np.array(
            [
                self._linear_solve_stats.temp_min,
                self._linear_solve_stats.temp_max,
                self._linear_solve_stats.cfl,
                self._linear_solve_stats.enthalpy_max,
                self._linear_solve_stats.enthalpy_mean,
                self._linear_solve_stats.fourier_max,
                self._linear_solve_stats.fourier_mean,
            ]
        )
