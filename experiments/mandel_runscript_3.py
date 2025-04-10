import porepy as pp
import numpy as np
from experiments.models import (
    ConstraintLineSearchNonlinearSolver,
    Physics,
    get_barton_bandis_config,
    get_friction_coef_config,
)
from FTHM_Solver.hm_solver import IterativeHMSolver as Solver
from plot_utils import write_dofs_info
from FTHM_Solver.stats import StatisticsSavingMixin

XMAX = 1000
YMAX = 1000
ZMAX = 1000


class Geometry(pp.SolutionStrategy):
    def initial_condition(self) -> None:
        super().initial_condition()
        num_cells = sum([sd.num_cells for sd in self.mdg.subdomains()])
        val = self.reference_variable_values.pressure * np.ones(num_cells)
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                variables=[self.pressure_variable],
                time_step_index=time_step_index,
            )

        for iterate_index in self.iterate_indices:
            self.equation_system.set_variable_values(
                val,
                variables=[self.pressure_variable],
                iterate_index=iterate_index,
            )

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

    def bc_values_pressure(self, boundary_grid):
        vals = super().bc_values_pressure(boundary_grid)
        sides = self.domain_boundary_sides(boundary_grid)
        val = 20
        vals[sides.east] *= val
        return vals

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, sides.south, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        bc_values = np.zeros((self.nd, boundary_grid.num_cells))
        val = self.units.convert_units(3e6, units="Pa")
        bc_values[2, sides.bottom] = val * boundary_grid.cell_volumes[sides.bottom]
        bc_values[2, sides.top] = -val * boundary_grid.cell_volumes[sides.top]
        bc_values[1, sides.north] = -val * boundary_grid.cell_volumes[sides.north]
        bc_values[0, sides.north] = val * boundary_grid.cell_volumes[sides.north] * 0.1
        bc_values[0, sides.west] = val * boundary_grid.cell_volumes[sides.west]
        bc_values[0, sides.east] = -val * boundary_grid.cell_volumes[sides.east]

        return bc_values.ravel("F")

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {"xmin": 0, "xmax": XMAX, "ymin": 0, "ymax": YMAX, "zmin": 0, "zmax": ZMAX}
        )

    def set_fractures(self) -> None:
        pts_list = np.array(
            [
                [[0.1, 0.1, 0.9, 0.9], [0.5, 0.5, 0.5, 0.5], [0.2, 0.8, 0.8, 0.2]],
                [[0.15, 0.15, 0.4, 0.4], [0.7, 0.7, 0.2, 0.2], [0.2, 0.8, 0.8, 0.2]],
                [[0.45, 0.45, 0.6, 0.6], [0.3, 0.3, 0.8, 0.8], [0.2, 0.8, 0.8, 0.2]],
                [[0.6, 0.6, 0.8, 0.8], [0.2, 0.2, 0.8, 0.8], [0.2, 0.8, 0.8, 0.2]],
            ]
        )
        pts_list[:, 0] *= XMAX
        pts_list[:, 1] *= YMAX
        pts_list[:, 2] *= ZMAX

        self._fractures = [pp.PlaneFracture(pts) for pts in pts_list]


class Setup(Geometry, Solver, StatisticsSavingMixin, Physics):
    pass


def make_model(setup: dict):

    cell_size_multiplier = setup["grid_refinement"]

    WEEK = 24 * 60 * 60 * 7

    shear = 1.2e10
    lame = 1.2e10
    biot = 0.47
    porosity = 1.3e-2
    specific_storage = 1 / (lame + 2 / 3 * shear) * (biot - porosity) * (1 - biot)

    params = {
        "linear_solver_config": setup,
        "material_constants": {
            "solid": pp.SolidConstants(
                shear_modulus=shear,  # [Pa]
                lame_lambda=lame,  # [Pa]
                dilation_angle=5 * np.pi / 180,  # [rad]
                residual_aperture=1e-4,  # [m]
                normal_permeability=1e-4,
                permeability=1e-14,  # [m^2]
                # granite
                biot_coefficient=biot,  # [-]
                density=2683.0,  # [kg * m^-3]
                porosity=porosity,  # [-]
                specific_storage=specific_storage,  # [Pa^-1]
                **get_barton_bandis_config(setup),
                **get_friction_coef_config(setup),
            ),
            "fluid": pp.FluidComponent(
                compressibility=4.559 * 1e-10,  # [Pa^-1], fluid compressibility
                density=998.2,  # [kg m^-3]
                viscosity=1.002e-3,  # [Pa s], absolute viscosity
            ),
            "numerical": pp.NumericalConstants(
                characteristic_displacement=1e-1,  # [m]
            ),
        },
        "reference_variable_values": pp.ReferenceVariableValues(
            pressure=1e6,  # [Pa]
        ),
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=0.25 * WEEK,
            schedule=[0, 1.5 * WEEK],
            iter_max=25,
            constant_dt=True,
        ),
        "units": pp.Units(kg=1e10),
        "meshing_arguments": {
            "cell_size": (0.1 * XMAX / cell_size_multiplier),
        },
        # experimental
        "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively to increase robustness
    }
    return Setup(params)


def run_model(setup: dict):
    model = make_model(setup)
    model.prepare_simulation()
    print(model.simulation_name())
    # pp.plot_grid(model.mdg, plot_2d=True, fracturewidth_1d=5)

    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            "progressbars": False,
            "nl_convergence_tol": float("inf"),
            "nl_convergence_tol_res": 1e-7,
            "nl_divergence_tol": 1e8,
            "max_iterations": 100,
            # experimental
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "Global_line_search": 1,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 1,  # Set to 0 to use turn off the tailored line search
        },
    )
    write_dofs_info(model)

    print(model.simulation_name())


if __name__ == "__main__":
    # solver = 21
    # for g in [0.25, 0.5]:
    #     run_model(
    #         {
    #             "physics": 1,
    #             "geometry": 0.3,
    #             "barton_bandis_stiffness_type": 2,
    #             "friction_type": 1,
    #             "grid_refinement": g,
    #             "solver": solver,
    #             "permeability": 0,
    #         }
    #     )

    solver = 3
    for g in [0.25, 0.5, 1, 2, 3, 3.6, 4]:
        run_model(
            {
                "physics": 1,
                "geometry": 0.3,
                "barton_bandis_stiffness_type": 2,
                "friction_type": 1,
                "grid_refinement": g,
                "solver": solver,
                "permeability": 0,
                # "save_matrix": True,
            }
        )
