import porepy as pp
import numpy as np
from experiments.models import Physics
from hm_solver import IterativeHMSolver as Solver
from experiments.thermal.thm_models import (
    ConstraintLineSearchNonlinearSolver,
    # Physics,
    get_barton_bandis_config,
    get_friction_coef_config,
)
from plot_utils import write_dofs_info
from stats import StatisticsSavingMixin

# from experiments.thermal.thm_solver import ThermalSolver

from porepy.models.poromechanics import Poromechanics

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
        vals[sides.east] *= 10
        return vals

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, sides.south, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        bc_values = np.zeros((self.nd, boundary_grid.num_cells))
        # 10 Mpa
        x = 0.5
        val = self.units.convert_units(5e6, units="Pa")
        # bc_values[1, sides.north] = -val * boundary_grid.cell_volumes[sides.north]
        # bc_values[0, sides.west] = val * boundary_grid.cell_volumes[sides.west] * x
        
        bc_values[1, sides.north] = -val * boundary_grid.cell_volumes[sides.north]
        bc_values[0, sides.north] = val * boundary_grid.cell_volumes[sides.north] * 0.1
        bc_values[0, sides.west] = val * boundary_grid.cell_volumes[sides.west]
        bc_values[0, sides.east] = -val * boundary_grid.cell_volumes[sides.east]
        
        return bc_values.ravel("F")

    def set_domain(self) -> None:
        self._domain = pp.Domain({"xmin": 0, "xmax": XMAX, "ymin": 0, "ymax": YMAX})

    def set_fractures(self) -> None:
        x = 0.3
        y = 0.5
        pts_list = [
            np.array(
                [
                    [x * XMAX, (0.95) * XMAX],  # x
                    [y * YMAX, y * YMAX],  # y
                ]
            )
        ]
        # self._fractures = []
        self._fractures = [pp.LineFracture(pts) for pts in pts_list]


class Setup(Geometry, Solver, StatisticsSavingMixin, Poromechanics):
    pass


def make_model(setup: dict):

    cell_size_multiplier = setup["grid_refinement"]

    DAY = 24 * 60 * 60

    shear = 1.2e10
    lame = 1.2e10
    biot = 0.47
    porosity = 1.3e-2
    specific_storage = 1/(lame + 2/3 * shear) * (biot - porosity) * (1 - biot)

    params = {
        "setup": setup,
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
                # "biot_coefficient": 1,  # for mandel
                density=2683.0,  # [kg * m^-3]
                porosity=porosity,  # [-]
                specific_storage=specific_storage,  # [Pa^-1]
                **get_barton_bandis_config(setup),
                **get_friction_coef_config(setup),
            ),
            "fluid": pp.FluidComponent(
                compressibility=4.559 * 1e-10,  # [Pa^-1], fluid compressibility
                # "compressibility": 0,  # for mandel
                density=998.2,  # [kg m^-3]
                viscosity=1.002e-3,  # [Pa s], absolute viscosity
            ),
            "numerical": pp.NumericalConstants(
                # experimnetal
                characteristic_displacement=1e-2,  # [m]
            ),
        },
        "reference_variable_values": pp.ReferenceVariableValues(
            pressure=1e6,  # [Pa]
        ),
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=0.5 * DAY,
            # dt_min_max=(0.01, 0.5),
            schedule=[0, 3 * DAY],
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

    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            "progressbars": False,
            "nl_convergence_tol": float("inf"),
            "nl_convergence_tol_res": 1e-7,
            "nl_divergence_tol": 1e8,
            "max_iterations": 25,
            # experimental
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "Global_line_search": 1,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 1,  # Set to 0 to use turn off the tailored line search
        },
    )

    write_dofs_info(model)
    print(model.simulation_name())


def experiment_1_barton_bandis_friction():
    setups = []
    for barton_bandis in [2]:
        for friction in [1]:
            for solver in [1, ]:
                setups.append(
                    {
                        "physics": 0,
                        "geometry": 1,
                        "barton_bandis_stiffness_type": barton_bandis,
                        "friction_type": friction,
                        "grid_refinement": 1,
                        "solver": solver,
                        "save_matrix": False,
                    }
                )
    for setup in setups:
        run_model(setup)


if __name__ == "__main__":
    # experiment_1_barton_bandis_friction()
    for g in [
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 10,
        # 33,
        # 40
        1, 2, 5, 25, 33, 40
    ]:
        run_model(
            {
                "physics": 1,
                "geometry": 0,
                "barton_bandis_stiffness_type": 2,
                "friction_type": 1,
                "grid_refinement": g,
                "solver": 2,
                "save_matrix": False,
            }
        )
    # run_model(
    #     {
    #         "physics": 0,
    #         "geometry": 0,
    #         "barton_bandis_stiffness_type": 2,
    #         "friction_type": 1,
    #         "grid_refinement": 1,
    #         "solver": 1,
    #         "save_matrix": True,
    #     }
    # )