import porepy as pp
import numpy as np
import time
from thermal.models import Physics, ConstraintLineSearchNonlinearSolver
from thermal.thm_solver import THMSolver
from plot_utils import write_dofs_info
from stats import StatisticsSavingMixin

XMAX = 2000
YMAX = 2000


class Geometry(pp.SolutionStrategy):
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, sides.south, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        bc_values = np.zeros((self.nd, boundary_grid.num_cells))
        # rho * g * h
        # 2683 * 10 * 3000
        val = self.units.convert_units(8e7, units="Pa")
        bc_values[1, sides.north] = -val * boundary_grid.cell_volumes[sides.north]
        #  make the gradient
        bc_values[0, sides.west] = val * boundary_grid.cell_volumes[sides.west] * 1.2
        bc_values[0, sides.east] = -val * boundary_grid.cell_volumes[sides.east] * 1.2

        return bc_values.ravel("F")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        if self.params["linear_solver_config"].get("thermal_diffusion_bc") == "noflux":
            return pp.BoundaryCondition(sd, boundary_faces, "neu")
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def locate_source(self, subdomains):
        source_loc_x = XMAX * 0.5
        source_loc_y = YMAX * 0.5
        ambient = [sd for sd in subdomains if sd.dim == self.nd]
        fractures = [sd for sd in subdomains if sd.dim == self.nd - 1]
        lower = [sd for sd in subdomains if sd.dim <= self.nd - 2]

        x, y, z = np.concatenate([sd.cell_centers for sd in ambient], axis=1)
        source_loc = np.argmin((x - source_loc_x) ** 2 + (y - source_loc_y) ** 2)
        src_mat = np.zeros(x.size)
        src_mat[source_loc] = 1

        zeros_frac = np.zeros(sum(sd.num_cells for sd in fractures))
        zeros_lower = np.zeros(sum(sd.num_cells for sd in lower))
        return np.concatenate([src_mat, zeros_frac, zeros_lower])

    def fluid_source_mass_rate(self):
        if self.params["linear_solver_config"]["steady_state"]:
            return 0
        else:
            return self.units.convert_units(1e1, "kg * s^-1")
            # maybe inject and then stop injecting?

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        src = self.locate_source(subdomains)
        src *= self.fluid_source_mass_rate()
        return super().fluid_source(subdomains) + pp.ad.DenseArray(src)

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        src = self.locate_source(subdomains)
        src *= self.fluid_source_mass_rate()
        cv = self.fluid.components[0].specific_heat_capacity
        t_inj = (
            self.units.convert_units(273 + 40, "K")
            - self.reference_variable_values.temperature
        )
        src *= cv * t_inj
        return super().energy_source(subdomains) + pp.ad.DenseArray(src)

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmin": 0,
                "xmax": XMAX,
                "ymin": 0,
                "ymax": YMAX,
            }
        )

    def set_fractures(self) -> None:
        self._fractures = []

    def after_simulation(self):
        super().after_simulation()
        vals = self.equation_system.get_variable_values(time_step_index=0)
        name = f"{self.simulation_name()}_endstate_{int(time.time() * 1000)}.npy"
        print("Saving", name)
        self.params["linear_solver_config"]["end_state_filename"] = name
        np.save(name, vals)


class Setup(Geometry, THMSolver, StatisticsSavingMixin, Physics):
    pass


def make_model(setup: dict):

    cell_size_multiplier = setup["grid_refinement"]

    DAY = 24 * 60 * 60

    shear = 1.2e10
    lame = 1.2e10
    if setup["steady_state"]:
        biot = 0
        dt_init = 1e0
        end_time = 1e1
    else:
        biot = 0.47
        dt_init = 1e-3
        if setup["grid_refinement"] >= 33:
            dt_init = 1e-4  # Is this necessary?
        end_time = 5e2
    porosity = 1.3e-2  # probably on the low side

    params = {
        "linear_solver_config": setup,
        "folder_name": "visualization_2d_nofrac",
        "material_constants": {
            "solid": pp.SolidConstants(
                # IMPORTANT
                permeability=1e-13,  # [m^2]
                residual_aperture=1e-3,  # [m]
                # LESS IMPORTANT
                shear_modulus=shear,  # [Pa]
                lame_lambda=lame,  # [Pa]
                dilation_angle=5 * np.pi / 180,  # [rad]
                normal_permeability=1e-4,
                # granite
                biot_coefficient=biot,  # [-]
                density=2683.0,  # [kg * m^-3]
                porosity=porosity,  # [-]
                friction_coefficient=0.577,  # [-]
                # Thermal
                specific_heat_capacity=720.7,
                thermal_conductivity=0.1,  # Diffusion coefficient
                thermal_expansion=9.66e-6,
            ),
            "fluid": pp.FluidComponent(
                compressibility=4.559 * 1e-10,  # [Pa^-1], fluid compressibility
                density=998.2,  # [kg m^-3]
                viscosity=1.002e-3,  # [Pa s], absolute viscosity
                # Thermal
                specific_heat_capacity=4182.0,  # Вместимость
                thermal_conductivity=0.5975,  # Diffusion coefficient
                thermal_expansion=2.068e-4,  # Density(T)
            ),
            "numerical": pp.NumericalConstants(
                characteristic_displacement=2e0,  # [m]
            ),
        },
        "reference_variable_values": pp.ReferenceVariableValues(
            pressure=3.5e7,  # [Pa]
            temperature=273 + 120,
        ),
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=dt_init * DAY,
            schedule=[0, end_time * DAY],
            iter_max=30,
            constant_dt=False,
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
            "max_iterations": 30,
            # experimental
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "Global_line_search": 0,  # Set to 1 to use turn on a residual-based line search
            "Local_line_search": 1,  # Set to 0 to use turn off the tailored line search
        },
    )

    write_dofs_info(model)
    print(model.simulation_name())


if __name__ == "__main__":

    for g in reversed([
        # 1,
        # 2,
        # 5,
        25,
        # 33,
        # 40,
    ]):
        for s in [
            "CPR",
            "SAMG",
            # "S4_diag",
            # "SAMG+ILU",
            # "S4_diag+ILU",
            # "AAMG+ILU",
        ]:
            for bc in [
                # "noflux",
                "dir",
            ]:
                common_params = {
                    "geometry": "nofrac",
                    "save_matrix": False,
                    "solver": s,
                    "thermal_diffusion_bc": bc,
                }
                print("Running steady state")
                params = {
                    "grid_refinement": g,
                    "steady_state": True,
                } | common_params
                run_model(params)
                end_state_filename = params["end_state_filename"]

                print("Running injection")
                params = {
                    "grid_refinement": g,
                    "steady_state": False,
                    "initial_state": end_state_filename,
                } | common_params
                run_model(params)
