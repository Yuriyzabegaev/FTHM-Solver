import numpy as np
import porepy as pp
from porepy.models.constitutive_laws import (
    CubicLawPermeability,
    DimensionDependentPermeability,
    SpecificStorage,
)
from porepy.applications.md_grids.fracture_sets import (
    seven_fractures_one_L_intersection,
)
from porepy.models.poromechanics import Poromechanics

from plot_utils import get_gmres_iterations, load_data, load_matrix_rhs_state_iterate_dt
from pp_utils import MyPetscSolver, StatisticsSavingMixin


SETUP_REFERENCE = {
    "physics": 1,  # 0 - simplified; 1 - full
    "geometry": 1,  # 1 - 2D 1 fracture, 2 - 2D 7 fractures, 3 - 3D (TODO);
    "barton_bandis_stiffness_type": 1,  # 0 - off; 1 - small; 2 - medium, 3 - large
    "friction_type": 1,  # 0 - small, 1 - medium, 2 - large
    "grid_refinement": 1,  # 1 - coarsest level
    "solver": 2,  # 0 - Direct solver, 1 - Richardson + fixed stress splitting scheme, 2 - GMRES + scalable prec.
    "save_matrix": False,  # Save each matrix and state array. Might take space.
}


class Physics(DimensionDependentPermeability, SpecificStorage, Poromechanics):

    # Source

    fluid_source_key = "fluid_source"

    def locate_source(self, cell_centers: np.ndarray) -> np.ndarray:
        x = cell_centers[0]
        y = cell_centers[1]
        distance = np.sqrt((x - 0.5 * XMAX) ** 2 + (y - 0.5 * YMAX) ** 2)
        loc = np.where(distance == distance.min())[0][0]
        return loc

    def get_source_intensity(self, t):
        t_max = 6
        peak_intensity = self.fluid.convert_units(1e-3, "m^3 * s^-1")
        t_mid = t_max / 2
        if t <= t_mid:
            return t / t_mid * peak_intensity
        else:
            return (2 - t / t_mid) * peak_intensity * -1

    def _fluid_source(self, sd: pp.GridLike) -> np.ndarray:
        src = np.zeros(sd.num_cells)
        if sd.dim == (self.nd - 1):
            source_loc = self.locate_source(sd.cell_centers)
            src[source_loc] = self.get_source_intensity(t=self.time_manager.time)
        return src

    def update_time_dependent_ad_arrays(self) -> None:
        super().update_time_dependent_ad_arrays()
        for sd, data in self.mdg.subdomains(return_data=True):
            vals = self._fluid_source(sd)
            pp.set_solution_values(
                name=self.fluid_source_key, values=vals, data=data, iterate_index=0
            )

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        intf_source = super().fluid_source(subdomains)
        source = pp.ad.TimeDependentDenseArray(
            self.fluid_source_key, domains=subdomains
        )
        rho = self.fluid_density(subdomains)
        return intf_source + rho * source

    # Boundary conditions

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
        # 10 Mpa
        x = 0.5  # 1
        val = self.solid.convert_units(1e7, units="Pa")
        bc_values[1, sides.north] = -val * boundary_grid.cell_volumes[sides.north]
        bc_values[0, sides.west] = val * boundary_grid.cell_volumes[sides.west] * x
        return bc_values.ravel("F")

    # Constitutive laws

    def simplified_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        matrix = [sd for sd in subdomains if sd.dim == self.nd]
        dp = self.perturbation_from_reference("pressure", matrix)
        phi_0 = self.reference_porosity(matrix)
        cf = self.fluid_compressibility(matrix)
        M_inv = self.specific_storage(matrix)  # 1 / M
        pressure_term = (M_inv + phi_0 * cf) * dp
        matrix_term = (
            pressure_term
            + self.porosity_change_from_displacement(matrix)
            + self._mpsa_consistency(matrix, self.darcy_keyword, self.pressure_variable)
        )
        # (1/M + phi_0 * cf) p + alpha ∇ * u

        fracture = [sd for sd in subdomains if sd.dim < self.nd]
        cfc = self.residual_aperture(fracture) * self.fluid_compressibility(fracture)
        dp = self.perturbation_from_reference("pressure", fracture)
        fracture_term = cfc * dp + self.aperture(fracture)
        # nu_0 * cf * p + nu

        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        result = (
            projection.cell_prolongation(matrix) @ matrix_term
            + projection.cell_prolongation(fracture) @ fracture_term
        )

        cell_volumes = self.wrap_grid_attribute(subdomains, "cell_volumes", dim=1)
        rho_ref = pp.ad.Scalar(self.fluid.density(), "reference_fluid_density")
        result *= cell_volumes * rho_ref

        result.set_name("simplified mass")
        return result

    def fluid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        physics_type = self.params["setup"]["physics"]
        if physics_type == 1:
            return super().fluid_mass(subdomains)
        elif physics_type == 0:
            return self.simplified_mass(subdomains)
        raise ValueError(physics_type)

    def fracture_permeability(self, subdomains):
        physics_type = self.params["setup"]["physics"]
        if physics_type == 1:
            return CubicLawPermeability.cubic_law_permeability(self, subdomains)
        elif physics_type == 0:
            const_perm = DimensionDependentPermeability.fracture_permeability(
                self, subdomains
            )
            return const_perm / 12

        raise ValueError(physics_type)


XMAX = 1
YMAX = 1
ZMAX = 1

class Geometry2D1F(pp.SolutionStrategy):

    def set_domain(self) -> None:
        self._domain = pp.Domain({"xmin": 0, "xmax": XMAX, "ymin": 0, "ymax": YMAX})

    def set_fractures(self) -> None:
        x = 0.3
        y = 0.5
        pts_list = [
            np.array(
                [
                    [x * XMAX, (1 - x) * XMAX],  # x
                    [y * YMAX, y * YMAX],  # y
                ]
            )
        ]
        self._fractures = [pp.LineFracture(pts) for pts in pts_list]


class Geometry2D7F(pp.SolutionStrategy):

    def set_domain(self) -> None:
        self._domain = pp.Domain({"xmin": 0, "xmax": XMAX, "ymin": 0, "ymax": YMAX})

    def set_fractures(self) -> None:
        self._fractures = seven_fractures_one_L_intersection()


class Geometry3D1F(pp.SolutionStrategy):

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {"xmin": 0, "xmax": XMAX, "ymin": 0, "ymax": YMAX, "zmin": 0, "zmax": ZMAX}
        )

    def set_fractures(self) -> None:
        x = 0.3
        y = 0.3
        pts_list = [
            np.array(
                [
                    [x * XMAX, (1 - x) * XMAX, (1 - x) * XMAX, x * XMAX],  # x
                    [y * YMAX, (1 - y) * YMAX, y * YMAX, (1 - y) * YMAX],  # y
                    [z * ZMAX, z * ZMAX, z * ZMAX, z * ZMAX],  # z
                ]
            )
            for z in [
                0.2,
                0.4,
                0.5,
                0.6,
                0.8,
            ]
        ]
        self._fractures = [pp.PlaneFracture(pts) for pts in pts_list]


def get_barton_bandis_config(setup: dict):
    bb_type = setup["barton_bandis_stiffness_type"]
    if bb_type == 0:
        return {
            # Barton-Bandis elastic deformation. Defaults to 0.
            "fracture_gap": 0,
            "maximum_elastic_fracture_opening": 0,
            # [Pa m^-1]  # increase this to make easier to converge (* 10) (1/10 of lame parameter)
            "fracture_normal_stiffness": 0,
        }
    elif bb_type == 1:
        return {
            "fracture_gap": 1e-4,
            "maximum_elastic_fracture_opening": 5e-5,
            "fracture_normal_stiffness": 1.2e8,
        }
    elif bb_type == 2:
        return {
            "fracture_gap": 1e-4,
            "maximum_elastic_fracture_opening": 5e-5,
            "fracture_normal_stiffness": 1.2e9,
        }
    elif bb_type == 3:
        return {
            "fracture_gap": 1e-4,
            "maximum_elastic_fracture_opening": 5e-5,
            "fracture_normal_stiffness": 1.2e10,
        }
    raise ValueError(bb_type)


def get_friction_coef_config(setup: dict):
    friction_type = setup["friction_type"]
    if friction_type == 0:
        return {
            "friction_coefficient": 0.1,  # [-]
        }
    elif friction_type == 1:
        return {
            "friction_coefficient": 0.577,  # [-]
        }
    elif friction_type == 2:
        return {"friction_coefficient": 0.8}
    raise ValueError(friction_type)


def make_model(setup: dict):
    geometry_type = setup["geometry"]
    if geometry_type == 1:

        class Setup(Geometry2D1F, MyPetscSolver, StatisticsSavingMixin, Physics):
            pass

    elif geometry_type == 2:

        class Setup(Geometry2D7F, MyPetscSolver, StatisticsSavingMixin, Physics):
            pass

    elif geometry_type == 3:

        class Setup(Geometry3D1F, MyPetscSolver, StatisticsSavingMixin, Physics):
            pass 

    else:
        raise ValueError(geometry_type)

    dt = 0.5

    cell_size_multiplier = setup["grid_refinement"]

    params = {
        "setup": setup,
        "material_constants": {
            "solid": pp.SolidConstants(
                (
                    {
                        "shear_modulus": 1.2e10,  # [Pa]
                        "lame_lambda": 1.2e10,  # [Pa]
                        "dilation_angle": 5 * np.pi / 180,  # [rad]
                        "friction_coefficient": 0.577,  # [-]
                        "residual_aperture": 1e-4,  # [m]
                        "normal_permeability": 1e-4,
                        "permeability": 1e-14,  # [m^2]
                        # granite
                        "biot_coefficient": 0.47,  # [-]
                        "density": 2683.0,  # [kg * m^-3]
                        "porosity": 1.3e-2,  # [-]
                        "specific_storage": 4.74e-10,  # [Pa^-1]
                    }
                    | get_barton_bandis_config(setup)
                    | get_friction_coef_config(setup)
                )
            ),
            "fluid": pp.FluidConstants(
                {
                    "compressibility": 4.559
                    * 1e-10,  # [Pa^-1], isentropic compressibility
                    "density": 998.2,  # [kg m^-3]
                    "viscosity": 1.002e-3,  # [Pa s], absolute viscosity
                }
            ),
        },
        "grid_type": "simplex",
        "time_manager": pp.TimeManager(
            dt_init=dt,
            # dt_min_max=(0.01, 0.5),
            schedule=[0, 3, 6],
            iter_max=25,
            constant_dt=True,
        ),
        "units": pp.Units(kg=1e10),
        "meshing_arguments": {
            "cell_size": (0.1 * XMAX / cell_size_multiplier),
        },
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
            "nl_convergence_tol_res": 1e-6,
            "nl_divergence_tol": 1e8,
            "max_iterations": 25,
        },
    )

    print(model.simulation_name())
    # pp.plot_grid(
    #     model.mdg,
    #     cell_value=model.pressure_variable,
    #     vector_value=model.displacement_variable,
    #     alpha=0.5,
    # )


def load_linear_system(config: dict, mat_idx: int):

    model = make_model(config)
    model.prepare_simulation()
    model.assemble_linear_system()

    data = load_data(f"../stats/{model.simulation_name()}.json")

    print(get_gmres_iterations(data)[mat_idx])
    mat, rhs, state, iterate, dt = load_matrix_rhs_state_iterate_dt(data, mat_idx)

    model.linear_system = mat, rhs
    model.equation_system.set_variable_values(iterate, iterate_index=0)
    model.equation_system.set_variable_values(state, time_step_index=0)  # 1

    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.ad_time_step.set_value(dt)
    model.bmat.mat = mat

    return model