import porepy as pp
import numpy as np
from matplotlib import pyplot as plt
from porepy.models.constitutive_laws import GravityForce, CubicLawPermeability

from stats import StatisticsSavingMixin
from thermal.thm_solver import THMSolver

XMAX = 2000
YMAX = 1000
ZMAX = 1000
DAY = float(24 * 60 * 60)


class ModelTHM(THMSolver, CubicLawPermeability, pp.Thermoporomechanics):
    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self.units.convert_units(273 + 120, "K")

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self.reference_variable_values.pressure

    def simulation_name(self) -> str:
        return "solver_selection"

    def biot_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        try:
            return getattr(self, "_biot")
        except AttributeError:
            self._biot = pp.ad.Scalar(self.solid.biot_coefficient, "biot_coefficient")
            return self._biot

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, sides.bottom, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.all_bf, "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.all_bf, "neu")

    def bc_values_temperature(self, bg):
        return self.units.convert_units(273 + 120, "K") * np.ones(bg.num_cells)

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        bc_values = np.zeros((self.nd, boundary_grid.num_cells))
        # rho * g * h
        # 2683 * 10 * 3000
        val = self.units.convert_units(8e7, units="Pa")
        cell_volumes = boundary_grid.cell_volumes
        bc_values[2, sides.top] = -val * cell_volumes[sides.top]
        bc_values[1, sides.south] = +val * cell_volumes[sides.south] * 1.2
        bc_values[1, sides.north] = -(val * cell_volumes[sides.north]) * 1.2
        bc_values[0, sides.west] = +val * cell_volumes[sides.west] * 0.8
        bc_values[0, sides.east] = -val * cell_volumes[sides.east] * 0.8
        return bc_values.ravel("F")

    def locate_well(self, subdomains, xwell, ywell, zwell):
        xmax = self._domain.bounding_box["xmax"]
        ymax = self._domain.bounding_box["ymax"]
        zmax = self._domain.bounding_box["zmax"]
        ambient = [sd for sd in subdomains if sd.dim == self.nd]
        fractures = [sd for sd in subdomains if sd.dim == self.nd - 1]
        lower = [sd for sd in subdomains if sd.dim <= self.nd - 2]
        x, y, z = np.concatenate([sd.cell_centers for sd in fractures], axis=1)
        src_frac = np.zeros(x.size)

        # injection
        source_loc_x = xmax * xwell
        source_loc_y = ymax * ywell
        source_loc_z = zmax * zwell
        source_loc = np.argmin(
            (x - source_loc_x) ** 2 + (y - source_loc_y) ** 2 + (z - source_loc_z) ** 2
        )
        src_frac[source_loc] = 1

        zeros_ambient = np.zeros(sum(sd.num_cells for sd in ambient))
        zeros_lower = np.zeros(sum(sd.num_cells for sd in lower))
        return np.concatenate([zeros_ambient, src_frac, zeros_lower])

    def fluid_source_rate(self):
        try:
            return getattr(self, "_source_rate")
        except AttributeError:
            self._source_rate = pp.ad.Scalar(0, "source_rate")
            return self._source_rate

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        rate = self.fluid_source_rate()

        inj_loc = self.locate_well(subdomains, xwell=0.3, ywell=0.5, zwell=0.5)
        inj_density = self.fluid.reference_component.density

        prod_density = self.fluid.density(subdomains)
        prod_loc = self.locate_well(subdomains, xwell=0.7, ywell=0.5, zwell=0.5)

        return (
            super().fluid_source(subdomains)
            + pp.ad.DenseArray(inj_loc * inj_density) * rate
            - pp.ad.DenseArray(prod_loc) * prod_density * rate
        )

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        cv = self.fluid.components[0].specific_heat_capacity
        t_ref = self.reference_variable_values.temperature
        rate = self.fluid_source_rate()

        inj_density = self.fluid.reference_component.density
        inj_loc = self.locate_well(subdomains, xwell=0.3, ywell=0.5, zwell=0.5)
        t_inj = self.units.convert_units(273 + 40, "K")

        prod_density = self.fluid.density(subdomains)
        prod_loc = self.locate_well(subdomains, xwell=0.7, ywell=0.5, zwell=0.5)
        t_prod = self.temperature(subdomains)

        return (
            super().energy_source(subdomains)
            + pp.ad.DenseArray(inj_loc * cv * (t_inj - t_ref) * inj_density) * rate
            - pp.ad.DenseArray(prod_loc * cv) * (t_prod - t_ref) * prod_density * rate
        )

    def set_domain(self):
        self._domain = pp.Domain(
            {
                "xmin": 0,
                "ymin": 0,
                "zmin": 0,
                "xmax": XMAX,
                "ymax": YMAX,
                "zmax": ZMAX,
            }
        )

    def set_fractures(self):
        fracs = [
            # injection cluster
            [[0.2, 0.4, 0.4, 0.2], [0.45, 0.45, 0.45, 0.45], [0.3, 0.3, 0.6, 0.6]],
            [[0.225, 0.49, 0.49, 0.225], [0.4, 0.4, 0.8, 0.8], [0.4, 0.7, 0.7, 0.4]],
            [[0.225, 0.49, 0.49, 0.225], [0.4, 0.4, 0.8, 0.8], [0.6, 0.3, 0.3, 0.6]],
            # production cluster
            [[0.6, 0.8, 0.8, 0.6], [0.55, 0.55, 0.55, 0.55], [0.3, 0.3, 0.6, 0.6]],
            [[0.51, 0.775, 0.775, 0.51], [0.3, 0.3, 0.7, 0.7], [0.3, 0.6, 0.6, 0.3]],
            [[0.51, 0.775, 0.775, 0.51], [0.3, 0.3, 0.7, 0.7], [0.7, 0.4, 0.4, 0.7]],
        ]
        fracs = np.array(fracs)
        fracs[:, 0] *= XMAX
        fracs[:, 1] *= YMAX
        fracs[:, 2] *= ZMAX
        self._fractures = [
            pp.PlaneFracture(frac, check_convexity=True) for frac in fracs
        ]


def solid_params(biot):
    return pp.SolidConstants(
        # IMPORTANT
        permeability=1e-13,  # [m^2]
        residual_aperture=1e-3,  # [m]
        # LESS IMPORTANT
        shear_modulus=1.2e10,  # [Pa]
        lame_lambda=1.2e10,  # [Pa]
        dilation_angle=5 * np.pi / 180,  # [rad]
        normal_permeability=1e-3,
        # granite
        biot_coefficient=biot,  # [-]
        density=2683.0,  # [kg * m^-3]
        porosity=5e-2,  # [-]
        friction_coefficient=0.577,  # [-]
        # Thermal
        specific_heat_capacity=720.7,
        thermal_conductivity=0.1,  # Diffusion coefficient
        thermal_expansion=9.66e-6,
    )


newton_max_iters = 20
params = {
    "meshing_arguments": {
        "cell_size": (0.05 * XMAX),
        "cell_size_boundary": (0.1 * XMAX),
    },
    "folder_name": "vis_4_thm",
    "grid_type": "simplex",
    "material_constants": {
        "solid": solid_params(biot=0.47),
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
        temperature=273 + 20,
    ),
    "units": pp.Units(kg=1e10),
    "time_manager": pp.TimeManager(
        dt_init=0.001 * DAY,
        schedule=[0, 10 * DAY],
        iter_max=newton_max_iters,
        constant_dt=False,
        recomp_max=1,
    ),
    "max_iterations": newton_max_iters,
    "nl_convergence_tol": 1e-5,
    "nl_convergence_tol_res": 1e-8,
    "progressbars": True,
    "prepare_simulation": False,
    "setup": {
        "solver": "CPR",
    },
}
model = ModelTHM(params)
model.prepare_simulation()


def initialize(model: pp.PorePyModel):
    model._biot.set_value(0.0)
    model._source_rate.set_value(0.0)
    tm = pp.TimeManager(
        dt_init=1 * DAY,
        schedule=[0, 2 * DAY],
        iter_max=newton_max_iters,
        constant_dt=False,
        dt_min_max=[0.5 * DAY, 1.5 * DAY],
        recomp_max=1,
    )
    model.params["material_constants"]["time_manager"] = tm
    model.time_manager = tm
    pp.run_time_dependent_model(model, params)


def run(model: pp.PorePyModel):
    model._biot.set_value(float(model.solid.biot_coefficient))
    model._source_rate.set_value(model.units.convert_units(1e-1, "m^3 * s^-1"))
    dt = 0.01 * pp.DAY
    tm = pp.TimeManager(
        dt_init=dt,
        schedule=[0, 10000 * pp.DAY],
        iter_max=newton_max_iters,
        constant_dt=False,
        recomp_max=1,
    )
    model.params["material_constants"]["time_manager"] = tm
    model.time_manager = tm
    pp.run_time_dependent_model(model, params)


print("Initialising")
initialize(model)
print("Running")
run(model)
