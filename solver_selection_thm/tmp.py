import porepy as pp
import numpy as np

XMAX = 1000
YMAX = 1000


class MyModel(pp.MassAndEnergyBalance):
    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self.reference_variable_values.temperature

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self.reference_variable_values.pressure

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")

    def set_domain(self):
        self._domain = pp.Domain(
            {
                "xmin": 0,
                "ymin": 0,
                "xmax": XMAX,
                "ymax": YMAX,
            }
        )

    def locate_well(self, subdomains, xwell, ywell):
        xmax = self._domain.bounding_box["xmax"]
        ymax = self._domain.bounding_box["ymax"]
        x, y, z = np.concatenate([sd.cell_centers for sd in subdomains], axis=1)
        src_loc = np.zeros(x.size)

        source_loc_x = xmax * xwell
        source_loc_y = ymax * ywell
        source_loc = np.argmin((x - source_loc_x) ** 2 + (y - source_loc_y) ** 2)
        src_loc[source_loc] = 1
        return src_loc

    def fluid_source_rate(self):
        return self.units.convert_units(1e-1, "m^3 * s^-1")

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        inj_loc = self.locate_well(subdomains, xwell=0.5, ywell=0.5)
        rate = self.fluid_source_rate()
        inj_density = self.fluid.reference_component.density
        return (
            super().fluid_source(subdomains)
            + pp.ad.DenseArray(inj_loc * inj_density) * rate
        )

    # def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
    #     inj_density = self.fluid.reference_component.density
    #     inj_loc = self.locate_well(subdomains, xwell=0.5, ywell=0.5)
    #     cv = self.fluid.components[0].specific_heat_capacity
    #     t_inj = self.units.convert_units(273 + 40, "K")
    #     t_ref = self.reference_variable_values.temperature
    #     rate = self.fluid_source_rate()
    #     inj = pp.ad.DenseArray(inj_loc * cv * (t_inj - t_ref) * inj_density * rate)
    #     return super().energy_source(subdomains) + (inj) * rate


DAY = 24 * 60 * 60
newton_max_iters = 20
params = {
    "meshing_arguments": {
        "cell_size": 0.1 * XMAX,
    },
    "time_manager": pp.TimeManager(
        schedule=[0, 100 * DAY],
        dt_init=1 * DAY,
        iter_max=newton_max_iters,
    ),
    "progressbars": True,
    "reference_variable_values": pp.ReferenceVariableValues(
        pressure=3.5e7,  # [Pa]
        temperature=273 + 120,
    ),
    "material_constants": {
        "solid": pp.SolidConstants(
            # IMPORTANT
            permeability=1e-13,  # [m^2]
            residual_aperture=1e-3,  # [m]
            # LESS IMPORTANT
            shear_modulus=1.2e10,  # [Pa]
            lame_lambda=1.2e10,  # [Pa]
            dilation_angle=5 * np.pi / 180,  # [rad]
            normal_permeability=1e-6,
            # granite
            biot_coefficient=0.47,  # [-]
            density=2683.0,  # [kg * m^-3]
            porosity=5e-2,  # [-]
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
        "max_iterations": newton_max_iters,
        "nl_convergence_tol": 1e-5,
        "nl_convergence_tol_res": 1e-8,
    },
}
model = MyModel(params=params)
# model.prepare_simulation()
pp.run_time_dependent_model(model=model, params=params)
pp.plot_grid(model.mdg, cell_value="temperature", plot_2d=True)
