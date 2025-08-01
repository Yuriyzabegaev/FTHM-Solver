import numpy as np
import porepy as pp

POROSITY_MINIMAL = 1e-10
TEMPERATURE_STEADY = 273 + 120
SOURCE_RATE_VOLUMETRIC = 1e-1
TEMPERATURE_INJECTION = 273 + 40

SCALE = 1e6


class SPE10Model(pp.MassAndEnergyBalance):
    def __init__(self, params):
        self._spe10_perm = np.load("solver_selection_thm/spe10_data/spe_perm.npy")
        self._spe10_perm *= SCALE
        self._spe10_phi = np.load("solver_selection_thm/spe10_data/spe_phi.npy")
        self._spe10_phi = np.maximum(self._spe10_phi, POROSITY_MINIMAL)

        x_slice = params["x_slice"]
        z_slice = params["z_slice"]
        y_slice = slice(0, 100)
        self._spe10_perm = self._spe10_perm[:, x_slice, y_slice, z_slice]
        self._spe10_phi = self._spe10_phi[x_slice, y_slice, z_slice]
        self._spe10_perm = self._spe10_perm[0]  # only x component - isotropic.
        perm_threshold = 1e-15 * SCALE
        self._perm_estimate = np.sum(self._spe10_perm > perm_threshold)
        self._perm_estimate /= np.prod(self._spe10_perm.shape)

        self._spe10_phi = np.transpose(self._spe10_phi, (2, 1, 0))
        self._spe10_perm = np.transpose(self._spe10_perm, (2, 1, 0))

        super().__init__(params)

    def set_domain(self):
        cell_size_x = 6.096 / self.units.m
        cell_size_y = 3.048 / self.units.m
        cell_size_z = 3.048 / self.units.m
        domain_shape = self._spe10_phi.shape
        self._domain = pp.Domain(
            {
                "xmin": 0,
                "xmax": cell_size_x * domain_shape[0],
                "ymin": 0,
                "ymax": cell_size_y * domain_shape[1],
                "zmin": 0,
                "zmax": cell_size_z * domain_shape[2],
            }
        )

    def meshing_arguments(self) -> dict:
        return {
            "cell_size_x": 6.096 / self.units.m,
            "cell_size_y": 3.048 / self.units.m,
            "cell_size_z": 3.048 / self.units.m,
        }

    def grid_type(self):
        return "cartesian"

    def porosity(self, subdomains):
        return pp.ad.DenseArray(self._spe10_phi.ravel(order="f"))

    def permeability(self, subdomains):
        isotropic_perm = pp.ad.DenseArray(self._spe10_perm.ravel(order="f"))
        return self.isotropic_second_order_tensor(subdomains, isotropic_perm)

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        ones = np.ones(sd.num_cells)
        return ones * self.units.convert_units(TEMPERATURE_STEADY, "K")

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self.reference_variable_values.pressure

    def simulation_name(self) -> str:
        return self.params["folder_name"]

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, sides.all_bf, "dir")
        return bc

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.all_bf, "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, sides.all_bf, "dir")

    def bc_values_temperature(self, bg):
        return self.units.convert_units(TEMPERATURE_STEADY, "K") * np.ones(bg.num_cells)

    def locate_inlet(self, subdomains):
        xwell, ywell, zwell = self.params["inlet_placement"]
        return self.locate_wells(subdomains, wells=[(0.5, 0.5, 0.5)])

    def locate_outlet(self, subdomains):
        xwell, ywell, zwell = self.params["outlet_placement"]
        # 8 corners
        outlets = self.locate_wells(
            subdomains,
            wells=[
                (0.1, 0.1, 0.1),
                (0.1, 0.1, 0.9),
                (0.1, 0.9, 0.1),
                (0.1, 0.9, 0.9),
                (0.9, 0.1, 0.1),
                (0.9, 0.1, 0.9),
                (0.9, 0.9, 0.1),
                (0.9, 0.9, 0.9),
            ],
        )
        return outlets / 8

    def locate_wells(self, subdomains, wells: list[tuple[float, float, float]]):
        xmax = self._domain.bounding_box["xmax"]
        ymax = self._domain.bounding_box["ymax"]
        zmax = self._domain.bounding_box["zmax"]

        # We have only one subdomain
        assert len(subdomains) == 1
        x, y, z = subdomains[0].cell_centers
        src = np.zeros(x.size)

        # well position
        for xwell, ywell, zwell in wells:
            source_loc_x = xmax * xwell
            source_loc_y = ymax * ywell
            source_loc_z = zmax * zwell
            source_loc = np.argmin(
                (x - source_loc_x) ** 2
                + (y - source_loc_y) ** 2
                + (z - source_loc_z) ** 2
            )
            src[source_loc] = 1

        return src

    def fluid_source_rate(self):
        return self.units.convert_units(SOURCE_RATE_VOLUMETRIC, "m^3 * s^-1")

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        rate = self.fluid_source_rate()

        inj_loc = self.locate_inlet(subdomains)
        inj_density = self.fluid.reference_component.density

        prod_density = self.fluid.density(subdomains)
        # prod_loc = self.locate_outlet(subdomains)

        return (
            super().fluid_source(subdomains)
            + pp.ad.DenseArray(inj_loc * inj_density) * rate
            # - pp.ad.DenseArray(prod_loc) * prod_density * rate
        )

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        cv = self.fluid.components[0].specific_heat_capacity
        t_ref = self.reference_variable_values.temperature
        rate = self.fluid_source_rate()

        inj_density = self.fluid.reference_component.density
        inj_loc = self.locate_inlet(subdomains)
        t_inj = self.units.convert_units(TEMPERATURE_INJECTION, "K")

        prod_density = self.fluid.density(subdomains)
        # prod_loc = self.locate_outlet(subdomains)
        t_prod = self.temperature(subdomains)

        return (
            super().energy_source(subdomains)
            + pp.ad.DenseArray(inj_loc * cv * (t_inj - t_ref) * inj_density) * rate
            # - pp.ad.DenseArray(prod_loc * cv) * (t_prod - t_ref) * prod_density * rate
        )


def solid_params():
    return pp.SolidConstants(
        # IMPORTANT
        residual_aperture=1e-3,  # [m]
        # LESS IMPORTANT
        shear_modulus=1.2e10,  # [Pa]
        lame_lambda=1.2e10,  # [Pa]
        dilation_angle=5 * np.pi / 180,  # [rad]
        normal_permeability=1e-3,
        # granite
        biot_coefficient=0.47,  # [-]
        density=2683.0,  # [kg * m^-3]
        porosity=5e-2,  # [-]
        friction_coefficient=0.577,  # [-]
        # Thermal
        specific_heat_capacity=720.7,
        thermal_conductivity=0.1,  # Diffusion coefficient
        thermal_expansion=9.66e-6,
    )


def fluid_params():
    return pp.FluidComponent(
        compressibility=4.559 * 1e-10 * SCALE,  # [Pa^-1], fluid compressibility
        density=998.2,  # [kg m^-3]
        viscosity=1.002e-3,  # [Pa s], absolute viscosity
        # Thermal
        specific_heat_capacity=4182.0,  # Вместимость
        thermal_conductivity=0.5975,  # Diffusion coefficient
        thermal_expansion=2.068e-4,  # Density(T)
    )


def run(model: pp.PorePyModel, params):
    dt = 0.01 * pp.DAY
    tm = pp.TimeManager(
        dt_init=dt,
        schedule=[0, 1e4 * pp.DAY],
        # schedule=[0, 2 * dt],
        iter_max=newton_max_iters,
        constant_dt=False,
        recomp_max=10,
        dt_min_max=[0.001 * pp.DAY, 1000 * pp.DAY],
    )
    model.time_manager = tm
    pp.run_time_dependent_model(model, params)


newton_max_iters = 20
params = {
    "folder_name": "tmp",
    "material_constants": {
        "solid": solid_params(),
        "fluid": fluid_params(),
    },
    "reference_variable_values": pp.ReferenceVariableValues(
        pressure=3.5e7 / SCALE,  # [Pa]
        temperature=273 + 20,
    ),
    # "units": pp.Units(kg=1e10),
    "max_iterations": newton_max_iters,
    "nl_convergence_tol": 1e-5,
    "nl_convergence_tol_res": 1e-7,
    "progressbars": False,
    "prepare_simulation": False,
    "setup": {
        "solver": "direct",
    },
    "inlet_placement": [0.5, 0.3, 0.5],
    "outlet_placement": [0.5, 0.7, 0.5],
    "x_slice": slice(0, None),
    "z_slice": slice(0, None),
}

Z_SLICES = [slice(0, 20), slice(10, 30), slice(20, 40), slice(30, 50), slice(40, 60)]
X_SLICES = [slice(0, 42), slice(21, 63), slice(42, 84)]


def simulation_name(params: dict) -> str:
    name = "spe10"
    try:
        x_slice: slice = params["x_slice"]
        z_slice: slice = params["z_slice"]
    except KeyError:
        pass
    else:
        name = f"{name}_xslice_{x_slice.start}_{x_slice.stop}_zslice_{z_slice.start}_{z_slice.stop}"
    return name


if __name__ == "__main__":
    # run with the direct solver
    import logging
    import sys

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    model = SPE10Model(params)
    model.prepare_simulation()

    print("Running")
    run(model, params)
