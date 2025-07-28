import porepy as pp
import numpy as np


class MyModel(pp.SinglePhaseFlow):
    def solve_linear_system(self):
        vals = super().solve_linear_system()
        vals[:] = np.nan
        return vals


time_manager = pp.TimeManager(schedule=[0, 1], dt_init=0.1, dt_min_max=(0.0001, 0.2))
pp.run_time_dependent_model(
    MyModel({"time_manager": time_manager}),
    {"progressbars": False},  # try with True
)
