import numpy as np
import pandas as pd
from limited_data_experiment import load_experiments_data_thm, make_pandas

import solver_selection_thm.dashboard as dashboard
from solver_selection_thm.solver_space import explain_decisions

ALL_RUNS = [100, 101, 102, 103, 104]

sim_data_random, perf_data_random, solver_selector = load_experiments_data_thm(
    runs=ALL_RUNS, case='random', dir="stats/"
)

num_solvers = solver_selector.solver_space.all_decisions_encoding.shape[0]

df_sim, df_perf = make_pandas(
    sim_data=sim_data_random,
    perf_data=perf_data_random,
    seq_ids=ALL_RUNS,
)

df_perf = df_perf[df_perf.reward > -200]


solver_space = solver_selector.solver_space
decision_names, decision_ranges = explain_decisions(solver_space)
decision_names

perf_dict_for_pandas = []
for row in df_perf.itertuples(index=False):
    decision_vec = solver_space.all_decisions_encoding[row.decision_idx]
    perf_dict_for_pandas.append(
        {name: val for name, val in zip(decision_names, decision_vec)}
    )

df_perf = df_perf.reset_index()
df_perf_new = pd.concat([df_perf, pd.DataFrame(perf_dict_for_pandas)], axis=1)
df_perf_new["time"] = np.exp(-df_perf_new.reward)
df_perf_new = df_perf_new.drop(
    ["features", "seq_id", "sim_idx", "reward", "expectation", "decision_idx"], axis=1
)


column_dtypes = {x: bool for x in decision_names[: solver_space.num_category_choices]}
column_dtypes |= {x: float for x in decision_names[solver_space.num_category_choices :]}
df_perf_new = df_perf_new.astype(column_dtypes)

dashboard.make_dashboard(df_perf_new, "time")
