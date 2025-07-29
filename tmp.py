from solver_selection_thm.physics import ModelTHM, initialize, run, params
from solver_selection_thm.selector import SolverSelector
from solver_selection_thm.solver_space import CategoricalChoices, NumericalChoices
from solver_selection_thm.performance_predictor import (
    PerformancePredictorPassiveAgressive,
    PerformancePredictorEpsGreedy,
    RewardEstimator,
)
from solver_selection_thm.solver_space import SolverSpace
from solver_selection_thm.pp_binding import (
    KNOWN_SOLVER_COMPONENTS_THM,
    SolverSelectionMixinTH,
)
from solver_selection_thm.test_solver_selector import make_solver_space_scheme_fthm


solver_space = SolverSpace(
    solver_space_scheme=make_solver_space_scheme_fthm(nd=3),
    solver_scheme_builders=KNOWN_SOLVER_COMPONENTS_THM,
)
num_solvers = solver_space.all_decisions_encoding.shape[0]
performance_predictor = PerformancePredictorPassiveAgressive(num_solvers=num_solvers)
solver_selector = SolverSelector(
    reward_estimator=RewardEstimator(),
    solver_space=solver_space,
    performance_predictor=performance_predictor,
)

x = solver_space.decision_tree.list_possible_solvers()
print(x)
