from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


class ProblemContext:
    def to_array(self) -> np.ndarray: ...


class PerformancePredictorEpsGreedy:
    def __init__(
        self,
        num_solvers: int,
        samples_before_fit: int = 10,
        exploration: float = 0.5,
        exploration_decrease_rate: float = 0.9,
    ) -> None:
        self.num_solvers: int = num_solvers
        self.exploration_expectation = 100.0
        self.exploration = np.full(num_solvers, exploration)
        self.exploration_decrease_rate: float = exploration_decrease_rate
        self.samples_before_fit: int = samples_before_fit
        self.is_ready_to_predict: bool = False

        self.regressor = make_pipeline(
            RobustScaler(),
            GradientBoostingRegressor(),
        )

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """Select optimal parameters based on the performance prediction."""
        sample_prediction = self.regressor.predict(features)

        argmax = np.argmax(sample_prediction)
        expectation = sample_prediction[argmax]
        return argmax, expectation

    def random_choice(self, features: np.ndarray) -> tuple[int, float]:
        """Select optimal parameters based on exploration."""
        choice = np.random.randint(features.shape[0])
        return choice, self.exploration_expectation

    def select_solver(self, features: np.ndarray) -> tuple[int, float, bool]:
        greedy = self.is_ready_to_predict and (np.random.random() > self.exploration)

        if greedy:
            choice, expectation = self.predict(features=features)
        else:
            self.exploration *= self.exploration_decrease_rate
            choice, expectation = self.random_choice(features)
        return choice, expectation, greedy

    def fit(self, features: np.ndarray, rewards: np.ndarray):
        self.regressor.fit(features, rewards)
        self.is_ready_to_predict = features.shape[0] >= self.samples_before_fit


# Problems:
# Save/load -> ignoring
# Flag to reuse solver
# fit


from solver_selection_thm.solver_space import (
    CategoricalChoices,
    NumericalChoices,
    SolverSpace,
)

nd = 3
contact = [0]
intf = [1, 2]
mech = [3, 4]
flow = [5, 6, 7]
temp = [8, 9, 10]

solver_space_scheme = {
    "block_type": "PetscKSPScheme",
    "petsc_options": {
        "ksp_monitor": None,
        "ksp_rtol": 1e-12,
    },
    "compute_eigenvalues": False,
    "preconditioner": {
        "block_type": "PetscFieldSplitScheme",
        "groups": contact,
        "block_size": nd,
        "fieldsplit_options": {
            "pc_fieldsplit_schur_precondition": "selfp",
        },
        "elim_options": {
            "pc_type": "pbjacobi",
        },
        "keep_options": {
            "mat_schur_complement_ainv_type": "blockdiag",
        },
        "complement": {
            "block_type": "PetscFieldSplitScheme",
            "groups": intf,
            "elim_options": {
                "pc_type": "ilu",
            },
            "fieldsplit_options": {
                "pc_fieldsplit_schur_precondition": "selfp",
            },
            "complement": {
                "block_type": "PetscFieldSplitScheme",
                "groups": mech,
                "elim_options": CategoricalChoices(
                    [
                        {
                            "pc_type": "hmg",
                            "hmg_inner_pc_type": "hypre",
                            "hmg_inner_pc_hypre_type": "boomeramg",
                            "hmg_inner_pc_hypre_boomeramg_strong_threshold": 0.7,
                            "mg_levels_ksp_type": "richardson",
                            "mg_levels_ksp_max_it": 2,
                            "mg_levels_pc_type": "sor",
                        },
                        {
                            "pc_type": "gamg",
                            "pc_gamg_threshold": NumericalChoices(
                                [0.01, 0.001, 0.0001]
                            ),
                        },
                    ]
                ),
                "block_size": nd,
                "invert": {
                    "block_type": "fs_analytical_slow_new",
                    "p_mat_group": 5,
                    "p_frac_group": 6,
                    "groups": flow + temp,
                },
                "complement": {
                    "block_type": CategoricalChoices(
                        [
                            {
                                "block_type": "PetscCompositeScheme",
                                "groups": flow + temp,
                                "solvers": {
                                    0: {
                                        "block_type": "PetscFieldSplitScheme",
                                        "fieldsplit_options": {
                                            "pc_fieldsplit_type": "additive",
                                        },
                                        "elim_options": {
                                            "pc_type": "hypre",
                                            "pc_hypre_type": "boomeramg",
                                            "pc_hypre_boomeramg_strong_threshold": 0.7,
                                        },
                                        "complement": {
                                            "block_type": "PetscFieldSplitScheme",
                                            "groups": temp,
                                            "elim_options": {
                                                "pc_type": "none",
                                            },
                                        },
                                    },
                                    1: {
                                        "block_type": "PetscFieldSplitScheme",
                                        "groups": flow + temp,
                                        "python_pc": {
                                            "block_type": "PcPythonPermutation",
                                            "permutation_type": "pt_permutation",
                                            "p_groups": flow,
                                            "t_groups": temp,
                                            "block_size": 2,
                                        },
                                        "elim_options": {
                                            "python_pc_type": "ilu",
                                        },
                                    },
                                },
                            },
                            {
                                "block_type": "PetscFieldSplitScheme",
                                "groups": flow + temp,
                                "python_pc": {
                                    "block_type": "PcPythonPermutation",
                                    "permutation_type": "pt_permutation",
                                    "p_groups": flow,
                                    "t_groups": temp,
                                    "block_size": 2,
                                },
                                "elim_options": {
                                    "python_pc_type": "hypre",
                                    "python_pc_hypre_type": "boomeramg",
                                    "python_pc_hypre_boomeramg_strong_threshold": 0.7,
                                    "python_pc_hypre_boomeramg_P_max": 16,
                                },
                            },
                        ]
                    ),
                },
            },
        },
    },
}


class RewardEstimator:
    def __init__(self):
        self.worst_known_reward: float = -np.log(50)

    def estimate_reward(self, solve_time: float, construct_time: float, success: bool):
        if success:
            reward = -np.log(construct_time + solve_time)
            self.worst_known_reward = min(self.worst_known_reward, reward)
        else:
            reward = -2 * abs(self.worst_known_reward)
        return reward


solver_space = SolverSpace(
    solver_space_scheme=solver_space_scheme, solver_scheme_builders={}
)
solvers = solver_space.all_decisions_encoding
performance_predictor = PerformancePredictorEpsGreedy(num_solvers=solvers.shape[0])

reward_estimator = RewardEstimator()

solver_in_use_idx: int | None = None

features_history: list[np.ndarray] = []
rewards_history: list[float] = []
choice_history: list[int] = []

np.random.seed(42)

for time_step in range(100):

    context = np.array([5, 6, 7.0]) + (time_step % 4)
    context_size = context.size

    solver_reused_flag = np.zeros((solvers.shape[0], 1))
    if solver_in_use_idx is not None:
        solver_reused_flag[solver_in_use_idx] = 1

    context = np.broadcast_to(context, (solvers.shape[0], context_size))
    features = np.concatenate([context, solver_reused_flag, solvers], axis=1)

    choice, expectation, greedy = performance_predictor.select_solver(features=features)

    solver_config = solver_space.config_from_decision(decision=solvers[choice])
    # solver_scheme = solver_space.build_solver_scheme(config=solver_config)

    solve_time = 0.5 + 0.5 * np.cos(choice / (solvers.shape[0] - 1) * 2 * np.pi)
    if solver_in_use_idx == choice:
        construct_time = 1e-2
    else:
        construct_time = 0.5 * np.sin(choice / (solvers.shape[0] - 1) * np.pi)
    success = choice != 4

    reward = reward_estimator.estimate_reward(
        solve_time=solve_time, construct_time=construct_time, success=success
    )
    features_history.append(features[choice])
    rewards_history.append(reward)
    choice_history.append(choice)

    performance_predictor.fit(
        np.array(features_history), rewards=np.array(rewards_history)
    )
    solver_in_use_idx = choice

pass
