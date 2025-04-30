from solver_selection_thm.performance_predictor import (
    PerformancePredictorEpsGreedy,
    RewardEstimator,
    concatenate_characteristics_solvers,
)
from solver_selection_thm.solver_space import SolverSchemeProtocol, SolverSpace
import numpy as np


class SolverSelectorHistory:
    def __init__(self):
        self.features: list[np.ndarray] = []
        self.reward: list[float] = []
        self.decision_idx: list[int] = []
        self.greedy: list[bool] = []
        self.expectation: list[float] = []


class SolverSelector:
    def __init__(
        self,
        reward_estimator: RewardEstimator,
        solver_space: SolverSpace,
        performance_predictor: PerformancePredictorEpsGreedy,
    ):
        self.solver_space: SolverSpace = solver_space
        self.performance_predictor: PerformancePredictorEpsGreedy
        self.performance_predictor = performance_predictor
        self.reward_estimator: RewardEstimator = reward_estimator
        self.history = SolverSelectorHistory()

    def select_linear_solver_scheme(
        self, characteristics: np.ndarray, porepy_model, active_solver_idx: int | None
    ) -> tuple[SolverSchemeProtocol, int]:
        features = concatenate_characteristics_solvers(
            characteristics=characteristics,
            solvers=self.solver_space.all_decisions_encoding,
            solver_in_use_idx=active_solver_idx,
        )
        decision_idx, expectation, greedy = self.performance_predictor.select_solver(
            features=features
        )
        decision = self.solver_space.all_decisions_encoding[decision_idx]

        self.history.decision_idx.append(decision_idx)
        self.history.expectation.append(expectation)
        self.history.greedy.append(greedy)
        self.history.features.append(features[decision_idx])

        config = self.solver_space.config_from_decision(decision=decision)
        return self.solver_space.build_solver_scheme(
            config=config, porepy_model=porepy_model
        ), decision_idx

    def provide_performance_feedback(
        self, solve_time: float, construct_time: float, success: bool
    ) -> None:
        reward = self.reward_estimator.estimate_reward(
            solve_time=solve_time, construct_time=construct_time, success=success
        )
        self.history.reward.append(reward)
        self.performance_predictor.partial_fit(
            features=self.history.features[-1], reward=reward
        )
