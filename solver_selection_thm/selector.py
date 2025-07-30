from solver_selection_thm.performance_predictor import (
    PerformancePredictorEpsGreedy,
    RewardEstimator,
    concatenate_characteristics_solvers,
)
from solver_selection_thm.solver_space import SolverSchemeProtocol, SolverSpace
import numpy as np
from pickle import dump, load


class SolverSelectorHistory:
    def __init__(self):
        self.features: list[np.ndarray] = []
        self.reward: list[float] = []
        self.decision_idx: list[int] = []
        self.greedy: list[bool] = []
        self.expectation: list[float] = []

    def save(self, path: str):
        with open(path, "wb") as f:
            dump(
                (
                    self.features,
                    self.reward,
                    self.decision_idx,
                    self.greedy,
                    self.expectation,
                ),
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            (
                self.features,
                self.reward,
                self.decision_idx,
                self.greedy,
                self.expectation,
            ) = load(f)


class SolverSelector:
    def __init__(
        self,
        # reward_estimator: RewardEstimator,
        solver_space: SolverSpace,
        performance_predictor: PerformancePredictorEpsGreedy,
    ):
        self.solver_space: SolverSpace = solver_space
        self.performance_predictor: PerformancePredictorEpsGreedy
        self.performance_predictor = performance_predictor
        # self.reward_estimator: RewardEstimator = reward_estimator
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

        self.__decision_idx = decision_idx
        self.__expectation = expectation
        self.__greedy = greedy
        self.__features = features[decision_idx].copy()

        config = self.solver_space.config_from_decision(decision=decision)
        return self.solver_space.build_solver_scheme(
            config=config, porepy_model=porepy_model
        ), decision_idx

    def provide_performance_feedback(
        self, solve_time: float, construct_time: float, success: bool
    ) -> None:
        reward = self.performance_predictor.reward_maker.estimate_reward(
            solve_time=solve_time, construct_time=construct_time, success=success
        )
        # reward with and wo construct
        self.history.decision_idx.append(self.__decision_idx)
        self.history.expectation.append(self.__expectation)
        self.history.greedy.append(self.__greedy)
        self.history.features.append(self.__features)
        self.history.reward.append(reward)
        self.performance_predictor.partial_fit(
            features=self.history.features[-1], solve_time=solve_time, construct_time=construct_time, success=success
        )
        # self.history.save("solver_selection_history.npy")
