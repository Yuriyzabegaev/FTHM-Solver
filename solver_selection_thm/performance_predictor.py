import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from solver_selection_thm.solver_space import SolverSpace


class ProblemContext:
    def to_array(self) -> np.ndarray: ...


class PerformancePredictorEpsGreedy:
    def __init__(
        self,
        samples_before_fit: int = 10,
        exploration: float = 0.5,
        exploration_decrease_rate: float = 0.9,
    ) -> None:
        self.exploration_expectation = 100.0
        self.exploration: float = exploration
        self.exploration_decrease_rate: float = exploration_decrease_rate
        self.samples_before_fit: int = samples_before_fit
        self.is_ready_to_predict: bool = False

        self.regressor = make_pipeline(
            StandardScaler(),
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


# Problems:
# Save/load -> ignoring
# Flag to reuse solver
# fit


class RewardEstimator:
    def estimate_reward(self, prediction, performance):
        pass
