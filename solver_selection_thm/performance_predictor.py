import numpy as np
from sklearn.base import clone

FAIL_REWARD = -99
EPSGREEDY_EXPECTATION = 100


class IncrementalRefitModel:
    def __init__(self, model):
        self.model = model
        self.X = []
        self.y = []

    def fit(self, X, y):
        self.X = X.tolist()
        self.y = y.tolist()
        self.model.fit(X, y)

    def partial_fit(self, X, y):
        self.X.extend(X.tolist())
        self.y.extend(y.tolist())
        self.model.fit(self.X, self.y)

    def predict(self, X):
        return self.model.predict(X)


class TwoEstimators:
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def __sklearn_clone__(self):
        return TwoEstimators(
            classifier=clone(self.classifier), regressor=clone(self.regressor)
        )

    def fit(self, X, y):
        success = y >= FAIL_REWARD
        self.classifier.fit(X, success)
        self.regressor.fit(X[success], y[success])

    def partial_fit(self, X, y):
        if len(X.shape) == 1:
            X = np.array(X).reshape(1, -1)
        y = np.atleast_1d(y)
        success = y >= FAIL_REWARD
        self.classifier.partial_fit(X, success)
        if np.any(success):
            self.regressor.partial_fit(X[success], y[success])

    def predict(self, X):
        reward_estimate = np.full(X.shape[0], FAIL_REWARD - 1, dtype=float)
        success_estimate = self.classifier.predict(X)
        if not np.any(success_estimate):
            return reward_estimate

        reward_estimate[success_estimate] = self.regressor.predict(X[success_estimate])
        return reward_estimate


class EpsGreedyExplorationModel:
    def __init__(self, model, eps, eps1) -> None:
        self.model = model
        self.eps = eps
        self.eps1 = eps1

    def fit(self, X, y):
        self.model.fit(X, y)

    def partial_fit(self, X, y):
        self.model.partial_fit(X, y)

    def predict(self, X):
        if np.random.random() < self.eps:
            self.eps *= self.eps1
            result = np.zeros(X.shape[0])
            result[np.random.randint(result.size)] = EPSGREEDY_EXPECTATION
            return result
        return self.model.predict(X)


class InitialExplorationEstimator:
    def __init__(self, model, num_initial_exploration: int, batch_size: int):
        self.model = model
        self.num_initial_exploration: int = num_initial_exploration
        self.batch_size: int = batch_size
        self.X_history = []
        self.y_history = []
        self.is_ready_to_predict = False
        self.exploration_expectation = 100
        self.reward_maker = RewardEstimator()

    def select_solver(self, features: np.ndarray) -> tuple[int, float, bool]:
        if not self.is_ready_to_predict:
            return (
                np.random.randint(features.shape[0]),
                self.exploration_expectation,
                False,
            )

        expectations = self.model.predict(features)
        argmax = int(np.argmax(expectations))
        expectation = float(expectations[argmax])
        return argmax, expectation, True

    def partial_fit(
        self,
        features: np.ndarray,
        solve_time: float,
        construct_time: float,
        success: bool,
    ):
        reward = self.reward_maker.estimate_reward(solve_time, construct_time, success)
        self.X_history.append(features)
        self.y_history.append(reward)
        if (
            not self.is_ready_to_predict
            and len(self.y_history) >= self.num_initial_exploration
        ):
            self.model.fit(np.array(self.X_history), np.array(self.y_history))
            self.is_ready_to_predict = True
            self.X_history.clear()
            self.y_history.clear()

        if self.is_ready_to_predict and len(self.y_history) >= self.batch_size:
            self.model.partial_fit(np.array(self.X_history), np.array(self.y_history))
            self.X_history.clear()
            self.y_history.clear()


class PerformancePredictorRandom:
    def __init__(self, num_solvers: int):
        self.reward_maker = RewardEstimator()
        self.num_solvers: int = num_solvers
        self.exploration_expectation = 100.0

    def select_solver(self, features: np.ndarray) -> tuple[int, float, bool]:
        return np.random.randint(self.num_solvers), self.exploration_expectation, True

    def partial_fit(
        self,
        features: np.ndarray,
        solve_time: float,
        construct_time: float,
        success: bool,
    ):
        pass


class RewardEstimator:
    def __init__(self):
        self.worst_known_reward: float = FAIL_REWARD - 1

    def estimate_reward(self, solve_time: float, construct_time: float, success: bool):
        if success:
            reward = -np.log(construct_time + solve_time)
            # self.worst_known_reward = min(self.worst_known_reward, reward)
        else:
            reward = -2 * abs(self.worst_known_reward)
        return reward


def concatenate_characteristics_solvers(
    characteristics: np.ndarray, solvers: np.ndarray, solver_in_use_idx: int | None
) -> np.ndarray:
    solver_reused_flag = np.zeros((solvers.shape[0], 1))
    if solver_in_use_idx is not None:
        solver_reused_flag[solver_in_use_idx] = 1

    characteristics = np.broadcast_to(
        characteristics, (solvers.shape[0], characteristics.size)
    )
    return np.concatenate([characteristics, solver_reused_flag, solvers], axis=1)
