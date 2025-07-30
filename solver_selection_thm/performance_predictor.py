import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import PassiveAggressiveRegressor


class OnlineSGDRegressor:
    def __init__(self, gamma=20.0, n_components=500, lr=0.001, alpha=1e-6):
        self.scaler = StandardScaler()
        self.rbf = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
        self.model = SGDRegressor(
            random_state=42,
            learning_rate="constant",
            eta0=lr,
            alpha=alpha,
            max_iter=10,  # single step per batch
            tol=None,
            penalty="l2",
        )
        self.is_initialized = False

    def partial_fit(self, X, y):
        if not self.is_initialized:
            X_scaled = self.scaler.fit_transform(X)
            X_features = self.rbf.fit_transform(X_scaled)
            self.is_initialized = True
        else:
            X_scaled = self.scaler.transform(X)
            X_features = self.rbf.transform(X_scaled)

        self.model.partial_fit(X_features, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_features = self.rbf.transform(X_scaled)
        return self.model.predict(X_features)


class PerformancePredictorEpsGreedy:
    def __init__(
        self,
        num_solvers: int,
        exploration: float = 0.5,
        exploration_decrease_rate: float = 0.9,
    ) -> None:
        self.num_solvers: int = num_solvers
        self.exploration_expectation = 100.0
        self.exploration: float = exploration
        self.exploration_decrease_rate: float = exploration_decrease_rate

        self.is_ready_to_predict: bool = False
        initial_choice_sequence = [i for i in range(self.num_solvers) for _ in range(2)]
        self.num_data_before_ready = len(initial_choice_sequence)
        self.initial_choice_sequence = iter(initial_choice_sequence)

        self.features: list[np.ndarray] = []
        self.rewards: list[float] = []

        # self.regressor = OnlineSGDRegressor()
        self.regressor = GradientBoostingRegressor(random_state=42)
        # make_pipeline(
        #     RobustScaler(),
        #     GradientBoostingRegressor(),
        # )

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """Select optimal parameters based on the performance prediction."""
        sample_prediction = self.regressor.predict(features)

        argmax = np.argmax(sample_prediction)
        expectation = sample_prediction[argmax]
        return argmax, expectation

    def random_choice(self) -> tuple[int, float]:
        """Select optimal parameters based on exploration."""
        choice = np.random.randint(self.num_solvers)
        return choice, self.exploration_expectation

    def select_solver(self, features: np.ndarray) -> tuple[int, float, bool]:
        if not self.is_ready_to_predict:
            try:
                choice = next(self.initial_choice_sequence)
                return choice, self.exploration_expectation, False
            except StopIteration:
                self.is_ready_to_predict = True
                self.regressor.fit(np.array(self.features), np.array(self.rewards))
                # assert False, "You should not be here"

        greedy = np.random.random() > self.exploration
        if greedy:
            choice, expectation = self.predict(features=features)
        else:
            self.exploration *= self.exploration_decrease_rate
            choice, expectation = self.random_choice()
        return choice, expectation, greedy

    def partial_fit(self, features: np.ndarray, reward: float):
        self.features.append(features)
        self.rewards.append(reward)
        self.is_ready_to_predict = len(self.rewards) >= self.num_data_before_ready
        if self.is_ready_to_predict:
            self.regressor.fit(np.array(self.features), np.array(self.rewards))


class PerformancePredictorPassiveAgressive:
    def __init__(self, num_solvers: int) -> None:
        self.num_solvers: int = num_solvers
        self.exploration_expectation = 100.0

        self.is_ready_to_predict: bool = False
        initial_choice_sequence = [i for i in range(self.num_solvers) for _ in range(1)]
        np.random.shuffle(initial_choice_sequence)
        # initial_choice_sequence = initial_choice_sequence[117:]

        self.num_data_before_ready = len(initial_choice_sequence)
        self.initial_choice_sequence = iter(initial_choice_sequence)

        self.features: list[np.ndarray] = []
        self.rewards: list[float] = []

        # self.passive_agressive_regressor = PassiveAggressiveRegressor(random_state=42)

        self.scaler = StandardScaler()
        self.regressor = SGDRegressor(
            penalty="l2",
            random_state=42,
            # learning_rate="pa1",  # better without
            # loss="epsilon_insensitive",
            epsilon=0.1,
        )

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """Select optimal parameters based on the performance prediction."""
        sample_prediction = self.regressor.predict(self.scaler.transform(features))

        argmax = int(np.argmax(sample_prediction))
        expectation = float(sample_prediction[argmax])
        return argmax, expectation

    def select_solver(self, features: np.ndarray) -> tuple[int, float, bool]:
        if not self.is_ready_to_predict:
            try:
                choice = next(self.initial_choice_sequence)
                return choice, self.exploration_expectation, False
            except StopIteration:
                self.is_ready_to_predict = True
                self.scaler.fit(np.array(self.features))
                self.regressor.fit(np.array(self.features), np.array(self.rewards))

        return *self.predict(features=features), True

    def partial_fit(self, features: np.ndarray, reward: float):
        self.features.append(features)
        self.rewards.append(reward)
        if self.is_ready_to_predict:
            self.scaler.partial_fit(np.array(features).reshape(1, -1))
            self.regressor.partial_fit(
                np.array(features).reshape(1, -1), np.atleast_1d(reward)
            )
            # self.passive_agressive_regressor.partial_fit(
            #     self.transform_pipeline.transform(np.array(features).reshape(1, -1)),
            #     np.atleast_1d(reward),
            # )


from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class SuccessClassifier(ClassifierMixin, BaseEstimator):
   

    def __init__(self):
        self.classes_ = np.array([False, True])
        self.scaler = StandardScaler()
        self.classifier = SGDClassifier(loss="log_loss", max_iter=100)

    def fit(self, X, y):
        classes = np.array([False, True])
        if np.all(y) or not np.any(y):
            class_weight = None
        else:
            class_weight = {
                class_: weight
                for class_, weight in zip(
                    classes, compute_class_weight("balanced", classes=classes, y=y)
                )
            }
            self.classifier.set_params(class_weight=class_weight)

        X = self.scaler.fit_transform(X)
        self.classifier.fit(X, y)
        return self

    def partial_fit(self, X, y):
        self.scaler.partial_fit(X)
        self.classifier.partial_fit(self.scaler.transform(X), y, classes=[False, True])

    def predict(self, X):
        return self.classifier.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.classifier.predict_proba(self.scaler.transform(X))

    def decision_function(self, X):
        return self.classifier.decision_function(self.scaler.transform(X))


class RewardRegressor(RegressorMixin, BaseEstimator):

    def __init__(self):
        self.scaler = StandardScaler()
        self.regressor = SGDRegressor(
            penalty="l2",
            random_state=42,
            epsilon=0.1,
        )

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.regressor.fit(X, y)
        return self

    def partial_fit(self, X, y):
        self.scaler.partial_fit(X)
        self.regressor.partial_fit(self.scaler.transform(X), y)

    def predict(self, X):
        prediction = self.regressor.predict(self.scaler.transform(X))
        return np.clip(prediction, FAIL_REWARD, -FAIL_REWARD)


class Estimator:

    def __init__(self, num_solvers: int):
        self.reward_maker = RewardEstimator()
        self.num_solvers: int = num_solvers
        self.exploration_expectation = 100.0

        self.is_ready_to_predict: bool = False
        initial_choice_list = [i for i in range(self.num_solvers) for _ in range(1)]
        np.random.shuffle(initial_choice_list)
        self.initial_choice_list = initial_choice_list

        self.initial_choice_iter = iter(initial_choice_list)

        self.features_good_and_bad = []
        self.success_good_and_bad = []
        self.features_only_good: list[np.ndarray] = []
        self.rewards_only_good: list[float] = []

        self.success_estimator = SuccessClassifier()
        self.reward_estimator = RewardRegressor()

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """Select optimal parameters based on the performance prediction."""
        success_estimate = self.success_estimator.predict(features)
        if not np.any(success_estimate):
            print("All solvers are expected to fail.")
            return np.random.randint(features.shape[0]), self.exploration_expectation

        BAD_REWARD = self.reward_maker.worst_known_reward
        reward_estimate = np.full(features.shape[0], BAD_REWARD, dtype=float)
        reward_estimate[success_estimate] = self.reward_estimator.predict(
            features[success_estimate]
        )
        argmax = int(np.argmax(reward_estimate))
        expectation = float(reward_estimate[argmax])
        return argmax, expectation

    def select_solver(self, features: np.ndarray) -> tuple[int, float, bool]:
        if not self.is_ready_to_predict:
            try:
                choice = next(self.initial_choice_iter)
                return choice, self.exploration_expectation, False
            except StopIteration:
                self.is_ready_to_predict = True
                self.success_estimator.fit(
                    np.array(self.features_good_and_bad),
                    np.array(self.success_good_and_bad),
                )
                self.reward_estimator.fit(
                    np.array(self.features_only_good), np.array(self.rewards_only_good)
                )

        return *self.predict(features=features), True

    # def partial_fit(self, features: np.ndarray, reward: float):
    def partial_fit(
        self,
        features: np.ndarray,
        solve_time: float,
        construct_time: float,
        success: bool,
    ):
        reward = self.reward_maker.estimate_reward(solve_time, construct_time, success)
        if self.is_ready_to_predict:
            features = np.array(features).reshape(1, -1)
            success = np.atleast_1d(success)
            self.success_estimator.partial_fit(features, success)
            if success:
                rewards = np.atleast_1d(reward)
                self.reward_estimator.partial_fit(features, rewards)
        else:
            self.features_good_and_bad.append(features)
            self.success_good_and_bad.append(success)
            if success:
                self.features_only_good.append(features)
                self.rewards_only_good.append(reward)


# Problems:
# Save/load -> ignoring
# Flag to reuse solver
# fit
FAIL_REWARD = -100.

class RewardEstimator:
    def __init__(self):
        self.worst_known_reward: float = FAIL_REWARD  # -np.log(50)

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
