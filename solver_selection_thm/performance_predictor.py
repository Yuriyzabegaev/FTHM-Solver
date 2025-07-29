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

        # This is the same but with L1 regularization
        self.passive_agressive_regressor = SGDRegressor(
            penalty="l1",
            random_state=42,
            learning_rate="pa1",
            loss="epsilon_insensitive",
            epsilon=0.1,
        )
        self.transform_pipeline = make_pipeline(
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            StandardScaler(),
        )
        self.full_pipeline = make_pipeline(
            self.transform_pipeline, self.passive_agressive_regressor
        )

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """Select optimal parameters based on the performance prediction."""
        sample_prediction = self.full_pipeline.predict(features)

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
                self.full_pipeline.fit(
                    np.array(self.features),
                    np.array(self.rewards),
                )
                # assert False, "You should not be here"

        return *self.predict(features=features), True

    def partial_fit(self, features: np.ndarray, reward: float):
        self.features.append(features)
        self.rewards.append(reward)
        if self.is_ready_to_predict:
            self.passive_agressive_regressor.partial_fit(
                self.transform_pipeline.transform(np.array(features).reshape(1, -1)),
                np.atleast_1d(reward),
            )


# Problems:
# Save/load -> ignoring
# Flag to reuse solver
# fit


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
