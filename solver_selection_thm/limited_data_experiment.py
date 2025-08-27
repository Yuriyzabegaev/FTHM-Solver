from collections import defaultdict
import multiprocessing as mp
from itertools import count
from time import time
import json

import numpy as np
import pandas as pd
from load_experiments_data import load_experiments_data_spe
from sklearn.ensemble import GradientBoostingRegressor
from solver_selection_thm.performance_predictor import SuccessClassifier


def make_pandas(sim_data, perf_data, seq_ids):
    sim_data_dict = defaultdict(lambda: [])
    perf_data_dict = defaultdict(lambda: [])
    i = 0
    for seq_id, data_simulations, solver_selection_history_seq in zip(
        seq_ids, sim_data, perf_data
    ):
        sim_idx = -1
        for data_row in data_simulations:
            for data in data_row:
                sim_idx += 1

                for ts_idx, ts in enumerate(data):
                    for ls_idx, ls in enumerate(ts.linear_solves):
                        sim_data_dict["seq_id"].append(seq_id)
                        sim_data_dict["sim_idx"].append(sim_idx)
                        sim_data_dict["ts_idx"].append(ts_idx)
                        sim_data_dict["ls_idx"].append(ls_idx)
                        sim_data_dict["real_solve_time"].append(ls.linear_solve_time)
                        sim_data_dict["krylov_iters"].append(ls.krylov_iters)
                        sim_data_dict["petsc_converged_reason"].append(
                            ls.petsc_converged_reason
                        )
                        sim_data_dict["cfl"].append(ls.cfl)

        solver_selection_history = None
        for x in solver_selection_history_seq:
            if x is not None:
                solver_selection_history = x

        for reward_idx in range(len(solver_selection_history.reward)):
            # if solver_selection_history.features[reward_idx][5] > 1e10 or solver_selection_history.features[reward_idx][4] > 1e10:
            #     i += 1
            #     print('dropping large', i)
            #     continue
            perf_data_dict["seq_id"].append(seq_id)
            perf_data_dict["sim_idx"].append(sim_idx)
            perf_data_dict["reward"].append(solver_selection_history.reward[reward_idx])
            perf_data_dict["expectation"].append(
                solver_selection_history.expectation[reward_idx]
            )
            perf_data_dict["decision_idx"].append(
                solver_selection_history.decision_idx[reward_idx]
            )
            perf_data_dict["features"].append(
                solver_selection_history.features[reward_idx]
            )
    return pd.DataFrame(data=sim_data_dict), pd.DataFrame(data=perf_data_dict)


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
        reward_estimate = np.full(X.shape[0], FAIL_REWARD, dtype=float)
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
        np.random.seed(42)

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


def stack_features_solvers(features: np.ndarray, solvers: np.ndarray) -> np.ndarray:
    dim0 = solvers.shape[0]
    dim1 = features.shape[0]
    features = np.broadcast_to(features, (dim0, dim1))
    return np.concatenate([features, solvers], axis=1)


def top_1_normalized_score(ypred, ytrue):
    i = np.argmax(ypred)

    ytrue = np.array(ytrue)
    ytrue_not_failed = ytrue[ytrue > FAIL_REWARD]
    if len(ytrue_not_failed) == 0:
        # all failed
        return 0
    ymin = np.min(ytrue[ytrue > FAIL_REWARD])
    ymax = np.max(ytrue)
    ytrue = np.clip(ytrue, ymin, ymax)

    s = ytrue[i]
    if ymax == ymin:
        return 0.0  # degenerate case: all ytrue values are equal
    else:
        return -1.0 + 2.0 * (s - ymin) / (ymax - ymin)


def top_eps_accuracy(ypred, ytrue, eps=1e-3):
    i = np.argmax(ypred)
    return abs(ytrue[i] - np.max(ytrue)) <= eps


FAIL_REWARD = -100
EPSGREEDY_EXPECTATION = 100


def do_experiment(experiment_setup: dict, dir="./stats/"):
    sim_data_random, perf_data_random, solver_selector = load_experiments_data_spe(
        runs=RUNS_RANDOM, random_selection=True, dir=dir
    )
    num_solvers = solver_selector.solver_space.all_decisions_encoding.shape[0]

    df_sim, df_perf = make_pandas(
        sim_data=sim_data_random,
        perf_data=perf_data_random,
        seq_ids=ALL_RUNS,
    )

    X = np.stack(df_perf.features)
    X = np.clip(X, -1e10, 1e10)
    y = np.array(df_perf.reward)

    oracle = oracle = TwoEstimators(
        classifier=SuccessClassifier(),
        regressor=GradientBoostingRegressor(random_state=42),
    )
    oracle.fit(X, y)

    incremental_learning = experiment_setup["incremental_learning"]
    one_decision = experiment_setup["one_decision"]
    seq_id = experiment_setup["seq_id"]
    gamma = experiment_setup["gamma"]
    eps = experiment_setup["eps"]
    batch_size = experiment_setup["batch_size"]
    eps1 = 0.9
    experiment_idx = experiment_setup["experiment_idx"]
    experiments_total = experiment_setup["experiments_total"]
    print(f"Start experiment {experiment_idx} / {experiments_total}")

    filter_ = np.array((df_perf.seq_id == seq_id))
    X_ranking = X[filter_]
    y_ranking = y[filter_]

    num_offline = int(num_solvers * gamma)

    X_offline = X_ranking[:num_offline]
    y_offline = y_ranking[:num_offline]
    X_online = X_ranking[num_offline:]
    # y_online = y_ranking[num_offline:]

    # np.random.seed(42)
    # np.random.shuffle(X_offline)
    # np.random.shuffle(y_offline)

    online_model = EpsGreedyExplorationModel(
        eps=eps,
        eps1=eps1,
        model=TwoEstimators(
            classifier=SuccessClassifier(),
            regressor=IncrementalRefitModel(
                model=GradientBoostingRegressor(random_state=42)
            ),
        ),
    )
    online_model.fit(X_offline, y_offline)

    all_solvers = solver_selector.solver_space.all_decisions_encoding
    X_online_features = X_online[
        :, : -all_solvers.shape[1]
    ]  # only solution context (no solvers encoding)

    data_incremental = []
    batch_feedback_X = []
    batch_feedback_y = []

    prev_prediction = None

    for Xfeature in X_online_features:
        x_to_predict = stack_features_solvers(Xfeature, all_solvers)

        # online
        tpred = time()
        if prev_prediction is None:
            predictions_online = online_model.predict(x_to_predict)
        else:
            predictions_online = prev_prediction
        if one_decision:
            prev_prediction = predictions_online
        max_score_idx_online = np.argmax(predictions_online)

        tpred = time() - tpred

        # offline
        predictions_offline = oracle.predict(x_to_predict)
        # predictions_offline = np.arange(predictions_offline.size)

        # feedback
        X_feedback = x_to_predict[max_score_idx_online].reshape(1, -1)
        y_feedback = predictions_offline[max_score_idx_online]

        tfeedback = time()
        if incremental_learning:
            batch_feedback_X.append(X_feedback)
            batch_feedback_y.append(y_feedback)

            if len(batch_feedback_X) >= batch_size:
                online_model.partial_fit(
                    np.array(batch_feedback_X).squeeze(), np.array(batch_feedback_y)
                )
                batch_feedback_X = []
                batch_feedback_y = []
        tfeedback = time() - tfeedback

        # saving stats
        data_incremental.append(
            {
                "ypred": predictions_online,
                "yoracle": predictions_offline,
                "tpred": tpred,
                "tfeedback": tfeedback,
            }
        )

    data_for_pandas = experiment_setup

    ypred = np.array([max(x["ypred"]) for x in data_incremental])
    yoracle = np.array([max(x["yoracle"]) for x in data_incremental])
    yfeedback = np.array(
        [x["yoracle"][np.argmax(x["ypred"])] for x in data_incremental]
    )
    decision_id = np.array([np.argmax(x["ypred"]) for x in data_incremental])

    data_for_pandas["ypred"] = ypred
    data_for_pandas["yoracle"] = yoracle
    data_for_pandas["yfeedback"] = yfeedback
    data_for_pandas["decision_id"] = decision_id

    nts = [
        top_1_normalized_score(ypred=x["ypred"], ytrue=x["yoracle"])
        for x in data_incremental
    ]
    data_for_pandas["NTS"] = np.array(nts)

    eps = 0.24 * 0.1
    topeps = [
        top_eps_accuracy(ypred=x["ypred"], ytrue=x["yoracle"], eps=eps)
        for x in data_incremental
    ]
    data_for_pandas["TopEPS"] = np.array(topeps)

    print(f"Done experiment {experiment_idx} / {experiments_total}")
    return data_for_pandas


RUNS_RANDOM = [30, 31, 32, 33, 34]
ALL_RUNS = [f"R{x}" for x in RUNS_RANDOM]

if __name__ == "__main__":
    experiment_setups = []
    for seq_id in ALL_RUNS:
        for gamma in [
            0.25,
            0.5,
            0.75,
            1,
        ]:
            for eps in [
                0,
                0.5,
                0.9,
            ]:
                experiment_setups.append(
                    {
                        "incremental_learning": True,
                        "one_decision": False,
                        "seq_id": seq_id,
                        "gamma": gamma,
                        "eps": eps,
                        "batch_size": 128,
                    }
                )

        for one_decision in [False, True]:
            experiment_setups.append(
                {
                    "incremental_learning": False,
                    "one_decision": one_decision,
                    "seq_id": seq_id,
                    "gamma": 1,
                    "eps": 0,
                    "batch_size": 128,
                }
            )

    cnt = count()
    for experiment in experiment_setups:
        experiment["experiment_idx"] = next(cnt)
        experiment["experiments_total"] = len(experiment_setups)

    num_processes = mp.cpu_count()
    print(f"{num_processes = }")
    with mp.Pool(num_processes) as pool:
        results = pool.map(do_experiment, experiment_setups)
