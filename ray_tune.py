import openml
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import pickle
from hyperopt import space_eval
from openml import tasks,datasets
from sklearn.metrics import accuracy_score
import hyperopt
from hyperopt.pyll.stochastic import sample
import ray
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import tpe, hp, STATUS_OK, Trials
import ray
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch


# def easy_objective(config, reporter):
#     import time
#     time.sleep(0.2)
#     assert type(config["activation"]) == str, \
#         "Config is incorrect: {}".format(type(config["activation"]))
#     for i in range(config["iterations"]):
#         reporter(
#             timesteps_total=i,
#             mean_loss=(config["height"] - 14)**2 - abs(config["width"] - 3))
#         time.sleep(0.02)
#
dataset = openml.datasets.get_dataset(31)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)

task = openml.tasks.get_task(31)

def acc_pipeline(params):
    result = {}
    full_result = 0
    for i in range(10):
        train_indices, test_indices = task.get_train_test_split_indices(fold=i)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        pca = PCA()
        rf = RandomForestClassifier(random_state=0)
        pip = Pipeline(steps=[('pca', pca), ('randomforestclassifier', rf)])

        # print("$$$$$$$$$$$$$$$$$")
        # params = check_param(params)
        # print("%%%%%%%%%%%%%%%%%%%%%5")
        # print(pip.get_params().keys())
        pip.set_params(**params)

        pip.fit(X_train, y_train)
        predicted= pip.predict(X_test)
        full_result = full_result + abs(accuracy_score(y_test, predicted))

    return full_result/10

if __name__ == "__main__":
    import argparse
    from hyperopt import hp

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init()

    # space = {
    #     "width": hp.uniform("width", 0, 20),
    #     "height": hp.uniform("height", -100, 100),
    #     "activation": hp.choice("activation", ["relu", "tanh"])
    # }

    param_space = {
        'pca__copy': hp.choice('pca__copy', [True, False]),
        'pca__iterated_power': hp.choice('pca__iterated_power', ['auto', 0, 10]),
        'pca__n_components': hp.choice('pca__n_components', [None, 'mle', 0.3, 0.5, 0.9, 5, 10, 15, 19]),
        'pca__svd_solver': hp.choice('pca__svd_solver', ['auto']),
        'pca__tol': hp.choice('pca__tol', [0.0, 0.3, 0.5, 0.8]),
        'pca__whiten': hp.choice('pca__whiten', [True, False]),

        'randomforestclassifier__bootstrap': hp.choice('randomforestclassifier__bootstrap', [True]),
        'randomforestclassifier__criterion': hp.choice('randomforestclassifier__criterion', ["gini", "entropy"]),
        'randomforestclassifier__max_depth': hp.choice('randomforestclassifier__max_depth', [5]),
        'randomforestclassifier__min_samples_leaf': hp.choice('randomforestclassifier__min_samples_leaf', range(1, 10)),
        'randomforestclassifier__min_samples_split': hp.choice('randomforestclassifier__min_samples_split',
                                                               range(2, 19)),
        'randomforestclassifier__n_estimators': hp.choice('randomforestclassifier__n_estimators', range(300, 900)),
        'randomforestclassifier__oob_score': hp.choice('randomforestclassifier__oob_score', [True, False]),

    }

    # current_best_params = [
    #     {
    #         "width": 1,
    #         "height": 2,
    #         "activation": 0  # Activation will be relu
    #     },
    #     {
    #         "width": 4,
    #         "height": 2,
    #         "activation": 1  # Activation will be tanh
    #     }
    # ]

    config = {
        "num_samples": 10 if args.smoke_test else 10,
        "config": {
            # "iterations": 10,
        },
        "stop": {
            "timesteps_total": 10
        },
    }
    openml_loaded = pickle.load(open("new_list_runs.p", "rb"))

    algo = HyperOptSearch(
        param_space,
        max_concurrent=4,
        metric="mean_loss",
        mode="min",
        points_to_evaluate=openml_loaded)
    scheduler = AsyncHyperBandScheduler(metric="mean_loss", mode="min")
    all_trials = run(acc_pipeline, search_alg=algo, scheduler=scheduler, **config)
    print(all_trials.get_best_config())