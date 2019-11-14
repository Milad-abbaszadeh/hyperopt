# import pickle
# import openml
# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# import pickle
# from hyperopt import space_eval
# from openml import tasks,datasets
# from sklearn.metrics import accuracy_score
# import hyperopt
#
# # PIK = "Base_line_100_run_3.dat"
# #
# # with open(PIK, "rb") as f:
# #     list_new = pickle.load(f)
# #     trials = list_new[3]
#
#
# dataset = openml.datasets.get_dataset(31)
# X, y, categorical_indicator, attribute_names = dataset.get_data(
#     dataset_format='array',
#     target=dataset.default_target_attribute
# )
#
# task = openml.tasks.get_task(31)
# data = dataset.get_data()
#
#
#
# def acc_pipeline(params):
#
#
#     result = {}
#     full_result = 0
#     for i in range(10):
#         train_indices, test_indices = task.get_train_test_split_indices(fold=i)
#         X_train = X[train_indices]
#         y_train = y[train_indices]
#         X_test = X[test_indices]
#         y_test = y[test_indices]
#
#         pca = PCA()
#         rf = RandomForestClassifier(random_state=0)
#         pip = Pipeline(steps=[('pca', pca), ('randomforestclassifier', rf)])
#         pip.set_params(**params)
#
#         pip.fit(X_train, y_train)
#         predicted= pip.predict(X_test)
#         full_result = full_result + abs(accuracy_score(y_test, predicted))
#
#     return full_result/10
#
#
# def f(params):
#     global best
#     acc = acc_pipeline(params)
#     if acc > best:
#         best = acc
#     print('new best:', best, params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
# param_space = {
#     'pca__copy': hp.choice('pca__copy', [True, False]),
#     'pca__iterated_power': hp.choice('pca__iterated_power', ['auto', 0, 10]),
#     'pca__n_components': hp.choice('pca__n_components', [None,1, 19]),
#     'pca__random_state': hp.choice('pca__random_state', [None]),
#     'pca__svd_solver': hp.choice('pca__svd_solver', ['auto']),
#     'pca__tol': hp.choice('pca__tol', [0.0, 0.5]),
#     'pca__whiten': hp.choice('pca__whiten', [True, False]),
#
#     'randomforestclassifier__bootstrap': hp.choice('randomforestclassifier__bootstrap', [True]),
#     'randomforestclassifier__criterion': hp.choice('randomforestclassifier__criterion', ["gini", "entropy"]),
#     'randomforestclassifier__max_depth': hp.choice('randomforestclassifier__max_depth', [9,10,11, 15, 20,None]),
#     'randomforestclassifier__max_features': hp.choice('randomforestclassifier__max_features', ['auto', 'sqrt', 'log2',0.15,0.25,0.3,0.45]),
#     'randomforestclassifier__max_leaf_nodes': hp.choice('randomforestclassifier__max_leaf_nodes',  [None, 10]),
#     'randomforestclassifier__min_samples_leaf': hp.choice('randomforestclassifier__min_samples_leaf', range(1, 10)),
#     'randomforestclassifier__min_samples_split': hp.choice('randomforestclassifier__min_samples_split', [2,3,4,5,7,9, 0.5]),
#     'randomforestclassifier__min_weight_fraction_leaf': hp.choice('randomforestclassifier__min_weight_fraction_leaf', [0.0, 0.5]),
#     'randomforestclassifier__n_estimators': hp.choice('randomforestclassifier__n_estimators', range(300, 600)),
#     'randomforestclassifier__oob_score': hp.choice('randomforestclassifier__oob_score', [True, False]),
#     'randomforestclassifier__random_state': hp.choice('randomforestclassifier__random_state', [None,3, 5]),
#     'randomforestclassifier__verbose': hp.choice('randomforestclassifier__verbose', [0, 1]),
#     'randomforestclassifier__warm_start': hp.choice('randomforestclassifier__warm_start', [True, False])
# }
#
# openml_loaded = pickle.load( open( "new_list_runs.p", "rb" ) )
#
#
# best = 0
# trials = Trials()
# # trials = pickle.load(open("trials.p", "rb"))
#
# best,trials_inside = fmin(f, param_space, algo=hyperopt.rand.suggest, max_evals=100, trials=trials)
#
# print('best:')
# print(space_eval(param_space,best))
# print("best Acc {}".format(trials_inside.best_trial['result']['loss']))
# print("losses {}".format(trials_inside.losses()))
#
#
#
# pickle.dump(trials, open("trials.p", "wb"))

import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK

def objective(args):
    x,y,z = args
    print(x,y,z)
    return {'loss': x ** 2+ y , 'status': STATUS_OK }


space  = [hp.quniform('x',2,19,1),hp.uniform('y',1,10),hp.choice('z',['A','B'])
]

best,trials_new = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=5,
    points_to_evaluate=[{'x':3,'y':2,'z':0}]
                       )

print(best)
print(trials_new.best_trial)


# from hyperopt.pyll.stochastic import sample
# print (best)
# for i in range(100):
#     print(sample(space))