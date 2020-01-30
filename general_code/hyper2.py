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

        pca = PCA(random_state=2)
        rf = RandomForestClassifier(random_state=3)
        pip = Pipeline(steps=[('pca', pca), ('randomforestclassifier', rf)])

        pip.set_params(**params)

        pip.fit(X_train, y_train)
        predicted= pip.predict(X_test)
        full_result = full_result + abs(accuracy_score(y_test, predicted))

    return full_result/10


def f(params):
    global best
    acc = acc_pipeline(params)
    if acc > best:
        best = acc
    print('new best:', best, params)
    return {'loss': -acc, 'status': STATUS_OK}


range_10 = ["auto"] +list(range(1, 10))
range_20 = list(range(1, 20))
range_20_none = [None] +list(range(1, 20))
range_300_2000 = list(range(299, 2000))
range_2_1000 = [None] + list(range(2, 1000))
range2_20 = list(range(2, 20))
range_01 = ['auto', 'sqrt', 'log2'] + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

param_space = {
    'pca__iterated_power': hp.choice('pca__iterated_power', range_10),
    'pca__n_components': hp.choice('pca__n_components',range_20_none),
    'pca__svd_solver':hp.choice('pca__svd_solver',['auto','full','randomized']),
    'pca__tol': hp.uniform('pca__tol',0,0.5),
    'pca__whiten': hp.choice('pca__whiten', [True, False]),

    'randomforestclassifier__criterion': hp.choice('randomforestclassifier__criterion', ["gini", "entropy"]),
    'randomforestclassifier__max_depth': hp.choice('randomforestclassifier__max_depth',range_2_1000),
    'randomforestclassifier__min_samples_leaf': hp.choice('randomforestclassifier__min_samples_leaf',range_20),
    'randomforestclassifier__min_samples_split': hp.choice('randomforestclassifier__min_samples_split',range2_20),
    'randomforestclassifier__min_weight_fraction_leaf':hp.uniform('randomforestclassifier__min_weight_fraction_leaf',0.0, 0.5),
    'randomforestclassifier__max_features':hp.choice('randomforestclassifier__max_features',range_01),
    'randomforestclassifier__n_estimators': hp.choice('randomforestclassifier__n_estimators', range_300_2000),
    'randomforestclassifier__oob_score': hp.choice('randomforestclassifier__oob_score', [True, False]),

}




# openml_loaded_limited = pickle.load(open("new_list_runs_limited.p","rb"))
# openml_best = pickle.load( open("best_of_openml.p","rb"))
limited_version_of8sample = pickle.load( open("limited_version_of8sample.p","rb"))


trials = pickle.load( open("openml_input_8confings_1000_iteration.p","rb"))
best = 0
# trials = Trials()


best,trials_inside = fmin(fn=f, space=param_space, algo=tpe.suggest, max_evals=2000, trials=trials,points_to_evaluate = limited_version_of8sample,)
# best,trials_inside = fmin(f, param_space, algo=tpe.suggest, max_evals=10, trials=trials)


print('best:')
print(space_eval(param_space,best))
print("best Acc {}".format(trials_inside.best_trial['result']['loss']))
print("losses {}".format(trials_inside.losses()))

pickle.dump(trials_inside,open('openml_input_1000confings_1000_iteration.p','wb'))