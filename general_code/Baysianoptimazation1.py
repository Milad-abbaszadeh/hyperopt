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
from sklearn.model_selection import RandomizedSearchCV
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

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

        pip.set_params(**params)

        pip.fit(X_train, y_train)
        predicted= pip.predict(X_test)
        full_result = full_result + abs(accuracy_score(y_test, predicted))

    return full_result/10

param_space = {
    'pca__copy': (0,1.9),
    'pca__iterated_power': (0, 10),
    'pca__whiten': (0,1.9),
    'pca__n_components': (1,20),
    'pca__svd_solver':(0,3),
    'pca__tol': (0.0, 0.9999),


    'randomforestclassifier__criterion': (0,1.9),
    'randomforestclassifier__max_depth': (1,1000),
    'randomforestclassifier__max_features': (0,1),
    'randomforestclassifier__max_leaf_nodes': (2,19),
    'randomforestclassifier__min_samples_leaf': (0,0.5),
    'randomforestclassifier__min_samples_split': (0,1),
    'randomforestclassifier__min_weight_fraction_leaf': (0,0.5),
    'randomforestclassifier__n_estimators':(300, 2000),
    'randomforestclassifier__oob_score': (0,1.9),
    'randomforestclassifier__verbose':(0,1.9),

}

def function_to_be_optimized(pca__copy,pca__iterated_power,pca__whiten,pca__n_components,pca__tol,pca__svd_solver,
                             randomforestclassifier__criterion,randomforestclassifier__max_leaf_nodes,randomforestclassifier__oob_score,randomforestclassifier__max_depth,randomforestclassifier__max_features,
                             randomforestclassifier__min_samples_leaf,randomforestclassifier__min_samples_split,randomforestclassifier__min_weight_fraction_leaf,randomforestclassifier__n_estimators,randomforestclassifier__verbose):
    a = {}
    a['pca__copy'] = True if int(pca__copy)>0 else False
    a['pca__iterated_power'] = int(pca__iterated_power)
    a['pca__whiten'] = True if int(pca__whiten)>0 else False
    a['pca__n_components'] = int(pca__n_components)

    # if int(pca__svd_solver) == 0:
    #     a['pca__svd_solver'] ='auto'
    # elif int(pca__svd_solver) == 1:
    #     a['pca__svd_solver'] = 'full'
    # elif int(pca__svd_solver) ==2:
    #     a['pca__svd_solver']='arpack'
    # elif int(pca__svd_solver) == 3:
    #     a['pca__svd_solver'] = 'randomized'

    a['pca__tol'] = pca__tol

    a['randomforestclassifier__criterion'] ='gini' if int(randomforestclassifier__criterion)>0 else 'entropy'
    a['randomforestclassifier__oob_score'] =True if int(randomforestclassifier__oob_score)>0 else False
    a['randomforestclassifier__max_depth'] = int(randomforestclassifier__max_depth)
    a['randomforestclassifier__max_features'] = randomforestclassifier__max_features
    a['randomforestclassifier__max_leaf_nodes'] = int(randomforestclassifier__max_leaf_nodes)
    a['randomforestclassifier__min_samples_leaf'] = randomforestclassifier__min_samples_leaf
    a['randomforestclassifier__min_samples_split'] = randomforestclassifier__min_samples_split
    a['randomforestclassifier__min_weight_fraction_leaf'] = randomforestclassifier__min_weight_fraction_leaf
    a['randomforestclassifier__n_estimators'] = int(randomforestclassifier__n_estimators)
    a['randomforestclassifier__verbose'] = int(randomforestclassifier__verbose)
    return acc_pipeline(a)


optimizer = BayesianOptimization(
    f=function_to_be_optimized,
    pbounds=param_space,
    verbose=0,
    random_state=1,
)
logger = JSONLogger(path="./logs_openml.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

# optimizer.set_gp_params(normalize_y=True)

optimizer.probe(
    params={'pca__copy':1,'pca__iterated_power':'auto','pca__n_components':'auto' ,'pca__svd_solver':'auto','pca__whiten':0,'rf__criterion':0, 'rf__max_depth':10, 'rf__max_features':0.45000000000000001, 'rf__min_impurity_split':1e-07, 'rf__min_samples_leaf':6, 'rf__min_samples_split':7, 'rf__min_weight_fraction_leaf':0.0,'rf__n_estimators':512,'rf__oob_score':1, 'rf__verbose':0},
    lazy=True,
)

optimizer.maximize(
    init_points=0,
    n_iter=0,
)
print(optimizer.max)