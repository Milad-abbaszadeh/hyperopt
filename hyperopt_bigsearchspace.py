from hyperopt import fmin, tpe, hp, STATUS_OK,Trials
from sklearn.preprocessing import Normalizer
import openml
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from get_tasks import get_task_ids
from sklearn.model_selection import cross_val_score
import numpy as np

dataset = openml.datasets.get_dataset(31)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)




search_space={
    'data_preprocessing':hp.choice('data_preprocessing',[
        # {'type':'Normalizer'},
        {'type':"do_noting"}
    ]),

    'feature_preprocessing':hp.choice('feature_preprocessing',[
            {'type':'pca',
            'iterated_power': hp.choice('pca__iterated_power', range(1,10)),
            'n_components': hp.choice('pca__n_components',range(1,20)),
            'svd_solver':hp.choice('pca__svd_solver',['auto','full','randomized']),
            'tol': hp.uniform('pca__tol',0,0.5),
            'whiten': hp.choice('pca__whiten', [True, False])},

            {'type':'fast_ica',
             'n_components':hp.choice('fast_ica_n_components',range(1,10)),
             'whiten': hp.choice('fast_ica_whiten', [True, False])
             },

            {'type':'do_noting'}

    ]),

    'classifier':hp.choice('classifier',[
        {'type':'rf',
        'criterion': hp.choice('randomforestclassifier__criterion', ["gini", "entropy"]),
        'max_depth': hp.choice('randomforestclassifier__max_depth', range(2,1000)),
        'min_samples_leaf': hp.choice('randomforestclassifier__min_samples_leaf', range(1,20)),
        'min_samples_split': hp.choice('randomforestclassifier__min_samples_split', range(2,20)),
        'min_weight_fraction_leaf': hp.uniform('randomforestclassifier__min_weight_fraction_leaf', 0.0, 0.5),
        'max_features': hp.choice('randomforestclassifier__max_features', ['auto', 'sqrt', 'log2']),
        'n_estimators': hp.choice('randomforestclassifier__n_estimators', range(10,1000)),
        'oob_score': hp.choice('randomforestclassifier__oob_score', [True, False]),
         },

        {'type':'KN',
         'n_neighbors':hp.choice('KN_n_neighbors',range(2,10)),
         'algorithm':hp.choice('KN_algorithm',['auto', 'ball_tree', 'kd_tree', 'brute'])
         }
    ])

}

def pre_objective(dataset_id):
    task_id = get_task_ids(dataset_id)
    task = openml.tasks.get_task(task_id)
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    # for i in range(10):
    #     train_indices, test_indices = task.get_train_test_split_indices(fold=i)
    #     X_train = X[train_indices]
    #     y_train = y[train_indices]
    #     X_test = X[test_indices]
    #     y_test = y[test_indices]
    #
    #     #make pipeline
    #     pip.set_params(**params)
    #
    #     pip.fit(X_train, y_train)
    #     predicted = pip.predict(X_test)
    #     full_result = full_result + abs(accuracy_score(y_test, predicted))
    #






def objective(params):
    step=[]
    data_preprocessing_type = params['data_preprocessing']['type']
    # del params['data_preprocessing']['type']
    if data_preprocessing_type == 'Normalizer':
        norm = Normalizer()
        step.append(('Normalizer',norm))
    elif data_preprocessing_type == 'do_noting':
        pass


    feature_preprocessing_type = params['feature_preprocessing']['type']
    del params['feature_preprocessing']['type']
    if feature_preprocessing_type == 'pca':
        pca_params = params['feature_preprocessing']
        pca  = PCA()
        pca.set_params(pca_params)
        step.append(('pca',pca))

    elif feature_preprocessing_type == 'fast_ica':
        fast_ica_params = params['feature_preprocessing']
        ica  = FastICA()
        ica.set_params(**fast_ica_params)
        step.append(('fast_ica',ica))

    elif feature_preprocessing_type == 'do_noting':
        pass



    classifier_type = params['classifier']['type']
    del params['classifier']['type']

    if classifier_type == 'rf':
        rf_params = params['classifier']
        rf = RandomForestClassifier()
        rf.set_params(**rf_params)
        step.append(('randomforestclassifier',rf))

    elif classifier_type == 'KN':
        kn_params = params['classifier']
        kn = KNeighborsClassifier()
        kn.set_params(**kn_params)
        step.append(('KN',kn))


    pip = Pipeline(steps=step)
    pip.fit(X,y)

    print(np.isnan(X))
    accuracy = cross_val_score(pip, X, y).mean()

    return {'loss': -accuracy, 'status': STATUS_OK}





if __name__ == '__main__':


    trials = Trials()
    best,trials_inside = fmin(objective, search_space, algo=tpe.suggest, max_evals=10, trials=trials,rstate=np.random.RandomState(10))

    # from hyperopt.pyll.stochastic import sample
    #
    # for i in range(1):
    #     print(sample(search_space))