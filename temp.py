# import pickle
# import time
# from hyperopt import fmin, tpe, hp, STATUS_OK
#
# def objective(args):
#     x,y,z = args
#     print(x,y,z)
#     return {'loss': x ** 2+ y , 'status': STATUS_OK }
#
#
# space  = [hp.quniform('x',2,19,1),hp.uniform('y',1,10),hp.choice('z',['A','B'])
# ]
#
# best,trials_new = fmin(objective,
#     space=space,
#     algo=tpe.suggest,
#     max_evals=5,
#     points_to_evaluate=[{'x':3,'y':2,'z':0}]
#                        )
#
# print(best)
# print(trials_new.best_trial)


# from hyperopt.pyll.stochastic import sample
# print (best)
# for i in range(100):
#     print(sample(space))

import openml
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


dataset = openml.datasets.get_dataset(31)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)

task = openml.tasks.get_task(31)

def acc_pipeline():
    result = {}
    full_result = 0
    for i in range(10):
        train_indices, test_indices = task.get_train_test_split_indices(fold=i)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]


        pca = PCA()
        tree = DecisionTreeClassifier(random_state=54004)
        pip = Pipeline(steps=[('pca', pca), ('tree', tree)])

        model = SVC(C=4.523994003904736,random_state=1,cache_size=200,gamma=0.033987763661532146)

        pip.fit(X_train, y_train)
        predicted= pip.predict(X_test)
        full_result = full_result + abs(accuracy_score(y_test, predicted))
    print(full_result/10)
    return full_result/10

acc_pipeline()