from hpsklearn import HyperoptEstimator, any_classifier ,any_preprocessing
from sklearn.datasets import load_iris
from hyperopt import tpe
import numpy as np
import openml


# Download the data and split into training and test sets
from hpsklearn import random_forest,pca


dataset = openml.datasets.get_dataset(31)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)

task = openml.tasks.get_task(31)
full_result = []
best_models={}
for i in range(10):
    train_indices, test_indices = task.get_train_test_split_indices(fold=i)
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]



    estim = HyperoptEstimator(classifier=random_forest('my_rf'),
                          preprocessing=[pca('my_pca')],
                          algo=tpe.suggest,
                          max_evals=100)

    estim.fit(X_train, y_train)
    full_result.append(estim.score(X_test, y_test))
    best_models[i] = estim.best_model()
    print( estim.best_model())

print(full_result)
print(np.array(full_result).mean())
print(best_models)