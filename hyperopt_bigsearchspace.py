from hyperopt import fmin, tpe, hp, STATUS_OK,Trials
from sklearn.preprocessing import Normalizer,scale,StandardScaler,MinMaxScaler
import openml
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score
from hyperopt import space_eval
import copy
from sklearn.decomposition import KernelPCA
from sklearn import linear_model
import pickle
import time
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import datetime
from datetime import timedelta
import temp



class run_hyperopt(object):
    def __init__(self,dataset_id,task_id):
        self.task_id = task_id
        self.dataset_id = dataset_id

        self.task = openml.tasks.get_task(self.task_id)
        self.dataset = openml.datasets.get_dataset(dataset_id)
        self.X, self.y, self.categorical_indicator, attribute_names = self.dataset.get_data(
            dataset_format='array',
            target=self.dataset.default_target_attribute
        )
        self.copy_X = copy.deepcopy(self.X)
        self.copy_y = copy.deepcopy(self.y)
        self.time_tracker=[]

    def rest_x_y(self):
        self.X = self.copy_X
        self.y = self.copy_y

    def make_search_space(self):
        search_space={
            'data_preprocessing':hp.choice('data_preprocessing',[
                {'type':'Normalizer'},
                {'type':'standard_scaler'},
                {'type':'minmaxscaler'},
                {'type':"do_noting"}
            ]),

            'feature_preprocessing':hp.choice('feature_preprocessing',[
                    {'type':'pca',
                    'iterated_power': hp.choice('pca__iterated_power', range(1,10)),
                    'n_components': hp.choice('pca__n_components',range(1,self.X.shape[1])),
                    'svd_solver':hp.choice('pca__svd_solver',['auto','full','randomized']),
                    'tol': hp.uniform('pca__tol',0,0.5),
                    'whiten': hp.choice('pca__whiten', [True, False])},

                    {'type':'kernel_pca',
                     # 'kernel':hp.choice('kernel',["linear","poly","rbf","sigmoid","cosine"]),
                     'n_components':hp.choice('n_components',range(10,self.X.shape[1]))
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
                 },

                {'type':'SGD',
                 'loss':hp.choice('loss',["log", "modified_huber", "squared_hinge", "perceptron"]),
                 'penalty':hp.choice('penalty',["l1", "l2", "elasticnet"]),
                 'alpha':hp.uniform('alpha',1e-7,1e-1),
                 'max_iter':hp.choice('max_iter',range(5,1000)),
                 'tol':hp.uniform('tol',1e-5, 1e-1)}
            ])

        }
        return search_space

    def objective(self,params):
        print("--------------------")
        start = datetime.datetime.now()
        self.time_tracker.append(start)

        print(params)
        copy_params = copy.deepcopy(params)
        step=[]
        data_preprocessing_type = params['data_preprocessing']['type']
        del params['data_preprocessing']['type']

        if data_preprocessing_type == 'Normalizer':
            scaler = Normalizer()
            self.X = scaler.fit_transform(self.X)
            # step.append(('Normalizer',norm))

        elif data_preprocessing_type == 'standard_scaler':
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
            # step.append(('scaler',scaler))

        elif data_preprocessing_type == 'minmaxscaler':
            scaler = MinMaxScaler()
            self.X = scaler.fit_transform(self.X)
            # step.append(('scaler', scaler))

        elif data_preprocessing_type == 'do_noting':
            pass


        feature_preprocessing_type = params['feature_preprocessing']['type']
        del params['feature_preprocessing']['type']

        if feature_preprocessing_type == 'pca':
            pca_params = params['feature_preprocessing']
            pca  = PCA()
            pca.set_params(**pca_params)
            pca.random_state = 0
            step.append(('pca',pca))

        elif feature_preprocessing_type == 'kernel_pca':
            kernel_pca_param = params['feature_preprocessing']
            kernel_pca = KernelPCA()
            kernel_pca.set_params(**kernel_pca_param)
            kernel_pca.random_state = 0
            step.append(('kernel_pca',kernel_pca))

        elif feature_preprocessing_type == 'do_noting':
            pass

        classifier_type = params['classifier']['type']
        del params['classifier']['type']

        if classifier_type == 'rf':
            rf_params = params['classifier']
            rf = RandomForestClassifier()
            rf.set_params(**rf_params)
            rf.random_state  = 0
            step.append(('randomforestclassifier',rf))

        elif classifier_type == 'KN':
            kn_params = params['classifier']
            kn = KNeighborsClassifier()
            kn.set_params(**kn_params)
            kn.random_state = 0
            step.append(('KN',kn))

        elif classifier_type == 'SGD':
            sgd_params = params['classifier']
            sgd = linear_model.SGDClassifier()
            sgd.set_params(**sgd_params)
            sgd.random_state = 0
            step.append(('SGD', sgd))


        pip = Pipeline(steps=step)


        aucs=[]
        accuracies =[]
        for i in range(10):
            train_indices, test_indices = self.task.get_train_test_split_indices(fold=i)
            X_train = self.X[train_indices]
            y_train = self.y[train_indices]
            X_test = self.X[test_indices]
            y_test = self.y[test_indices]

            pip.fit(X_train, y_train)
            predicted = pip.predict(X_test)

            # predict_proba = pip.predict_proba(X_test)
            # scores = np.amax(predict_proba, axis=1)

            # auc = roc_auc_score(y_test,scores)
            # aucs.append(auc)
            fold_accuracy = accuracy_score(y_test, predicted)
            accuracies.append(fold_accuracy)

        # full_auc = np.array(aucs).mean()
        accuracy = np.array(accuracies).mean()

        if 'do_noting' in (copy_params['data_preprocessing']['type']):
            pass
        else:
            self.rest_x_y()


        print(" \n Accuracy is {}".format(accuracy))
        # print("AUC is {}".format(full_auc))
        return {'loss': -accuracy, 'status': STATUS_OK}



import hyperopt.plotting


if __name__ == '__main__':

    runner = run_hyperopt(31,31)
    trials = Trials()
    trial_bigsearchspace_1000 = pickle.load(open("/home/dfki/Desktop/Thesis/hyperopt/results_onserver/ashkan_server/bigsearchspace/trial_bigsearchspace_5000.p","rb"))
    # trials = temp.find_n_initial(trial=trial_bigsearchspace_1000,N=1000,good=7,bad=993)

    # trials = trial_bigsearchspace_1000

    # trials = pickle.load(open("/home/dfki/Desktop/Thesis/hyperopt/results/madeup_trials/trial_1000_new_outof5000_1.p", "rb"))
    # tpe._default_linear_forgetting = 1000
    best,trials_inside = fmin(runner.objective, runner.make_search_space(), algo=tpe.suggest, max_evals=100, trials=trials,rstate=np.random.RandomState(10))
    print("Best Accuracy is {}\n {} \n".format(trials_inside.best_trial['result']['loss'],best))
    print(space_eval(runner.make_search_space(),best))
    temp.trial_utils(trials_inside,0,100)

    # pickle.dump(trials_inside, open('./results/result_bigsearchspace/trial_bigsearchspace_100_dataset31_1000initial_new_1.p', 'wb'))
    # pickle.dump(runner.time_tracker, open('./results/result_bigsearchspace/timetracker_bigsearchspace_100_dataset31_1000initial_new_1.p', 'wb'))

    temp.time_tracker_plot(runner.time_tracker, 'time', 'iteration', 'time(sec)}', show_plot=False)

