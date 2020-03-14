
###############################################
import pickle
import time
from audioop import reverse

from hyperopt import fmin, tpe, hp, STATUS_OK,Trials,trials_from_docs
import numpy as np
import random
from scipy import stats
import pandas as pd
from sklearn.cluster import KMeans

# def objective(args):
#     x= args
#     print("X is {} and loss is {}".format(x,x[0] ** 2))
#     return {'loss': x[0] ** 2, 'status': STATUS_OK }
#
#
# space  = [hp.uniform('x',-10,10)
# ]
# # trial = Trials()
# trial = pickle.load(open("trial_16_outof100.p",'rb'))
#
# best,trials_new = fmin(objective,
#     space=space,
#     algo=tpe.suggest,
#     max_evals=116,
#     trials=trial,
#     rstate=np.random.RandomState(10),
#
#                        )
#
# print(best)
# pickle.dump(trials_new, open('trial_100_16initial.p', 'wb'))
#################################################################
from datetime import timedelta
from matplotlib import pyplot as plt
import matplotlib
def find_n_initial(trial, N, good, bad):
    """
    This method sort, first the bad points and then the good points and works better than the Vs the reverse one.
    """
    losses = trial.losses()
    losses = [abs(i) for i in losses]
    losses = np.array(losses)
    losses_index = np.argsort(losses)

    losses_index = list(losses_index)
    losses = list(losses)

    selected_list = []
    for i, v in enumerate(losses_index):
        if i >= N:
            break
        else:
            if i < N - good:
                selected_list.append(v)
            else:
                Bests = losses_index[-good:]
                selected_list = selected_list + Bests
                break

    new_trial = []
    for i in selected_list:
        new_trial.append(trial.trials[i])

    empty_trial = Trials()
    trial_merged = trials_from_docs(list(empty_trial) + new_trial)

    for i, v in enumerate(trial_merged.trials):

        need_to_change = trial_merged.trials[i]['tid']

        trial_merged.trials[i]['tid'] = i
        trial_merged.trials[i]['misc']['tid'] = i
        for key in v['misc']['idxs']:
            if v['misc']['idxs'][str(key)] == [need_to_change]:
                trial_merged.trials[i]['misc']['idxs'][str(key)] = [i]
    return trial_merged


def find_n_initial1(trial, N, good, bad):
    '''
    This method sort the trial based on, good points.
    it means it picks the goods first and then the rest(bads), slower in compare the another version
    '''

    losses = trial.losses()
    losses = [abs(i) for i in losses]
    losses = np.array(losses)
    losses_index = np.argsort(losses)

    losses_index = np.flip(losses_index)

    losses_index = list(losses_index)
    losses = list(losses)

    selected_list = []

    for i, v in enumerate(losses_index):
        if i >= N:
            break
        else:
            if i < N - bad:
                selected_list.append(v)
            else:
                Bads= losses_index[-bad:]
                selected_list = selected_list + Bads
                break

    new_trial = []
    for i in selected_list:
        new_trial.append(trial.trials[i])

    empty_trial = Trials()
    trial_merged = trials_from_docs(list(empty_trial) + new_trial)

    for i, v in enumerate(trial_merged.trials):

        need_to_change = trial_merged.trials[i]['tid']

        trial_merged.trials[i]['tid'] = i
        trial_merged.trials[i]['misc']['tid'] = i
        for key in v['misc']['idxs']:
            if v['misc']['idxs'][str(key)] == [need_to_change]:
                trial_merged.trials[i]['misc']['idxs'][str(key)] = [i]
    return trial_merged





def find_n_initial_random(trial, N):
    """
    This method pick random history out of big trial
    """
    losses = trial.losses()
    losses = [abs(i) for i in losses]
    losses = np.array(losses)

    selected_points = random.sample(range(len(losses)), N)

    new_trial = []
    for i, v in enumerate(trial.trials):
        if i in selected_points:
            new_trial.append(v)


    empty_trial = Trials()
    trial_merged = trials_from_docs(list(empty_trial) + new_trial)

    for i, v in enumerate(trial_merged.trials):

        need_to_change = trial_merged.trials[i]['tid']

        trial_merged.trials[i]['tid'] = i
        trial_merged.trials[i]['misc']['tid'] = i
        for key in v['misc']['idxs']:
            if v['misc']['idxs'][str(key)] == [need_to_change]:
                trial_merged.trials[i]['misc']['idxs'][str(key)] = [i]
    return trial_merged


# trial_bigsearchspace_5000 = pickle.load(open("/home/dfki/Desktop/Thesis/hyperopt/results_onserver/ashkan_server/bigsearchspace/trial_bigsearchspace_5000.p","rb"))
# trial_1000_new = find_n_initial(trial_bigsearchspace_5000,1000,7,993)
# pickle.dump(trial_1000_new, open('/home/dfki/Desktop/Thesis/hyperopt/results/madeup_trials/trial_1000_new_outof5000_1.p', 'wb'))

def trial_utils(trial, start, end):
    losses = trial.losses()
    losses = [abs(i) for i in losses]
    losses = np.array(losses)
    fail_config_index = np.where(losses==0)[0]
    number_failconfig = len(fail_config_index)
    number_all_try = len(losses)

    best_indices = np.argwhere(losses == np.amin(losses))
    best_indices = best_indices.flatten().tolist()

    losses = np.delete(losses,fail_config_index)
    avg_score = losses[start:end].mean()
    standard_deviation = losses[start:end].std()
    max_start_end = losses[start:end].max()

    best_score_id = trial.best_trial['tid']
    best_score = abs(trial.best_trial['result']['loss'])
    print('STD: {}'.format(standard_deviation))
    print('Best score:{} \n best score id:{} \n Average score[{},{}]:{} \n number of all try: {} \n number of fail try:{}'.format(best_score, best_score_id, start, end,
                                                                                avg_score,number_all_try,number_failconfig))
    print("Best score in [{},{}]:{}".format(start,end,max_start_end))
    print("-----------")
    return avg_score,standard_deviation,max_start_end


def time_tracker_plot(times, plot_label, xlabel, ylabel, show_plot=True):
    # print(times)
    time_keeper = []
    iteration = len(times) - 1
    for i in range(iteration):
        if times[i][0] == '0start':
            elapsedTime = times[i + 1][1] - times[i][1]
            time_keeper.append(timedelta.total_seconds(elapsedTime))


        elif (times[i][0] == 'end') or (times[i][0] == 'end_fail'):
            elapsedTime = times[i + 1][1] - times[i][1]
            time_keeper.append(timedelta.total_seconds(elapsedTime))

    time_keeper.append(timedelta.total_seconds(elapsedTime))
    print("total time point finding is {}".format(np.array(time_keeper).sum()))
    print("mean time for each configuration finding {}".format(np.array(time_keeper).mean()))
    # print(time_keeper)
    if show_plot:
        matplotlib.rcParams.update({'font.size': 22})

        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 20
        fig_size[1] = 8
        plt.plot(time_keeper, label='{}'.format(plot_label))
        plt.grid(True)
        plt.xlabel('{}'.format(xlabel))
        plt.ylabel('{}'.format(ylabel))
        plt.legend(loc=3)
        plt.show()


def find_n_histogram_points(trial, full_budget, n_bin, plot=False):
    budget_per_bin = int(full_budget / n_bin)

    losses = trial.losses()
    losses = [abs(i) for i in losses]
    losses = np.array(losses)
    losses_index = np.argsort(losses)
    #find index of more 0.5 accuracies
    valuable_index=[]
    valuable_points=[]
    for index,value in enumerate(losses):
        if value >= 0.5:
            valuable_index.append(index)
            valuable_points.append(value)


    print("Size of the History is {}".format(len(losses)))
    print("Size of atleast 50 accuracy is {}".format(len(valuable_index)))
    print("we need to select {} for each bin".format(budget_per_bin))
    print("Best point accuracy is {}".format(losses[losses_index[-1]]))

    selected_index = []

    def select_points(binmember):
        print(len(binmember))

        if len(binmember) == 0:
            return len(binmember)
        elif len(binmember) < budget_per_bin:
            print("change bin size")
            raise Exception
        elif len(binmember) == budget_per_bin:
            for item in binmember:
                index = np.where(losses == item)[0][0]
                selected_index.append(index)
        else:
            sampling = random.choices(binmember, k=budget_per_bin)
            for item1 in sampling:
                index1 = np.where(losses == item1)[0][0]
                selected_index.append(index1)

        print("selected number {}".format(len(selected_index)))
        return len(binmember)

    out = stats.binned_statistic(valuable_points, statistic=select_points, bins=n_bin, values=valuable_points)

    # if number of point is still not enough
    # if len(selected_index) < full_budget:
    #     diff = full_budget - len(selected_index)
    #     Bests = losses_index[-diff:]
    #     selected_index = list(selected_index) + list(Bests)
    print("Number of Selected points is {}".format(len(selected_index)))
    if plot:
        plt.hist(losses, bins=n_bin)
        plt.xlabel('Accuracy')
        plt.ylabel('N - points')
        plt.grid(True)
        plt.show()

    # build the new trial
    new_trial = []
    for i in selected_index:
        new_trial.append(trial.trials[i])

    empty_trial = Trials()
    trial_merged = trials_from_docs(list(empty_trial) + new_trial)

    for i, v in enumerate(trial_merged.trials):

        need_to_change = trial_merged.trials[i]['tid']

        trial_merged.trials[i]['tid'] = i
        trial_merged.trials[i]['misc']['tid'] = i
        for key in v['misc']['idxs']:
            if v['misc']['idxs'][str(key)] == [need_to_change]:
                trial_merged.trials[i]['misc']['idxs'][str(key)] = [i]

    return trial_merged


def find_n_special_points(trial, N, strategy):
    losses = trial.losses()
    losses = [abs(i) for i in losses]
    losses = np.array(losses)
    losses_index = np.argsort(losses)
    if strategy == 'BEST':
        selected_points = losses_index[-N:]
    elif strategy == 'WORST':
        selected_points = losses_index[:N]
    else:
        print("the strategy is not in list [Best,WORST]")


    new_trial = []
    for i, v in enumerate(trial.trials):
        if i in selected_points:
            new_trial.append(v)

    empty_trial = Trials()
    trial_merged = trials_from_docs(list(empty_trial) + new_trial)

    for i, v in enumerate(trial_merged.trials):

        need_to_change = trial_merged.trials[i]['tid']

        trial_merged.trials[i]['tid'] = i
        trial_merged.trials[i]['misc']['tid'] = i
        for key in v['misc']['idxs']:
            if v['misc']['idxs'][str(key)] == [need_to_change]:
                trial_merged.trials[i]['misc']['idxs'][str(key)] = [i]
    return trial_merged


def remove_zero_trial(trial):
    losses = trial.losses()
    losses = [abs(i) for i in losses]
    losses = np.array(losses)
    fail_config_index = np.where(losses <=0.5)[0] # 0.47778473091364204
    number_failconfig = len(fail_config_index)
    print('Number of fail_point is {}'.format(number_failconfig))


    new_trial = []
    for i, v in enumerate(trial.trials):
        if i not in fail_config_index:
            new_trial.append(v)

    empty_trial = Trials()
    trial_merged = trials_from_docs(list(empty_trial) + new_trial)

    for i, v in enumerate(trial_merged.trials):

        need_to_change = trial_merged.trials[i]['tid']

        trial_merged.trials[i]['tid'] = i
        trial_merged.trials[i]['misc']['tid'] = i
        for key in v['misc']['idxs']:
            if v['misc']['idxs'][str(key)] == [need_to_change]:
                trial_merged.trials[i]['misc']['idxs'][str(key)] = [i]
    return trial_merged


def vector_builder(trial):


    features = trial.trials[0]['misc']['vals'].keys()
    d={}
    # d['acc'] = []
    for ii in features:
        d[ii] =[]


    for index, each_trial in enumerate(trial.trials):
        # d['acc'].append(abs(each_trial['result']['loss']))
        for i, x in enumerate(each_trial['misc']['vals']):

            if len(each_trial['misc']['vals'][x]) == 0:
                d[x].append(0.0)
            else:
                d[x].append(each_trial['misc']['vals'][x][0])

    dd = pd.DataFrame.from_dict(d)
    vector = dd.values
    print('shape vector is {}'.format(vector.shape))
    return vector



def specialindex_trial_builder(trial,selected_index):
    # build the new trial
    new_trial = []
    for i in selected_index:
        new_trial.append(trial.trials[i])

    empty_trial = Trials()
    trial_merged = trials_from_docs(list(empty_trial) + new_trial)

    for i, v in enumerate(trial_merged.trials):

        need_to_change = trial_merged.trials[i]['tid']

        trial_merged.trials[i]['tid'] = i
        trial_merged.trials[i]['misc']['tid'] = i
        for key in v['misc']['idxs']:
            if v['misc']['idxs'][str(key)] == [need_to_change]:
                trial_merged.trials[i]['misc']['idxs'][str(key)] = [i]

    return trial_merged


def selecet_index_base_kmeans(X, k, min_member):
    '''
    X: np.array
    k: number of k in kmeans
    min_member: number of sample should take out of each cluster
    '''
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = range(0, X.shape[0])
    cluster_map['cluster'] = kmeans.labels_

    selected_index = []
    for i in range(k):
        l = cluster_map[cluster_map.cluster == i].index
        if len(l) <= min_member:
            selected_index = list(l) + list(selected_index)
        else:
            sampling = random.choices(l, k=min_member)
            selected_index = list(selected_index) + list(sampling)
        l = []

    return selected_index







def ploter(x,y, plot_label, xlabel, ylabel):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 20
    fig_size[1] = 8
    plt.plot(x,y,label='{}'.format(plot_label))
    plt.xticks(x)
    plt.yticks(range(65,99))
    plt.grid(True)
    plt.xlabel('{}'.format(xlabel))
    plt.ylabel('{}'.format(ylabel))
    plt.legend(loc=3)
    plt.show()

#
# import pickle
# trial_3 = pickle.load(open("/home/dfki/Desktop/Thesis/openml_test/pickel_files/3/trial_3.p", "rb"))
# # trial_1035in_histogram5bin = find_n_histogram_points(trial_3, 1035, 5, plot=True)
#
# # good_trial = find_n_initial(trial=trial_3,N=4000,good=15,bad=3987)
# a= vector_builder(trial_3)
# print(a.shape)
# #save the result
# # pickle.dump(trial_1035in_histogram5bin, open('/home/dfki/Desktop/Thesis/hyperopt/result_openml/mylaptop/3/automatic/new/cluster/trial_1035in_histogram5bin.p', 'wb'))
