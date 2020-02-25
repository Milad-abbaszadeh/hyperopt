
###############################################
import pickle
import time
from audioop import reverse

from hyperopt import fmin, tpe, hp, STATUS_OK,Trials,trials_from_docs
import numpy as np
import random
from scipy import stats
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
    max_start_end = losses[start:end].max()

    best_score_id = trial.best_trial['tid']
    best_score = abs(trial.best_trial['result']['loss'])

    print('Best score:{} \n best score id:{} \n Average score[{},{}]:{} \n number of all try: {} \n number of fail try:{}'.format(best_score, best_score_id, start, end,
                                                                                avg_score,number_all_try,number_failconfig))
    print("Best score in [{},{}]:{}".format(start,end,max_start_end))
    print("-----------")


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
    print("Size of the History is {}".format(len(losses)))
    print("Best point accuracy is {}".format(losses[losses_index[-1]]))

    selected_index = []

    def select_points(binmember):

        if len(binmember) == 0:
            return len(binmember)
        elif len(binmember) <= budget_per_bin:
            for item in binmember:
                index = np.where(losses == item)[0][0]
                selected_index.append(index)
        else:
            sampling = random.choices(binmember, k=budget_per_bin)
            for item1 in sampling:
                index1 = np.where(losses == item1)[0][0]
                selected_index.append(index1)

        return len(binmember)

    out = stats.binned_statistic(losses, statistic=select_points, bins=n_bin, values=losses)

    # if number of point is still not enough
    if len(selected_index) < full_budget:
        diff = full_budget - len(selected_index)
        Bests = losses_index[-diff:]
        selected_index = list(selected_index) + list(Bests)

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

