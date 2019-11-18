# get tasks from openml
import openml
import pandas as pd


# List of dataset IDs used for the NIPS experiment.
dataset_ids = [1049,1050,1053,1067,1068,1112,1114,1116,1120,1128,1130,1134,1138,1139,1142,1146,1161,1166,12,14,16,180,182,185,18,22,23,24,273,28,293,300,30,31,32,351,354,36,38,390,391,393,396,399,3,44,46,554,60,6,914,953]


def get_task_ids(dataset_ids):
    # return task ids of corresponding datset ids.

    # active tasks
    tasks_a = openml.tasks.list_tasks(task_type_id=1, status='active')
    tasks_a = pd.DataFrame.from_dict(tasks_a, orient="index")

    # query only those with holdout as the resampling startegy.
    tasks_a = tasks_a[(tasks_a.estimation_procedure == "33% Holdout set")]

    # deactivated tasks
    tasks_d = openml.tasks.list_tasks(task_type_id=1, status='deactivated')
    tasks_d = pd.DataFrame.from_dict(tasks_d, orient="index")

    tasks_d = tasks_d[(tasks_d.estimation_procedure == "33% Holdout set")]

    task_ids = []
    for did in dataset_ids:
        task_a = list(tasks_a.query("did == {}".format(did)).tid)
        if len(task_a) > 1:  # if there are more than one task, take the lowest one.
            task_a = [min(task_a)]
        task_d = list(tasks_d.query("did == {}".format(did)).tid)
        if len(task_d) > 1:
            task_d = [min(task_d)]
        task_ids += list(task_a + task_d)

    return task_ids  # return list of all task ids.


def main():
    task_ids = sorted(get_task_ids(dataset_ids))
    string_to_print = ''
    for tid in task_ids:
        string_to_print += str(tid) + ' '
    print(string_to_print)  # print the task ids for bash script.


if __name__ == "__main__":
    main()
