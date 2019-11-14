import openml
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
runs = [2083190, 2083596,2083543,2083187,2083188,2083110,2088190,2083169]
# runs = [2083187,2083188,1860342,2083023]
# runs=[2083190]

def run_to_dic(run_id):
    run_downloaded = openml.runs.get_run(run_id)
    setup_id = run_downloaded.setup_id
    flowid = run_downloaded.flow_id
    flow = openml.flows.get_flow(flowid)
    flow_component = flow.components.keys()
    setup = openml.setups.get_setup(setup_id)
    last_dic = {}
    for component in flow_component:
        for hyperparameter in setup.parameters.values():
            if hyperparameter.parameter_name == 'steps':
                pass
            else:
                if str(component).lower() in str(hyperparameter.full_name).lower():
                    last_dic['{}__{}'.format(component, hyperparameter.parameter_name)] = hyperparameter.value

    print(last_dic)
    print("$$$$$$$$$$$")
    return last_dic

list_runs=[]
for i in runs:
    list_runs.append(run_to_dic(i))

def make_param_space_pure(preprosessor,model):

    # param_space_pure = {
    #     '{}__copy'.format(preprosessor): [True, False],
    #     '{}__iterated_power'.format(preprosessor):['auto', 0, 10],
    #     '{}__n_components'.format(preprosessor): [None, 1, 19],
    #     '{}__random_state'.format(preprosessor): [None],
    #     '{}__svd_solver'.format(preprosessor):['auto'],
    #     '{}__tol'.format(preprosessor): [0.0, 0.5],
    #     '{}__whiten'.format(preprosessor): [True, False],
    #
    #     '{}__bootstrap'.format(model): [True],
    #     # '{}__class_weight'.format(model):[None],
    #     '{}__criterion'.format(model): ["gini", "entropy"],
    #     '{}__max_depth'.format(model): [9,10,11, 15, 20,None],
    #     '{}__max_features'.format(model): ['auto', 'sqrt', 'log2',0.15,0.25,0.3, 0.45,0.4],
    #     '{}__max_leaf_nodes'.format(model): [None, 10],
    #     # '{}__min_impurity_split'.format(model):[1e-7],
    #     '{}__min_samples_leaf'.format(model): range(1, 19),
    #     '{}__min_samples_split'.format(model): [2,3,4,5,7,9, 0.5,16],
    #     '{}__min_weight_fraction_leaf'.format(model): [0.0, 0.5],
    #     '{}__n_estimators'.format(model): range(10, 600),
    #     '{}__oob_score'.format(model): [True, False],
    #     '{}__random_state'.format(model):[None,3, 5],
    #     '{}__verbose'.format(model): [0, 1],
    #     '{}__warm_start'.format(model):[True, False]
    # }

    #new experiment for 7 november
    range_10 = ["auto"] +list(range(1, 10))
    range_20_none = [None] +list(range(1, 20))
    range_300_2000 = list(range(299, 2000))
    range_2_1000 = [None] +list(range(2, 1000))
    range2_20 = list(range(2, 20))
    range_20 = list(range(1, 20))
    range_01 = ['auto', 'sqrt', 'log2'] + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


    param_space_pure = {
        # '{}__copy'.format(preprosessor): [True, False],
        '{}__iterated_power'.format(preprosessor):range_10,
        '{}__n_components'.format(preprosessor): range_20_none,
        # '{}__random_state'.format(preprosessor): [None],
        '{}__svd_solver'.format(preprosessor):['auto','full','randomized'],
        '{}__tol'.format(preprosessor): [0.0, 0.5],
        '{}__whiten'.format(preprosessor): [True, False],

        # '{}__bootstrap'.format(model): [True],
        # '{}__class_weight'.format(model):[None],
        '{}__criterion'.format(model): ["gini", "entropy"],
        '{}__max_depth'.format(model): range_2_1000,
        '{}__max_features'.format(model): range_01,
        # '{}__max_leaf_nodes'.format(model): [None, 10],
        # '{}__min_impurity_split'.format(model):[1e-7],
        '{}__min_samples_leaf'.format(model): range_20,
        '{}__min_samples_split'.format(model): range2_20,
        '{}__min_weight_fraction_leaf'.format(model): [0.0, 0.5],
        '{}__n_estimators'.format(model): range_300_2000,
        '{}__oob_score'.format(model): [True, False],
        # '{}__random_state'.format(model):[None,3, 5],
        # '{}__verbose'.format(model): [0, 1],
        # '{}__warm_start'.format(model):[True, False]
    }
    return param_space_pure
param_space_pure = make_param_space_pure('pca','randomforestclassifier')

def change_dic_hyperoptobj(param_space_pure, openml_dic):
    new_dic = {}
    for key, value in openml_dic.items():
        if key in param_space_pure.keys():
            if type(param_space_pure[key]) == list:
                try:
                    value = int(value)
                except:
                    try:
                        value = float(value)
                    except:
                        if value in ['True', 'False']:
                            try:
                                if value == 'True':
                                    value = 1
                                    new_dic[key] = 1
                                else:
                                    value = 0
                                    new_dic[key] = 0
                            except:
                                pass
                        if value =='None':
                            value = 0
                            new_dic[key] = value

                if type(value) == str:

                    new_dic[key] = param_space_pure[key].index(value)

                if key not in new_dic:
                    new_dic[key] = param_space_pure[key].index(value)
            elif type(param_space_pure[key]) == range:
                list_range = list(param_space_pure[key])
                new_dic[key] = list_range.index(int(value))
    new_dic['randomforestclassifier__min_weight_fraction_leaf'] = 0.0

    if 'randomforestclassifier__random_state' in new_dic:
        del new_dic['randomforestclassifier__random_state']
    if 'pca__random_state' in new_dic:
        del new_dic['pca__random_state']


    return new_dic







new_list_runs = []
for index in range(len(list_runs)):
    new_list_runs.append(change_dic_hyperoptobj(param_space_pure, list_runs[index]))

print(new_list_runs)




import pickle
pickle.dump( new_list_runs, open( "limited_version_of8sample.p", "wb"))
