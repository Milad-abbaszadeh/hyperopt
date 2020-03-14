import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import temp
import pickle
def trial_builder_kmeans(all_trials,num_clusters):
    X = temp.vector_builder(all_trials)

    # XX = []
    # for row in X:
    #     row = [float(xx) for xx in row]
    #     XX.append(row)
    # X = np.array(XX)
    selected_index = kmeans_point_selector(X,num_clusters)
    trial = temp.specialindex_trial_builder(all_trials,selected_index)
    print(len(trial.trials))
    return trial


def kmeans_point_selector(tf_matrix,num_clusters):

    m_km = KMeans(n_clusters=num_clusters,random_state=0)
    m_km.fit(tf_matrix)
    m_clusters = m_km.labels_.tolist()

    centers = np.array(m_km.cluster_centers_)

    closest_data = []
    for i in range(num_clusters):
        center_vec = centers[i]
        data_idx_within_i_cluster =[]
        data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]
        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster) , centers.shape[1] ) )
        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = tf_matrix[data_idx]
            one_cluster_tf_matrix[row_num] = one_row

        if one_cluster_tf_matrix.shape[0] == 0:
            # closest_data.append(np.where(center_vec  == aa for aa in tf_matrix)[0][0])
            pass
        else:
            closest, _ = pairwise_distances_argmin_min([center_vec], one_cluster_tf_matrix)
            closest_idx_in_one_cluster_tf_matrix = closest[0]
            closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
            closest_data.append(closest_data_row_num)

    closest_data = list(set(closest_data))
    # print(closest_data)
    # assert len(closest_data) == num_clusters
    return closest_data

# import pickle
# trial_3 = pickle.load(open("/home/dfki/Desktop/Thesis/openml_test/pickel_files/3/trial_3.p", "rb"))
# # a = trial_builder_kmeans(trial_3,num_clusters=8000)
# print(len(a.trials))