"""Window clustering and change point step of Version Clustering.

Copied unchanged from version-clustering-cdd (utilities/cluster_approach.py, MIT licence,
Abarca Zuniga, Yeshchenko and van der Aa, CAiSE 2026). Only the three functions that the detector
needs are kept, the plotting and file output code of the original is not used here.
"""
import hdbscan
import numpy as np
import ruptures as rpt


def find_transition_steps(probabilities,pen,n_windows):
    # Step 3: Change point detection (on embeddings)
    algo = rpt.KernelCPD(kernel="rbf").fit(probabilities)
    change_points = algo.predict(pen=pen)
    
    # Step 4: Interpret with cluster info
    change_points = [v for v in change_points if v < n_windows]

    return change_points

def order_clusters(clusters):
    # --- Relabel clusters by first appearance ---
    unique_in_order = []
    label_map = {-1:0}
    
    for c in clusters:
        if c not in label_map:
            label_map[c] = len(label_map)  # assign next available label
        unique_in_order.append(label_map[c])
    
    return np.array(unique_in_order)


def cluster_hdbscan(X,min_cluster_size=10,threshold=0.5):
    # X: your dataset as a 2D numpy array
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    clusterer.fit(X)
    
    # Soft cluster membership probabilities
    probabilities = hdbscan.all_points_membership_vectors(clusterer)
    
    # Assign cluster only if probability >= threshold
    labels = []
    for i, probs in enumerate(probabilities):
        max_prob = probs.max()
        if max_prob >= threshold:
            cluster = probs.argmax()
            labels.append(cluster)
        else:
            labels.append(-1)  # -1 indicates noise / unassigned
    
    return probabilities, order_clusters(labels)
        

