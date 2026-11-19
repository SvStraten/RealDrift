"""Preprocessing functions that convert an event log into windowed directly-follows
graph (DFG) frequency matrices and compute a pairwise similarity matrix between
windows (used as input for clustering-based drift detection)."""

import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy import spatial

from typing import Tuple
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py import discover_dfg_typed
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from . import utilities as utils
from . import config as cfg


#############################################################################################
#############################      DFG METHOD      ##########################################
#############################################################################################

def log_to_windowed_dfg_count(event_log: EventLog, n_windows: int, n_features : int = 5) \
    -> Tuple[np.ndarray,list,dict,tuple]:
    """Convert event log to directly follows frequency counts for each activity.

    Args:
        event_log (EventLog): Event log
        n_windows (int): Number of windows

    Returns:
        Tuple[np.ndarray,list,dict,tuple]: Tuple, containing the dfg frequency counts, 
        border information of windows in form of trace indices, 
        window information with first trace and timestamp and
        date information with first and last timestamp of event log
    """

    # convert event log to pandas dataframe
    event_log_df = log_converter.apply(
        event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    event_log_df = dataframe_utils.convert_timestamp_columns_in_df(
        event_log_df, timest_format="ISO8601")
    
    min_date = np.min(event_log_df["time:timestamp"])
    max_date = np.max(event_log_df["time:timestamp"])
    date_info = (utils.datetime_2_str(min_date), 
                 utils.datetime_2_str(max_date))
    
    # get unique event names
    act_names = np.unique(event_log_df["concept:name"])

    # get unique trace names
    # hint: pandas unique does not sort the result, therefore it is faster and the
    # chronological order is maintained
    unique_traces = pd.unique(event_log_df["case:concept:name"])
    # unique_traces = utils.extract_integers_from_list(unique_traces)
    
    # get window size based on number of windows and event log size
    # event log size is equal to number of traces
    window_size = len(event_log) // n_windows

    # initialize helper variables
    freq_count = 0
    left_boundary = 0
    right_boundary = window_size
    borders = []
    dfg_graphs = []

    window_information = {}

    # iterate through windows
    for i in range(1, n_windows + 1):
        dfg_matrix_df = pd.DataFrame(0, columns=act_names, index=act_names)

        # get all trace names that are in selected window, traces are sorted by
        # timestamp
        if i < n_windows:

            w_unique_traces = unique_traces[left_boundary:right_boundary]

        # at last window fill until the end of the list
        else:
            w_unique_traces = unique_traces[left_boundary:]
            right_boundary = len(unique_traces) - 1 

        # search all events for given traces
        log_window = event_log_df[event_log_df["case:concept:name"]
                                  .isin(w_unique_traces)]

        # get dfg graph for window
        graph, _, _ = discover_dfg_typed(log_window)

        # transform dfg graph into dfg matrix
        for relation, freq in graph.items():
            rel_a, rel_b = relation
            dfg_matrix_df.at[rel_a, rel_b] = freq

            freq_count += freq

        dfg_graphs.append(dfg_matrix_df)

        borders.append((left_boundary, right_boundary))

        left_boundary = right_boundary
        right_boundary += window_size

        # get id of first trace in window - for evaluation
        first_trace = w_unique_traces[0]
        first_timestamp = min(
            event_log_df.loc[event_log_df["case:concept:name"] == first_trace,
                             "time:timestamp"])
        first_timestamp = first_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        window_information[i] = (first_trace, first_timestamp)

    # compare dfg frequencies of all windows with dfg frequencies of complete log
    # ensures that there are no missing relations
    total_freq = check_dfg_graph_freq(event_log_df)
    assert total_freq == freq_count, (
        "Missing directly follow relations.\n"
        f"Number of relations in whole event log: {total_freq}.\n"
        f"Number of relations in all windows: {freq_count}"
    )

    
    return dfg_graphs, borders, window_information, date_info


def similarity_calculation(windowed_dfg: np.ndarray) -> np.ndarray:
    """Compute similarity values for each element in directly follows relations.
    Creates a quadratic matrix that can be treated as an image by normalizing 
    the similarity values.

    Args:
        windowed_dfg (np.ndarray): Array containing directly follows relations

    Returns:
        np.ndarray: Similarity matrix
    """
    # create matrix
    n = len(windowed_dfg)
    sim_matrix = np.zeros((n, n))

    # calculate similarity measure between all elements of dfg matrix
    for i, matrix_i in enumerate(windowed_dfg):
        for j, matrix_j in enumerate(windowed_dfg):
            if (i == j) or (sim_matrix[i, j] != 0):
                continue
            else:
                sim_matrix[i, j] = calc_distance_norm(matrix_i, 
                                                      matrix_j, 
                                                      cfg.DISTANCE_MEASURE)
                sim_matrix[j, i] = sim_matrix[i, j]

    # check diagonal of similarity matrix
    assert np.sum(np.diagonal(sim_matrix)
                  ) == 0, "The diagonal of the similarity matrix must be zero."

    # normalize similarity matrix
    norm_sim_matrix = 1 - sim_matrix / np.amax(sim_matrix)

    return norm_sim_matrix

def calc_distance_norm(matrix_1: np.ndarray, matrix_2: np.ndarray, 
                       option: str) -> float:
    """Calculate distance/similarity value using the given directly follows matrices.

    Args:
        matrix_1 (np.ndarray): Directly follows matrix
        matrix_2 (np.ndarray): Directly follows matrix
        option (str): Indicates which measure to use for calculation

    Returns:
        float: Similarity value
    """
    diff = matrix_1 - matrix_2

    # Only "cos" (cosine distance) is currently used. The other measures are
    # kept here as comments for reference / in case they are needed again.
    # if option == "fro":
    #     # Frobenius norm
    #     dist_value = LA.norm(diff, "fro")
    # elif option == "nuc":
    #     # nuclear norm
    #     dist_value = LA.norm(diff, "nuc")
    # elif option == "inf":
    #     # max norm
    #     dist_value = LA.norm(diff, np.inf)
    # elif option == "l2":
    #     # L2 norm
    #     dist_value = LA.norm(diff, 2)
    # elif option == "cos":
    # cosine distance
    dist_value = spatial.distance.cosine(matrix_1.ravel(), matrix_2.ravel())
    # elif option == "earth":
    #     # earth mover distance == wasserstein distance
    #     dist_value = 0
    #
    #     # compute histograms and wasserstein distance for each activity
    #     for i in range(0, len(matrix_1)):
    #         matrix_1_hist, _ = np.histogram(
    #             matrix_1[i, :], bins=np.arange(-0.5, len(matrix_1)), density=True)
    #         matrix_2_hist, _ = np.histogram(
    #             matrix_2[i, :], bins=np.arange(-0.5, len(matrix_2)), density=True)
    #         dist_value += wasserstein_distance(matrix_1_hist, matrix_2_hist)

    if np.isnan(dist_value):
        dist_value = LA.norm(diff, "fro")

    return dist_value



def check_dfg_graph_freq(log:pd.DataFrame) -> float:
    graph, sa, ea = discover_dfg_typed(log)
    
    freq = sum(graph.values())
    
    return freq

    

