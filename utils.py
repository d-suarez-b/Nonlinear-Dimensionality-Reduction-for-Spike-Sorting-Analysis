# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 01:34:36 2021

@author: Sebastian
"""
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory
from os import listdir
from numpy import genfromtxt


def _cluster_unique(cluster_labels):
    """This function is used when getting the number of spikes in each cluster
        We need to Use this instead of np.unique directly as np.bincount doesn't
        support negative integers and the noise cluster is always -1.

    Parameters
    ----------
    cluster_labels : Array shape (n_spikes,)
        clustering labels for the spikes.

    Returns
    -------
    cluster_hist : Array shape (2, n_clusters)
        First row corresponds to the cluster label.
        Second row is the number of spikes in the cluster.

    """
    labels = np.copy(cluster_labels)  # Necessary to remove reference glob var
    labels += 1

    cluster_hist = np.array(np.unique(labels, return_counts=True))
    cluster_hist[0] -= 1

    return cluster_hist


def isi_violations(spike_times, spike_trial, min_isi=2.5):
    """
    Calculates and returns a list of spikes that violate a given interspike
    interval (ISI) threshold.

    Parameters
    ----------
    spike_times : Array shape (n_spikes)
        Array containing the spike time for each spike.

    spike_trial : Array shape (n_spikes)
        Array containing the trial which every spike belongs to.

    min_isi : float, (optional, default 2.5)
        Threshold for the ISI. Value is in whatever untis spike_times is given.

    Returns
    -------
    isi_vector : Array shape (n_spikes,)
        False values indicate that spike has an ISI violation.

    """
    trials = np.unique(spike_trial)

    n_spikes = len(spike_times)
    isi_vector = np.zeros(n_spikes)

    for i in trials:
        spike_trial_index = np.argwhere(spike_trial == i)[:, 0]
        trial_diff = abs(np.diff(spike_times[spike_trial_index], axis=0)) >= min_isi

        # We note that if there are n spikes and m spike_diff violations, with
        #  2m<n in a trial --> there can be at least m+1 and at most 2m
        # violatory spikes. Essentially, the # of violatory spike diff is not
        # a perfect measure of the # of violatory spikes. Example:
        #
        #                       (s1s0s1s) != (s1s1s0s)
        #
        # With s representing spikes and 1,0 representing proper and violatory
        # Isi times. In both examples we have 2 spike_diff violations but in
        # the first one we have 4 violating spikes while in the second we only
        # have 3.
        #
        # To actually compute the number of spikes that have ISI violation we
        # have to find the spikes that correspond to the given violations.
        #
        # We default every spike to True (meaning no ISI violation) and then
        # we multiply each spike by the boolean trial_diff twice. The first
        # time with the left difference and the second the right difference.

        trial_spikes = np.full(shape=len(trial_diff) + 1, fill_value=True)
        trial_spikes[:-1] = trial_spikes[:-1] * trial_diff
        trial_spikes[1:] = trial_spikes[1:] * trial_diff
        isi_vector[spike_trial_index] = trial_spikes

    return isi_vector


def calculate_isi_percent(isi_vector):
    isi_percentage = 1 - np.count_nonzero(isi_vector) / len(isi_vector)
    isi_percentage *= 100  # convert to percentage

    return isi_percentage


def relative_validity(
    minimum_spanning_tree, labels,
):
    """Modified version of relative validity function from hdbscan to work as
     an independent function rather than a method.

     Parameters
     ----------
    minimum_spanning_tree_ : MinimumSpanningTree object
         The minimum spanning tree of the mutual reachability graph generated
         by HDBSCAN.

     labels : Array shape (n_spikes,)
         Cluster labels for each point in the dataset given to fit().
         Noisy samples are given the label -1.

     Returns
     -------
     score : float
         A fast approximation of the Density Based Cluster Validity (DBCV)
         score. The only differece, and the speed, comes from the fact
         that this relative_validity_ is computed using the mutual-
         reachability minimum spanning tree, i.e. minimum_spanning_tree_,
         instead of the all-points minimum spanning tree used in the
         article. This score might not be an objective measure of the
         goodness of clusterering. It may only be used to compare results
         across different choices of hyper-parameters, therefore is only a
         relative score.

    """

    mst_df = minimum_spanning_tree.to_pandas()
    sizes = np.bincount(labels + 1)
    noise_size = sizes[0]
    cluster_size = sizes[1:]
    total = noise_size + np.sum(cluster_size)
    num_clusters = len(cluster_size)
    DSC = np.zeros(num_clusters)
    min_outlier_sep = np.inf  # only required if num_clusters = 1
    correction_const = 2  # only required if num_clusters = 1

    # Unltimately, for each Ci, we only require the
    # minimum of DSPC(Ci, Cj) over all Cj != Ci.
    # So let's call this value DSPC_wrt(Ci), i.e.
    # density separation 'with respect to' Ci.
    DSPC_wrt = np.ones(num_clusters) * np.inf
    max_distance = 0

    for edge in mst_df.iterrows():
        label1 = labels[int(edge[1]["from"])]
        label2 = labels[int(edge[1]["to"])]
        length = edge[1]["distance"]

        max_distance = max(max_distance, length)

        if label1 == -1 and label2 == -1:
            continue
        elif label1 == -1 or label2 == -1:
            # If exactly one of the points is noise
            min_outlier_sep = min(min_outlier_sep, length)
            continue

        if label1 == label2:
            # Set the density sparseness of the cluster
            # to the sparsest value seen so far.
            DSC[label1] = max(length, DSC[label1])
        else:
            # Check whether density separations with
            # respect to each of these clusters can
            # be reduced.
            DSPC_wrt[label1] = min(length, DSPC_wrt[label1])
            DSPC_wrt[label2] = min(length, DSPC_wrt[label2])

    # In case min_outlier_sep is still np.inf, we assign a new value to it.
    # This only makes sense if num_clusters = 1 since it has turned out
    # that the MR-MST has no edges between a noise point and a core point.
    min_outlier_sep = max_distance if min_outlier_sep == np.inf else min_outlier_sep

    # DSPC_wrt[Ci] might be infinite if the connected component for Ci is
    # an "island" in the MR-MST. Whereas for other clusters Cj and Ck, the
    # MR-MST might contain an edge with one point in Cj and ther other one
    # in Ck. Here, we replace the infinite density separation of Ci by
    # another large enough value.
    #
    # TODO: Think of a better yet efficient way to handle this.
    correction = correction_const * (
        max_distance if num_clusters > 1 else min_outlier_sep
    )
    DSPC_wrt[np.where(DSPC_wrt == np.inf)] = correction

    V_index = [
        (DSPC_wrt[i] - DSC[i]) / max(DSPC_wrt[i], DSC[i]) for i in range(num_clusters)
    ]
    score = np.sum(
        [(cluster_size[i] * V_index[i]) / total for i in range(num_clusters)]
    )

    return score


def _ask_hbscan_params(
    default_mcs, default_min_samples, default_epsilon, defualt_sc,
):
    """Function that prompts user to input a set of HDBSCAN parameters.
    Must be given a set of defaults and returns user inputted parameters.
    """

    print("Previous value in parenthesis")
    min_cluster_size = int(input("(%u) Minimum Size of Clusters: " "" % default_mcs))

    epsilon = float(
        input(
            "(%0.2f) Cluster Selection Epsilon \nA distance"
            " threshold. Clusters below this value will be"
            " merged: " % default_epsilon
        )
    )

    min_samples = int(
        input(
            "(%u) min_samples \n The number of samples in a"
            " neighbourhood for a point to be considered a core "
            "point.\nLarger values will give you a more precise"
            "clustering but more noise points: "
            "" % default_min_samples
        )
    )

    sc_input = input(
        "(%s) Do you beleive there is a single cluster? \n \n"
        "The eom algorithm tends to favor the root (single)"
        " cluster so it is not an option by default. It is"
        " impossible to extract a single cluster unless you"
        " set this to True (True or False): " % defualt_sc
    )
    if sc_input.lower() in ["true", "1", "t", "y", "yes"]:
        single_cluster = True
    elif sc_input.lower() in ["false", "0", "f", "n", "no"]:
        single_cluster = False

    return (min_cluster_size, min_samples, epsilon, single_cluster)


def _alpha_values(cluster_sizes, alpha_modifier=100, lw_modifier=100, modify=True):
    """

    Parameters
    ----------
    cluster_sizes : TYPE
        DESCRIPTION.
    return_lw : TYPE, optional
        DESCRIPTION. The default is False.
    alpha_modifier : TYPE, optional
        DESCRIPTION. The default is 100.
    lw_modifier : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    """

    if modify is not True:
        alpha = np.zeros(len(cluster_sizes)) + 1  # defualt alpha
        linewidth = np.zeros(len(cluster_sizes)) + 1  # defualt linewidth
    else:
        alpha = (1 / cluster_sizes) * alpha_modifier
        linewidth = (1 / cluster_sizes) * lw_modifier
    x = (alpha, linewidth)

    return x


def ask_dir():
    """
    Call function to pull up a file search window and ask user to select a
    folder.

    Returns
    -------
    dir_path : str
        Return a string representing the selected directory.
    """
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    root.update()
    dir_path = askdirectory()
    root.destroy()

    return dir_path


def _get_data(folder_path, delimiter=","):
    """Function to extract and laod all csv files from a given directory
        into python as numpy arrays.


    Parameters
    ----------
    folder_path : str
        The path to the folder where you wish to load the csv files.


    delimiter: str, optional (defualt ',')
        The string used to separate values.

    Returns
    -------
    data_dict : dict
        Dictionary where the items are the data that has been loaded. The dict
        keys are the file names without the '.csv'.
    """

    filenames = listdir(folder_path)
    csv_files = [filename for filename in filenames if filename.endswith(".csv")]

    # Possible limit to the number of files you can load. But, as this function
    # is mainly to be used internally, we assume that it will be used responsibly
    # if len(csv_files)>15:
    #    raise ValueError("Too many files to load, this may be dangerous!")

    data_dict = {}

    for file in csv_files:
        var_name = file[:-4]  # remove .csv from string
        data = genfromtxt(folder_path + "/" + file, delimiter=delimiter)
        data_dict[var_name] = data

    return data_dict


def cluster_trial_frequency(spike_trial, labels, max_trial=None, trial_counts=None):
    if max_trial is None:
        max_trial = spike_trial.max()
    if trial_counts is None:
        trial_counts = np.bincount(spike_trial, minlength=max_trial + 1)[1:]

    trials = np.arange(max_trial) + 1
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    clus_frequency = np.zeros((n_clusters, max_trial))

    for j, clus in enumerate(clusters):
        clus_frequency[j] = np.bincount(
            spike_trial[labels == clus], minlength=max_trial + 1
        )[1:]

    return clus_frequency, clusters, trials, trial_counts


def selection_frequency(spike_trial, trial_counts, relative_freq, max_trial):
    if max_trial is None:
        max_trial = spike_trial.max()
    select_frequency = np.bincount(spike_trial, minlength=max_trial + 1)[1:]

    if relative_freq:
        ylabel = "Relative Frequency"
        y = np.divide(
            select_frequency,
            trial_counts,
            out=np.zeros(select_frequency.shape, dtype=float),
            where=trial_counts != 0,
        )
    else:
        ylabel = "Frequency"
        y = select_frequency

    return y, ylabel


def fast_isi(isi_spike_times, spike_trial, threshold, n_trials, sampling_rate=30000):
    """in order not to loop through trials we previously added a certain amount
        of time every trial so we can simply numpy diff the time vector to obtain
        isi violations


    Parameters
    ----------
    isi_spike_times : TYPE
        DESCRIPTION.
    threshold : TYPE
        in ms.
    sampling_rate : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # adjustment discouts ISI inbetween trials.
    adjustment = len(np.unique(spike_trial)) - 1
    violation = np.diff(isi_spike_times) > threshold
    if len(violation) - adjustment > 0:
        isi_rate = 1 - (np.count_nonzero(violation) - adjustment) / (
            len(violation) - adjustment
        )
    else:
        isi_rate = 0
    return isi_rate
