from toposort.utils import relative_validity  # , local_min
import numpy as np
from umap import UMAP
#from cuml.manifold import UMAP
import hdbscan
from hdbscan.hdbscan_ import _tree_to_labels
from umap.umap_ import nearest_neighbors
from sklearn.decomposition import PCA, FastICA
import pywt


def spike_reducer(
    spikes,
    n_neighbors=70,
    n_components=2,
    metric="manhattan",
    n_epochs=800,
    min_dist=0,
    random_state=None,
    low_memory=False,
    outlier_disconnection=0,
    set_op_mix_ratio=0.5,
    **kwargs
):
    """Projects collection of spikes in the timeseries space to a lower
    dimensional UMAP embedding.

    Parameters
    ----------
    spikes : array, shape (n_spikes,n_sample_points)
        Array of all spikes you to embedd in a reduced dimensional space.

     n_neighbors: float (optional, default 100)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.

    n_components: int (optional, default 2)
        The dimension of the space to embed into. By default this is 2, other
        embedding dimensions may not work well with the rest of TopoSort

    metric: string or function (optional, default 'manhattan')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function.

    n_epochs: int (optional, default 800)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings.

    min_dist: float (optional, default 0)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The default value is 0 as
        this results in better performance when clustering. It's best not to
        change this to anything above 0.1

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    low_memory: bool (optional, default False)
        For some datasets the nearest neighbor computation can consume a lot of
        memory. If you have issues with memory consumption, consider changing
        this to True

    outlier_disconnection : float, (optional, default 0)
        When 0 we do not disconnect any vertices in our knn graph. When >0 it
        is the number of styandard deviations that we use as a cutoff for
        disconnecting vertices. If not 0, we reccomend a value of at least 6,
        anything less will disconnect too many vertices and mess with the
        embedding.

    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    **kwargs :
        Additional UMAP parameters.

    Returns
    -------
    reducer: umap object that has already been fitted to the data. The spikes
             in the embedding space can be extracted with .embedding_
             attribute. Similarly, outliers can be extracted with the
             umap.utils.disconnected_vertices() function.
    """
    if outlier_disconnection > 0:
        knn = nearest_neighbors(
            spikes,
            n_neighbors,
            metric="manhattan",
            low_memory=False,
            metric_kwds=None,
            angular=False,
            random_state=42,
        )[1]

        cut_off = np.mean(knn) + 8 * np.std(knn)
    else:
        cut_off = None

    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        n_epochs=n_epochs,
        min_dist=min_dist,
        set_op_mix_ratio=set_op_mix_ratio,
        disconnection_distance=cut_off,
        random_state=random_state,
        low_memory=low_memory,
        **kwargs
    ).fit(X=spikes)

    # outliers = np.where(umap.utils.disconnected_vertices(reducer))[0]

    return reducer


def hdbscan_grid(
    spike_embedding,
    min_cluster_size=75,
    min_samples=[15, 25, 50, 75, 100, 150, 225, 300, 400, 600],
    epsilon=[0, 0.5, 0.75, 1, 2, 3, 4, 5, 7, 9],
):
    """Calculates HDBSCAN for 100 different parameter combinations of
    min_samples and cluster_selection_epsilon and returns the results.

    Parameters
    ----------
    spike_embedding : UMAP object,
        A trained UMAP object of the spike_waveforms that has a 2D embedding.

    min_cluster_size : int, optional (default = 75)
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.

     min_samples : list of int, optional
                   (default = [15, 25, 50, 75, 100, 150, 225, 300, 400, 600])
        List of the min_samples paramaeters to be tested in the HDBSCAN grid.
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    epsilon : list of float,
              optional (default = [0, .5, 0.75, 1, 2, 3, 4, 5, 7, 9])

        List of the cluster_selection_epsilon paramaeters to be tested in the
        HDBSCAN grid. cluster_selection_espilon is a distance threshold.
        Clusters below this value will be merged.

    Returns
    -------
    parameters : array, shape (2,10,10)
        First dimension is an array containing the min_samples
        parameter for all the HDBSCAN grid runs. Second dimension contains
        the cluster_selection_epsilons for al runs of the grid.

    grid_results : array, shape (2,10,10)
        First dimension is an array containing the number of clusters found
        in all the HDBSCAN grid runs. Second dimension contains
        the DBCV validation index for all runs of the grid..

    cluster_labels : array, shape (10,10,n_spikes)
        Array containing the labels found by all the iterations of the HDBSCAN
        grid.

    """
    if np.isnan(spike_embedding.sum()):  # Fast way to check for nan values
        raise ValueError(
            "Your spike_embedding array contains nan values! \n"
            + "This is usually because some vertices were "
            + "disconnected during the UMAP step.\nPlease remove "
            + "nan values before feeding the data to this function."
        )

    number_of_spikes = np.shape(spike_embedding)[0]

    min_samples = np.tile(min_samples, (10, 1))
    epsilon = np.tile(epsilon, (10, 1)).T
    n_clusters = np.zeros((10, 10))
    validation = np.zeros((10, 10))
    # cluster_persistance = np.zeros((10, 10))
    cluster_labels = np.zeros((10, 10, number_of_spikes), dtype=np.int64)

    for j in range(10):
        # Iterate over min_samples; invariant across rows

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size,
            cluster_selection_epsilon=float(epsilon[0, j]),
            min_samples=int(min_samples[0, j]),
            gen_min_span_tree=True,
        ).fit(spike_embedding)
        sl_tree = clusterer.single_linkage_tree_.to_numpy()
        ms_tree = clusterer.minimum_spanning_tree_
        cluster_labels[0, j] = clusterer.labels_
        n_clusters[0, j] = len(np.unique(clusterer.labels_))
        validation[0, j] = clusterer.relative_validity_

        print("Done {}/100 iterations of HDBSCAN Grid".format(j * 10))

        for i in range(10):
            # iterate over cluster_selection_epsilon; invariant across columns
            cluster_labels[i, j] = _tree_to_labels(
                X=spike_embedding,
                single_linkage_tree=sl_tree,
                min_cluster_size=min_cluster_size,
                cluster_selection_epsilon=epsilon[i, j],
            )[
                0
            ]  # We only want labels
            n_clusters[i, j] = len(np.unique(cluster_labels[i, j]))
            validation[i, j] = relative_validity(ms_tree, cluster_labels[i, j])
            # cluster_persistance[i, j] = clusterer.cluster_persistence_

    parameters = np.stack((min_samples, epsilon), axis=0)
    grid_results = np.stack((n_clusters, validation), axis=0)

    return parameters, grid_results, cluster_labels


def spike_hdbscan(
    spike_waveforms,
    spike_embedding,
    min_cluster_size=75,
    min_samples=50,
    epsilon=0,
    single_cluster=False,
    save_path=None,
    approx_mst=False,
    ylim=None,
    **kwargs
):
    """
    Parameters
    ----------
    spike_waveforms : array, shape (n_spikes,n_sample_points)
        Array containing the spike waveforms that you are clustering in a low
        dimensional space.

    spike_embedding : array, shape (n_spikes,2)
        The spikes that you wish to cluster in the embedded space. Note
        that the spike embedding must not contain NAN values.

    min_cluster_size : int, optional (default = 75)
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.

    min_samples : int, optional (default=50)
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    epsilon : float, optional (default=0.0)
        A distance threshold. Clusters below this value will be merged.

    single_cluster : bool, optional (default=False)
        By default HDBSCAN* will not produce a single cluster, setting this to
        True will override this and allow single cluster results in the case
        that you feel this is a valid result for your dataset.

    save_path : str, optional (default=None)
        Path where you wish to save all the images.

    approx_mst : bool, optional (default=False)
        Whether to accept an only approximate minimum spanning tree. For some
        algorithms this can provide a significant speedup, but the resulting
        clustering may be of marginally lower quality.

    ylim : List, optional (default=None)
        List containing the bottom and top (in that order) y limits that you
        wish to set in your plot. The limits must be floats in data coordinates.

    **kwargs : dict, optional
        Additional arguments to be passed on to the HDBSCAN clustering.

    Raises
    ------
    ValueError
        Error is raised when spike embedding contains Nan values. Hdbscan
        cannot deal with Nan values.

    Returns
    -------
    clusterer :trained HDBSCAN object
        Returns an HDBSCAN object that has been trained on the data. One can
        extract the labels by simply calling the 'labels_' attribute.

    """
    if np.isnan(spike_embedding.sum()):  # Fast way to check for nan values
        raise ValueError(
            "Your spike_embedding array contains nan values! \n"
            + "This is usually because some vertices were "
            + "disconnected during the UMAP step.\nPlease remove "
            + "nan values before feeding the data to this function."
        )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=epsilon,
        min_samples=min_samples,
        allow_single_cluster=single_cluster,
        gen_min_span_tree=True,
        approx_min_span_tree=approx_mst,
        prediction_data=True,
        **kwargs
    )

    clusterer.fit(spike_embedding)
    #    cluster_labels = clusterer.labels_
    #    dbcv = clusterer.relative_validity_

    if save_path is not None:
        save_path = save_path + "/"

    #    hdbscan_plots(
    #        clusterer,
    #        spike_waveforms,
    #        spike_embedding,
    #        cluster_labels,
    #        dbcv,
    #        alpha_modifier=None,
    #        lw_modifier=None,
    #        save_path=save_path,
    #        ylim=ylim,
    #    )

    return clusterer


def run_pca(spikes):
    reducer = PCA(n_components=2)
    embedding = reducer.fit_transform(spikes)
    return embedding


def run_ica(spikes):
    reducer = FastICA(n_components=2)
    embedding = reducer.fit_transform(spikes)
    return embedding


def add_wavelets(self):
    if self.wavelets is None:
        self.wavelets = pywt.dwt(self.spikes, "haar")
