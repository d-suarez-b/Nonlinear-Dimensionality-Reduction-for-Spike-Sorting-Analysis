# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 01:34:36 2021

@author: Sebastian
"""
from toposort.utils import cluster_trial_frequency, selection_frequency, fast_isi

from os import mkdir
from os.path import join as pjoin
from os.path import dirname

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.cm import tab20

import holoviews as hv
from holoviews.selection import link_selections
from holoviews.operation.datashader import datashade
import datashader as ds
import panel as pn
import param


def add_extraticks(
    ax,
    fontsize,
    extratick_labs,
    extratick_pos,
    axvspan_pos,
    axspan_color="grey",
    xlim=[-2, 7],
):
    ax2 = ax.twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(list(extratick_pos))
    ax2.set_xticklabels(list(extratick_labs))
    ax2.tick_params(
        length=fontsize * 0.25, width=fontsize * 0.14, labelsize=fontsize * 0.625
    )

    if axvspan_pos is not None:
        for left, right in axvspan_pos:
            plt.axvspan(left, right, facecolor=axspan_color, alpha=0.5)

    return None


def raster_builder(
    title,
    raster_array,
    extratick_labs,
    extratick_pos,
    axvspan_pos,
    line_offsets,
    stim_intensity,
    clas_counts,
    CLASS_SEP,
    savepath=None,
    return_fig=False,
    xlim=(-2, 7),
):
    if savepath is not None:
        plot_kwargs = {"figsize": (60, raster_array.shape[0] / 7.5), "dpi": 150}
        fontsize = 80
    else:
        plot_kwargs = {"figsize": (20, raster_array.shape[0] / 22), "dpi": 100}
        fontsize = 15

    fig, ax = plt.subplots(**plot_kwargs)
    ax.set_title(title, size=fontsize)
    ax.eventplot(
        raster_array,
        lineoffsets=line_offsets,
        color="black",
        linelengths=1.2,
        alpha=0.5,
    )
    ax.set_ylim((-1, line_offsets[-1] + 1))
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (s)", size=fontsize * 0.88)
    yticklabs = [" / ".join(row.astype(str)) for row in stim_intensity]
    yticks = np.cumsum(clas_counts + CLASS_SEP) - (clas_counts + 2 * CLASS_SEP) / 2
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabs)
    ax.tick_params(
        length=fontsize * 0.25, width=fontsize * 0.14, labelsize=fontsize * 0.625
    )

    add_extraticks(ax, fontsize, extratick_labs, extratick_pos, axvspan_pos, xlim=xlim)

    if savepath is not None:
        fig.savefig(savepath)

    return fig


def raster_plotter(
    selection_index,
    spike_times,
    spike_trial,
    psico,
    sample_rate=30000,
    title="",
    path=None,
    clas_order=None,
    return_fig=False,
    a_dim="o1",
    extraticks=["o1", "f1", "o2", "f2", "o3", "f3"],
    stim_dims=["FreqStim1", "FreqStim2", "FreqStim3"],
    axvspan=None,
    xlim=(-2, 7),
    show_plot=False,
):
    """
    Parameters
    ----------
    selection_index : array
        Array of integer indices fo the spikes that you wish to plot.

    spike_times: array, shape (n_spikes, )
        Array indicating the time at which each spike was registered within its
        specific trial.

    spike_trial: array, shape (n_spikes, ), dtype int
        Array indicating the trial in which each spike was registered.

    psico : DataFrame
        Pandas Dataframe where each row is a trial and the columns
        correspond to a type of psychophysical info.

    title : str
        plot title

    path : str
        Direct path to folder where you wish to save raster.

    clas_order : list
        Permutation of trials specifying how spike trials should be ordered in the
        generated raster.

    return_fig: bool
        Whether to return figure object or simply show plot.

    Returns
    -------
    Figure of simply show plot.
    """
    if selection_index is not None:
        spike_times = spike_times[selection_index]
        spike_trial = spike_trial[selection_index]

    spike_times = spike_times / 30000  # 30khz sample rate
    CLASS_SEP = 3  # Num. of empty spaces between classes.
    clases, clas_indx, clas_counts = np.unique(
        psico.Class, return_counts=True, return_index=True
    )
    n_trials = len(pd.unique(psico.Trial))
    # We will create a nan array to efficiently plot rasters. First we have to calculate
    # the highest number of spikes per trial.
    max_trial_spikes = int(np.max(np.unique(spike_trial, return_counts=True)[1]))
    raster_array = np.full((n_trials, max_trial_spikes), np.nan)
    align_times = np.array(psico[a_dim])
    stim_intensity = np.array(psico[stim_dims].iloc[clas_indx]).astype(int)
    trials = np.array(psico.Trial, dtype=int)

    if clas_order is None:
        sort_order = [
            -stim_intensity[:, i] for i in reversed(range(stim_intensity.shape[1]))
        ]
        # minus sign changes to descending order sort
        stim_descending_orderer = np.lexsort(sort_order)
        clas_order = stim_descending_orderer
        trial_order = []
        for i in clas_order:
            trial_order.extend(trials[psico.Class == i + 1] - 1)

    clases = clases[clas_order]
    clas_counts = clas_counts[clas_order]
    stim_intensity = stim_intensity[clas_order]

    trials = np.array(psico.Trial)[trial_order]
    align_times_indexed = align_times[trial_order]
    for i in range(n_trials):
        x = spike_times[spike_trial == trials[i]]
        raster_array[i, : len(x)] = x - align_times_indexed[i]

    line_offsets = np.arange(n_trials) + np.repeat(
        (np.arange(len(clases))) * CLASS_SEP, repeats=clas_counts
    )

    # We only show ticks for stimulus time; probe and key times are not aligned between trials.
    extratick_pos = np.array(psico.iloc[0][extraticks] - psico.iloc[0][a_dim])
    if axvspan is not None and len(axvspan) % 2 == 0:
        axvspan_pos = (
            np.array(psico.iloc[0][axvspan]).reshape(int(len(axvspan) / 2), 2)
            - psico.iloc[0][a_dim]
        )
    else:
        axvspan_pos = None

    fig = raster_builder(
        title,
        raster_array,
        extraticks,
        extratick_pos,
        axvspan_pos,
        line_offsets,
        stim_intensity,
        clas_counts,
        CLASS_SEP,
        xlim=xlim,
        savepath=path,
    )

    if show_plot:
        plt.show()
    if return_fig:
        return fig


def psico_to_aligned_time(
    psico, spike_times, spike_trial, sample_rate=None, a_dim="o1"
):
    """
    Parameters
    ----------
    psico : DataFrame
        DESCRIPTION.

    spike_times: array, shape (n_spikes, )
        Array indicating the time at which each spike was registered within its
        specific trial.

    spike_trial: array, shape (n_spikes, ), dtype int
        Array indicating the trial in which each spike was registered.

    sample_rate : Int, (optional, default None)
        Sample Rate of spike times in order to convert to seconds.

    a_dim : str, optional (optional, default='o1')
        The dimension (column) in the psico df that is used as the alignment time

    Returns
    -------
    aligned_times: array, shape (n_spikes, )
        Spike times aligned to a certain event.

    """
    if sample_rate is not None:
        aligned_times = spike_times / sample_rate
    else:
        aligned_times = spike_times.copy()
    trials = np.unique(psico.Trial)
    for i in trials:
        aligned_times[spike_trial == i] = (
            aligned_times[spike_trial == i] - float(psico[psico.Trial == i][a_dim])
        )

    return aligned_times


def save_neurons(
    labels,
    spike_times,
    spike_class,
    spike_trial,
    savepath,
    psico,
    sample_rate=30000,
    raster_func=None,
    raster_kwargs=None,
    return_neurons=False,
    a_dim="o1",
):
    """Function that takes information about the spikes and the generated cluster
    labels and saves all the neurons in a standard format in a given savepath.

    Parameters
    ----------
    labels : array, shape (n_spikes, )
        Array indicating the cluster label for each spike. These labels must follow
        the sklearn clustering convention where -1 represents noise points and
        positive integers are clusters.

    spike_times: array, shape (n_spikes, )
        Array indicating the time at which each spike was registered within its
        specific trial.

    spike_class : array, shape (n_spikes, )
        Array indicating the class which the spike was registered in. This is
        possibly redundant as the trial uniquely identifies the class.

    spike_trial: array, shape (n_spikes, ), dtype int
        Array indicating the trial in which each spike was registered.

    savepath : str
        The path to the directory where you wish to save the neurons.

    Returns
    -------
    neurons : dict
        Dictionary containing the all the neurons. The neurons are indexed by
        their cluster labels.
    """
    cols = ["Trial", "Spike Time", "Class"]
    neurons = {}

    if raster_func is not None and raster_kwargs is not None:
        raster_func(
            selection_index=None,
            spike_times=spike_times,
            spike_trial=spike_trial,
            title="All spikes",
            sample_rate=sample_rate,
            path=pjoin(savepath, "all_spikes_raster.png"),
            **raster_kwargs,
        )
    savepath2 = pjoin(savepath, "sorted_neurons")
    mkdir(savepath2)
    np.savetxt(pjoin(savepath2, "sorted_labels.csv"), labels, fmt="%i", delimiter=",")
    aligned_times = psico_to_aligned_time(
        psico, spike_times, spike_trial, sample_rate, a_dim
    )

    for j in np.unique(labels):
        labels_mask = labels == j
        trials = spike_trial[labels_mask, np.newaxis]
        times = aligned_times[labels_mask, np.newaxis]
        clase = spike_class[labels_mask, np.newaxis]
        neurons[j] = pd.DataFrame(np.hstack((trials, times, clase)), columns=cols)

        if j == -1:
            name = "noise"
        else:
            name = f"neuron_{int(j + 1)}"
        neurons[j].to_csv(pjoin(savepath2, f"{name}.csv"), index=False)

        if raster_func is not None and raster_kwargs is not None:
            raster_times = spike_times[labels_mask, np.newaxis]
            raster_func(
                selection_index=None,
                spike_times=raster_times,
                spike_trial=trials,
                title=str(name),
                sample_rate=sample_rate,
                path=pjoin(savepath2, f"{name}.png"),
                **raster_kwargs,
            )
    if return_neurons == True:
        return neurons


def plot_cluster_frequency(
    clus_frequency, clusters, trials, trial_counts, color_vector, percent=True
):
    if percent:
        ylabel = "Relative Frequency"
        y = np.divide(
            clus_frequency,
            trial_counts,
            out=np.zeros(clus_frequency.shape, dtype=float),
            where=trial_counts != 0,
        )
    else:
        ylabel = "Frequency"
        y = clus_frequency

    curve_list = []
    for j, i in enumerate(clusters):
        curve = hv.Curve(np.stack((trials, y[j]), axis=1), label=f"Clus {i}").opts(
            color=color_vector[j]
        )
        curve_list.append(curve)

    kw_opts = {
        "legend_position": "right",
        "width": 1350,
        "height": 500,
        "legend_opts": {"click_policy": "hide"},
        "xlabel": "Trial",
        "ylabel": ylabel,
        "title": "Cluster Frequency",
        "fontsize": {"title": 15, "labels": 14, "xticks": 10, "yticks": 10},
    }
    return hv.Overlay(curve_list).opts(**kw_opts)


def plot_selection_frequency(frequency, trials, title="", ylabel=""):

    kw_opts = {
        "width": 1000,
        "height": 400,
        "xlabel": "Trial",
        "ylabel": ylabel,
        "title": title,
        "fontsize": {"title": 13, "labels": 12, "xticks": 9, "yticks": 9,},
    }
    curve = hv.Curve(np.stack((trials, frequency), axis=1))

    return curve.opts(**kw_opts)


def highlight(val, color_vector=None):
    """
    Function for pandas styling of table.
    """
    color = color_vector[val + 1]
    return "background-color: %s" % color


def table_formatter(table, color_vector, subset):
    """
    receives unformatted pandas df table to print in panel and returns

    Parameters
    ----------
    table : TYPE
        DESCRIPTION.

    Returns
    -------
    table: pandas df

    """
    # https://stackoverflow.com/questions/66172553/styling-a-pandas-dataframe-in-python-makes-the-float-values-have-more-zeros-afte
    d = dict.fromkeys(table.select_dtypes("float").columns, "{:.2%}")
    styled_table = (
        table.style.map(
            highlight, color_vector=color_vector, subset=subset
        ).format(d)
        # .hide(axis="index"); Will be available in pandas 1.4
    )

    styled_table = styled_table.set_properties(
        **{
            "text-align": "center",
            "border-color": "Black",
            "border-width": "thin",
            "border-style": "solid",
        }
    )
    styled_table.set_table_styles(
        [
            dict(
                selector="th",
                props=[("text-align", "center"), ("border-collapse", "collapse")],
            ),
            dict(selector="td", props=[("border-collapse", "collapse")]),
        ]
    )

    return styled_table


class interactive_clustering(param.Parameterized):
    """Panel dashboard for hand clustering of neural action potentials (spikes).
    This is part of the toposort library and is intended for hand clustering of
    spikes in two UMAP embeddings.

    Parameters
    ----------
    embedding1 : Array, shape (n_spikes,2)
        2D UMAP embedding of the spikes.

    embedding2 : Array, shape (n_spikes,2)
        Second 2D UMAP embedding of the spikes.

    spikes : Array, shape (n_spikes, n_samples)
        Timeseries of the spikes; Each row corresponds to one spike.

    labels : Array, shape (n_spikes, ), (optional, default None)
        Optionally pass predefined labels of the spikes before clustering.

    raster : function, (optional, default None)
        Optionally pass a raster plotting function to plot currently selected points.
        TODO: description of function arguments

    raster_kwargs : dict, (optional, default None)
        Keyword arguments necessary for the raster plotting function.

    **params : dict
        Additional parameters for the parametrized object.
        TODO: additional documentation

    """

    INDEX_HTML = pjoin(dirname(__file__), "..", "assets", "index.html")

    table_watcher = param.Boolean(True)
    link = link_selections.instance()

    reset_checkbox = pn.widgets.Checkbox(name="Reset Only Selection")
    cluster_checkbox = pn.widgets.Checkbox(name="Only Cluster Noise Points")
    window_checkbox = pn.widgets.Checkbox(name="Fix Plot Sizes")

    reset_button = param.Action(
        lambda self: self.reset_callback(), label="Reset Clustering"
    )
    cluster_button = param.Action(
        lambda self: self.cluster_callback(), label="Cluster Points"
    )
    quit_button = param.Action(lambda self: self.quit_callback(), label="Save and Quit")
    raster_button = param.Action(
        lambda self: self.raster_wrapper(), label="Generate Raster"
    )
    align_button = pn.widgets.Button(name="Change Alignment", button_type="primary")

    cluster_dropdown = param.ObjectSelector(
        default="No Selection",
        objects=["No Selection"] + ["Cluster {}".format(i - 1) for i in range(20)],
        label="Select Cluster To View",
    )
    merge_button = param.Action(
        lambda self: self.merge_callback(), label="Merge Clusters"
    )
    merge_dropdown1 = pn.widgets.Select(
        name="Cluster to Merge",
        options=["Cluster {}".format(i - 1) for i in range(20)],
        width=150,
    )
    merge_dropdown2 = pn.widgets.Select(
        name="Cluster to Merge",
        options=["Cluster {}".format(i - 1) for i in range(20)],
        width=150,
    )
    agg_dropdown = pn.widgets.Select(
        name="Timeseries Aggregator",
        options=["log", "eq_hist", "cbrt", "linear"],
        value="eq_hist",
    )
    cluster_stability_button = pn.widgets.Toggle(
        name="Frequency Type", value=True, button_type="success"
    )
    isi_slider = pn.widgets.FloatSlider(
        name="ISI Threshold (ms)",
        start=1.5,
        end=5,
        step=0.05,
        value_throttled=2,
        value=2,
    )

    def __init__(
        self,
        embedding1,
        embedding2,
        spikes,
        spike_times,
        spike_trial,
        spike_class,
        psico,
        align_save_dim="o1",
        sample_rate=30000,
        savepath=None,
        labels=None,
        raster=None,
        raster_kwargs=None,
        **params,
    ):
        super(interactive_clustering, self).__init__(**params)
        self.embedding1 = embedding1
        self.embedding2 = embedding2
        self.spikes = spikes
        self.spike_times = spike_times
        self.spike_trial = spike_trial
        self.spike_class = spike_class
        self.psico = psico
        self.align_save_dim = align_save_dim
        self.sample_rate = sample_rate
        self.savepath = savepath
        self.labels = labels
        self.selection = None
        self.bokeh = None
        self.next_cluster = 0
        self.raster = raster
        self.raster_kwargs = raster_kwargs

    def validate_params(self):
        self.n_spikes = self.embedding1.shape[0]
        self.n_samples = self.spikes.shape[1]
        if self.labels is None:
            self.labels = np.zeros(self.n_spikes) - 1

        df = pd.DataFrame(
            np.concatenate((self.embedding1, self.embedding2), axis=1),
            columns=["a", "b", "c", "d"],
        )
        # 20 is the max number of colors/clusters for our plot
        df["labs"] = pd.Categorical(self.labels, categories=np.arange(20) - 1)
        self.df = hv.Dataset(df)
        # We give 20s inbetween each spike trial's times.
        self.isi_times = self.spike_times + 20 * self.spike_trial * self.sample_rate
        self.n_trials = len(np.unique(self.spike_trial))
        self.color_vector = [
            rgb2hex(tab20(i))
            for i in np.concatenate((np.arange(10) * 2, np.arange(10) * 2 + 1))
        ]
        self.max_trial = self.spike_trial.max()
        self.trial_counts = np.bincount(self.spike_trial, minlength=self.max_trial + 1)[
            1:
        ]
        if self.savepath is None:
            print("\n savepath was not given. You will not be able to save results!")
        if self.raster_kwargs is None:
            self.raster_kwargs = {}
        if "a_dim" not in self.raster_kwargs.keys():
            self.raster_kwargs["a_dim"] = "o1"
        if "xlim" not in self.raster_kwargs.keys():
            self.raster_kwargs["xlim"] = (-2, 7)
        self.align_dropdown = pn.widgets.Select(
            name="Select Event to Align",
            options=list(self.psico.keys()),
            value=self.raster_kwargs["a_dim"],
            width=300,
        )
        self.xlim_slider = pn.widgets.RangeSlider(
            name="X-axis limits",
            start=-10,
            end=10,
            value=self.raster_kwargs["xlim"],
            step=0.1,
        )

    @param.depends("cluster_dropdown", watch=True)
    def cluster_selector(self):
        if self.cluster_dropdown != "No Selection":
            try:
                # :-2 ensures negative values are captured
                selected_cluster = int(self.cluster_dropdown[-2:])
                self.link.selection_expr = hv.dim("labs") == selected_cluster
            except AttributeError:
                print(
                    "You have selected a cluster with no points!\n"
                    "Dropdown will return to 'No Selection'."
                )
        else:
            self.link.selection_expr = None

    def raster_wrapper(self):
        """
        Wrapper for raster plotting function and plots selected points.
        """
        plt.close("All")
        self.selection = self.df.select(self.link.selection_expr).data.index
        self.raster_kwargs["a_dim"] = self.align_dropdown.value
        self.raster_kwargs["xlim"] = self.align_dropdown.value
        if self.raster is not None and self.raster_kwargs is not None:
            self.raster(
                selection_index=self.selection,
                spike_trial=self.spike_trial,
                spike_times=self.spike_times,
                **self.raster_kwargs,
                show_plot=True,
            )

    @param.depends(link.param.selection_expr, align_button.param.value)
    def raster_tab(self, _, __):
        self.selection = self.df.select(self.link.selection_expr).data.index
        self.raster_kwargs["a_dim"] = self.align_dropdown.value
        self.raster_kwargs["xlim"] = self.xlim_slider.value

        if self.raster is not None and self.raster_kwargs is not None:
            fig = self.raster(
                selection_index=self.selection,
                spike_trial=self.spike_trial,
                spike_times=self.spike_times,
                **self.raster_kwargs,
                return_fig=True,
            )
            plt.close()

        raster = pn.pane.Matplotlib(fig, dpi=100, tight=True)
        # pn.Column(raster, pn.Row(align_dropdown, self.align_button))
        return raster

    def quit_callback(self):
        """
        Callback to stop the bokeh server and save clustering.
        """
        save_neurons(
            labels=self.df["labs"].__array__(dtype=int),
            spike_times=self.spike_times,
            spike_class=self.spike_class,
            spike_trial=self.spike_trial,
            savepath=self.savepath,
            psico=self.psico,
            raster_func=self.raster,
            raster_kwargs=self.raster_kwargs,
            a_dim=self.align_save_dim,
        )
        print("\nNeurons have been saved successfully!")

        try:
            self.bokeh.stop()
            print("\nYou have successfully stopped the server")
        except Exception:
            pn.state.kill_all_servers()
            print(
                "\nCouldn't stop server directly so instead we killed all bokeh servers"
            )

    def reset_callback(self):
        """
        Callback to reset clustering so that all points are registered as noise.
        """
        if not self.reset_checkbox.value:
            self.df["labs"][np.arange(self.n_spikes, dtype=int)] = -1
            self.next_cluster = 0
        else:
            index = self.df.select(self.link.selection_expr).data.index
            self.df["labs"][index] = -1

        self.link.selection_expr = None  # This refreshes the plot after clustering
        self.table_watcher = not self.table_watcher

    def cluster_callback(self):
        """
        Callback to cluster points that are currently selected.
        """
        self.selection = self.df.select(self.link.selection_expr).data.index
        if not self.cluster_checkbox:
            self.df["labs"][
                self.selection
            ] = self.next_cluster  #  TODO speedup with cudf
        else:
            selection_labs = self.df["labs"][self.selection]
            mask = selection_labs == -1
            self.df["labs"][self.selection[mask]] = self.next_cluster

        self.link.selection_expr = None  # This refreshes the plot after clustering
        self.table_watcher = not self.table_watcher

    @param.depends("merge_button")
    def merge_callback(self):
        clus1 = int(self.merge_dropdown1.value[-2:])
        clus2 = int(self.merge_dropdown2.value[-2:])

        if clus1 == min(clus1, clus2):
            self.df["labs"][self.df.select(hv.dim("labs") == clus2).data.index] = clus1
        else:
            self.df["labs"][self.df.select(hv.dim("labs") == clus1).data.index] = clus2
        self.table_watcher = not self.table_watcher

    @param.depends("table_watcher")
    def cluster_table(self):
        # --- Debug Info (plotting.py) ---
        print("--- Debug Info inside cluster_table ---")
        value_counts_result = None # Initialize to None
        df_data = None             # Initialize to None
        try:
            # Use pandas directly if self.df is already a DataFrame
            # Otherwise, access .data if it's a HoloViews object
            if isinstance(self.df, pd.DataFrame):
                 df_data = self.df
            elif hasattr(self.df, 'data') and isinstance(self.df.data, pd.DataFrame):
                 df_data = self.df.data
            else:
                 print("Cannot determine DataFrame structure for self.df")
                 # df_data remains None

            if df_data is not None:
                print("self.df type (underlying):", type(df_data))
                print("self.df shape:", df_data.shape if hasattr(df_data, 'shape') else 'N/A')

                if 'labs' in df_data.columns:
                    print("self.df['labs'] head:\n", df_data['labs'].head())
                    value_counts_result = df_data['labs'].value_counts()
                    print("Value Counts:\n", value_counts_result)
                    print("Is value_counts empty?", value_counts_result.empty)

                    # Debug the temporary DataFrame creation method that failed
                    temp_table_df_old_method = pd.DataFrame(
                         value_counts_result, columns=["Number of Spikes"]
                    )
                    print("temp_table_df (OLD METHOD result):\n", temp_table_df_old_method)
                    print("Is temp_table_df (OLD METHOD) empty?", temp_table_df_old_method.empty)

                else:
                    print("self.df['labs'] column not found in df_data")
            else:
                print("Could not extract DataFrame data from self.df")

        except Exception as e:
            print(f"DEBUG: Error during plotting.py debug prints: {e}")
        print("--- End Debug Info ---")
        # --- End Debug Info ---

        # --- Start of Corrected Logic ---
        # Ensure df_data and value_counts_result are available from the try block
        if df_data is None or 'labs' not in df_data.columns or value_counts_result is None:
            print("ERROR: Missing data needed to create cluster table.")
            # Handle error appropriately - maybe create an empty df or return None
            self.table_df = pd.DataFrame(columns=["Number of Spikes"]) # Create empty df
            # return None # Alternative: exit if data is bad
        else:
            # --- Create DataFrame explicitly (Corrected Method) ---
            self.table_df = pd.DataFrame({
                 'Number of Spikes': value_counts_result.values
            }, index=value_counts_result.index)
            # self.table_df.index.name = 'Index' # Optional: name index if needed
            # --- End explicit creation ---

        # --- End of Corrected Logic ---

        # Check if table_df is empty *before* trying idxmin()
        if self.table_df.empty or self.table_df["Number of Spikes"].empty:
             print("WARNING: self.table_df or 'Number of Spikes' column is empty before idxmin!")
             # Handle empty case: maybe assign a default, skip, or raise an error
             self.next_cluster = -1 # Example: Assign default if empty
             # Set an empty DataFrame for formatting if it failed
             if self.table_df.empty:
                 # Ensure 'Cluster' column exists for formatter even if empty
                 self.table_df['Cluster'] = []
                 self.table_df = self.table_df.astype({'Cluster': int}) # Match expected type
        else:
             # Add 'Cluster' column only if DataFrame is not empty
             self.table_df["Cluster"] = self.table_df.index.astype(int)
             # Ensure there are non-zero counts before finding min if necessary
             # (idxmin raises error on empty or all-NaN series, but value_counts shouldn't produce NaNs here)
             if not self.table_df["Number of Spikes"].dropna().empty:
                 self.next_cluster = self.table_df["Number of Spikes"].idxmin()
             else:
                 print("WARNING: 'Number of Spikes' column is empty or all NaN after potential dropna().")
                 self.next_cluster = -1 # Fallback if no valid min

        # Ensure 'Cluster' column exists before formatting, even if table was initially empty
        if 'Cluster' not in self.table_df.columns:
             self.table_df['Cluster'] = self.table_df.index.astype(int) if not self.table_df.empty else []

        styled_table = table_formatter(self.table_df, self.color_vector, ["Cluster"])
        return styled_table
    def embedding_plots(self):
        kw_opts = {
            "shared_axes": False,
            "width": 600,
            "height": 475,
            "xlabel": "",
            "ylabel": "",
            "active_tools": ["box_select"],
            # Figure out how to erase the other tools
            # "tools":['wheel_zoom', 'reset', 'box_zoom', 'box_select', 'pan'],
            # "default_tools":[],
        }
        s1_opts = {}
        s2_opts = {}
        if self.window_checkbox.value:
            s1_opts["ylim"] = self.scatter1.range("y")
            s1_opts["xlim"] = self.scatter1.range("x")
            s2_opts["ylim"] = self.scatter2.range("y")
            s2_opts["xlim"] = self.scatter2.range("x")

        self.scatter1 = hv.Points(self.df, ["a", "b"], ["labs"])
        self.scatter2 = hv.Points(self.df, ["c", "d"], ["labs"])

        raster1 = datashade(
            self.scatter1,
            aggregator=ds.count_cat("labs"),
            precompute=True,
            color_key=self.color_vector,
        ).opts(title="UMAP Embedding 1", **kw_opts, **s1_opts)

        raster2 = datashade(
            self.scatter2,
            aggregator=ds.count_cat("labs"),
            precompute=True,
            color_key=self.color_vector,
        ).opts(title="UMAP Embedding 2", **kw_opts, **s2_opts)

        return self.link(raster1 + raster2).cols(2).opts(merge_tools=True)

    # The definition of the following fucntion requires having two arguments
    # It will not run without the '_'.
    # TODO: investigate why
    # @param.depends(link.param.selection_expr)
    @param.depends(agg_dropdown.param.value, link.param.selection_expr)
    def plot_curve(self, _, __):
        kw_opts = {"width": 1200, "height": 425}
        if self.window_checkbox.value:
            kw_opts["ylim"] = self.curve.range("y")
            kw_opts["xlim"] = self.curve.range("x")

        index = self.df.select(self.link.selection_expr).data.index
        flat_spikes = ds.utils.dataframe_from_multiple_sequences(  #  TODO make compatible with cupy
            np.arange(self.n_samples), self.spikes[index]
        )
        self.curve = datashade(
            hv.Curve(flat_spikes),
            precompute=True,
            cnorm=self.agg_dropdown.value,
            cmap=["lightblue", "red"],
        ).opts(**kw_opts)

        return pn.pane.HoloViews(self.curve, linked_axes=False)

    @param.depends("agg_dropdown.value", "table_watcher")
    def plot_all_clusters(self):
        non_zero = self.table_df.index[self.table_df["Number of Spikes"] != 0].tolist()
        # gspec = pn.GridSpec(width=1900, height=950)
        cluster_plots = []
        for j, i in enumerate(non_zero):
            index = self.df.select(hv.dim("labs") == i).data.index
            spikes = ds.utils.dataframe_from_multiple_sequences(  #  TODO make compatible with cupy
                np.arange(self.n_samples), self.spikes[index]
            )
            subplot = datashade(
                hv.Curve(spikes),
                precompute=True,
                cnorm=self.agg_dropdown.value,
                cmap=["white", self.color_vector[i + 1]],
            ).options(
                width=430,
                height=240,
                title="Cluster {}".format(i),
                toolbar="above",
                xlabel="",
                ylabel="",
            )
            cluster_plots.append(subplot)

        return hv.Layout(cluster_plots).cols(5)

    @param.depends("table_watcher", "cluster_stability_button.value")
    def plot_trial_freq(self):
        relative_freq = self.cluster_stability_button.value
        labs = self.df["labs"].__array__(dtype=int)
        freq, clus, trials, trial_counts = cluster_trial_frequency(
            self.spike_trial, labs, self.max_trial, self.trial_counts
        )
        plot = plot_cluster_frequency(
            freq, clus, trials, trial_counts, self.color_vector, relative_freq
        )

        return pn.pane.HoloViews(plot, linked_axes=False)

    @param.depends("table_watcher", "isi_slider.value_throttled")
    def isi_table(self):
        threshold = self.isi_slider.value_throttled
        adjusted_threshold = (threshold * self.sample_rate) / 1000
        isi_table_df = pd.DataFrame(index=self.table_df.index)
        isi_table_df["Cluster"] = isi_table_df.index.astype(int)
        isi_table_df["ISI Violations"] = float(0)

        non_zero = isi_table_df.index[self.table_df["Number of Spikes"] != 0].tolist()
        for j, i in enumerate(non_zero):
            index = self.df.select(hv.dim("labs") == i).data.index
            isi_table_df.at[i, "ISI Violations"] = fast_isi(
                self.isi_times[index],
                self.spike_trial[index],
                adjusted_threshold,
                self.n_trials,
            )

        styled_table = table_formatter(isi_table_df, self.color_vector, ["Cluster"])
        return styled_table

    @param.depends(
        link.param.selection_expr,
        cluster_stability_button.param.value,
        isi_slider.param.value_throttled,
    )
    def selection_validity(self, _, __, ___):
        index = self.df.select(self.link.selection_expr).data.index
        relative_freq = self.cluster_stability_button.value

        threshold = self.isi_slider.value_throttled
        adjusted_threshold = (threshold * self.sample_rate) / 1000
        selection_isi = fast_isi(
            self.isi_times[index],
            self.spike_trial[index],
            adjusted_threshold,
            self.n_trials,
        )
        y, ylabel = selection_frequency(
            self.spike_trial, self.trial_counts, relative_freq, self.max_trial
        )
        trials = np.arange(self.max_trial) + 1
        title = f"Selection  |  ISI = {selection_isi: .2%} "
        curve = plot_selection_frequency(y, trials, title, ylabel)

        return pn.pane.HoloViews(curve, linked_axes=False)

    def set_layout(self):
        pn.extension(loading_spinner="arc", loading_color="#00aa41", sizing_mode=None)
        pn.param.ParamMethod.loading_indicator = True
        hv.extension("bokeh")
        logo = pn.panel(
            "https://i0.wp.com/www.atmosfera.unam.mx/wp-content/uploads/2019/06/unam-escudo.png",
            height=150,
        )
        title = pn.panel("https://i.imgur.com/C2qQa3Z.png")
        checkboxes = pn.Column(
            self.reset_checkbox, self.cluster_checkbox, self.window_checkbox
        )
        buttons = pn.Row(
            self.param.quit_button,
            self.param.reset_button,
            self.param.cluster_button,
            self.param.raster_button,
            checkboxes,
            width=1300,
        )
        top = pn.Row(pn.Spacer(styles={'background': 'white'}, width=200), title)
        bottom = pn.Row(pn.Spacer(styles={'background': 'white'}, width=450), logo)
        merge = pn.Column(
     	self.param.merge_button,
     	pn.Row(self.merge_dropdown1, self.merge_dropdown2),
     	styles={'background': 'WhiteSmoke'},
    	 width=350,
	)
        left = pn.Column(self.plot_curve, self.embedding_plots, buttons)
        right = pn.Column(
            top,
            self.param.cluster_dropdown,
            pn.Row(self.cluster_table, pn.Column(merge, self.agg_dropdown),),
            bottom,
            width=600,
        )
        return pn.Row(left, right)

    def set_validity(
        self, show_clusters=False, show_raster=False, show_stability=False
    ):
        active_tabs = []
        if show_clusters:
            active_tabs.append(("Cluster Spikes", self.plot_all_clusters))
        if show_raster:
            self.raster_tab(1, 2)
            rasters = pn.Column(
                self.raster_tab,
                pn.Row(self.align_button, self.align_dropdown, self.xlim_slider),
            )
            active_tabs.append(("Rasters", rasters))
        if show_stability:
            validity_widgets = pn.Column(self.cluster_stability_button, self.isi_slider)
            validity_top = pn.Row(self.plot_trial_freq, self.isi_table)
            validity_bottom = pn.Row(self.selection_validity, validity_widgets)
            validity = pn.Column(validity_top, validity_bottom)
            active_tabs.append(("ISI/Stability", validity))
        return pn.Tabs(*active_tabs, width=1930, height=1000, dynamic=True)

    def show(self, show_clusters=False, show_raster=False, show_stability=False):
        self.validate_params()
        if self.raster is None:
            show_raster = False
            print("No raster function provided so own't be able to plot.")

        pages = {"main_sorter": self.set_layout()}
        titles = {"main_sorter": "TopoSort Main"}

        if show_clusters or show_raster or show_stability:
            pages["diagnostics"] = self.set_validity(
                show_clusters, show_raster, show_stability
            )
            titles["diagnostics"] = "Toposort Diagnostics"

        self.bokeh = pn.serve(
            pages, start=False, show=True, index=self.INDEX_HTML, title=titles,
        )
        self.bokeh.start()
        try:
            self.bokeh.io_loop.start()
        except:
            pass
