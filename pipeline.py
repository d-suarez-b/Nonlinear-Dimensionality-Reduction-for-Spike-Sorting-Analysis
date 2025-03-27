"""
Created by Sebastian Pujalte for the Rossi Pool Lab-UNAM
"""
import numpy as np
from re import search
import os
from os.path import join as pjoin
from os.path import basename
from shutil import copytree
from toposort.ml import spike_reducer
from toposort.preprocessing import spike_aligner, spike_denoiser
from datetime import datetime
from logging import info, error
from umap.umap_ import nearest_neighbors


def trailing_number(s):
    """Finds trailing number in a string. Used to find what trial a certain .nev
        file belongs to.
    https://stackoverflow.com/questions/7085512/check-what-number-a-string-ends-with-in-python/7085715
    """
    m = search(r"\d+$", s)
    return int(m.group()) if m else None


def get_lowest_dirs(root_dir):
    """Gets the lowest directories in a given file path. Removes root directory from
        beginning of string. Trick is that lowest directories have no directories
        inside of them.

    Parameters
    ----------
    root_dir : str
        filepath of directory you seek to find the root directories of.

    Returns
    -------
    lowest_dirs : list
        list of lowest directories in root_dir.

    """
    lowest_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if not dirs:
            lowest_dirs.append(basename(root))
    return lowest_dirs


def ig_f(dir, files):
    """Function to filter copytree; only copy directory tree and not files inside.
    https://stackoverflow.com/questions/15663695/shutil-copytree-without-files

    """
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


def spike_processer(
    spikes, savepath, n_neighs=[4, 25], random_state=96, disconnection_sd=None, sample_points=[None, None]
):
    """Denoises and aligns raw spikes that are already in array format and
         later obtains two UMAP embeddings for the processed spikes. Results are
         saved as csv's in given directory.

     Parameters
     ----------
     spikes : array, shape (n_spikes,n_sample_points)
         Array containing the raw spike waveforms.
    savepath : str
        Path to directory where you wish to save processed spikes and embeddings
        as csv's
     n_neighs : list (optional, default [13, 70])
         List containing two integers to be n_neighbors parameters for the two
         UMAP projections. Values must not be equal.
    sample_points: list (optional, default [None, None])
        List containing the indices of where you would like to cut off the aligned spikes.
        Depends on what each user wants. Example: [4, 45] -> aligned[:, 4: 45]
    """
    denoised = spike_denoiser(spikes)
    aligned = spike_aligner(denoised)[:, sample_points[0]: sample_points[1]]

    reduced1 = spike_reducer(
        aligned, n_neighbors=n_neighs[0], random_state=random_state
    )
    reduced2 = spike_reducer(
        aligned, n_neighbors=n_neighs[1], random_state=random_state
    )
    np.savetxt(
        pjoin(savepath, "Waveforms.csv"), aligned, delimiter=",", newline="\n",
    )
    np.savetxt(
        pjoin(savepath, f"embedding_neigh{n_neighs[0]}.csv"),
        reduced1.embedding_,
        delimiter=",",
        newline="\n",
    )
    np.savetxt(
        pjoin(savepath, f"embedding_neigh{n_neighs[1]}.csv"),
        reduced2.embedding_,
        delimiter=",",
        newline="\n",
    )

    return None


def spike_pipeline(filepath, savepath, monkey, mat_processer, nev_processer):
    """Complete spike pipeline that combines all previous functions to process
        entire directory of raw data. First the directory structure is copied
        (without files; only directories) and is slowly filled with processed
        data. This includes Spike Times, Spike Trials, Spike Waveforms and two
        UMAP projections to use in clustering. All resulting files are saved as
        .csv's.

    Parameters
    ----------
    filepath : str
        Path to the directory where the data directories are stored. Each directory
        should have a single .mat file and one .nev file for each trial.

    savepath : str
        Path to directory where you wish to save processed data.

    monkey : str
        Name of monkey. All processed data will be saved in a directory called
        f"{monkey}_processed"; where f indicates string formatting.

    mat_processor: callable
        Function that processes .mat information for given session into standard psicophysical
        format discussed in documentation. Every database will require a different function.
        Function must only include the inputs and outputs described here.

            inputs:
                filepath (str): filepath to where .mat files (or files) is stored.
                savepath (str): directory path where you wish to save processed data.
            output:
                psico (pd DF): Returns Dataframe where each row is a trial and the columns
                correspond to a type of psychophysical info. If savepath is not None then
                the dataframe will be saved to a .csv instead of being returned by the function.

    nev_processor: callable
        Function that processes all the nev files in a given directory. Data is seperated
        into the individual electrodes where they come from. Spike Trial and Spike Time
        information is saved into .csv's and the rest of the data is returned for further
        processing downstream. Function must only include the inputs and outputs described here.

            inputs:
                filepath (str): filepath to where .mat files (or files) is stored.
                savepath (str): directory path where you wish to save processed data.
            output:
                data (dict): Returns dictionary where the keys are the electrodes that were
                detected in the nev files. Inside those cells is the processed data for
                Spike Times, Spike Trials and Waveforms.
    """
    savepath = pjoin(savepath, f"{monkey}_processed")
    diagnostic_file = pjoin(savepath, "diagnostics.txt")
    dir_stack = get_lowest_dirs(filepath)

    if not os.path.exists(savepath):
        copytree(filepath, savepath, ignore=ig_f)
        with open(diagnostic_file, "w") as txt:
            txt.write("Files Successfully Processed:\n\n")
    else:
        # else to continue processing if somehow stopped in previous attempt
        # get empty files in savepath. Empty files have not been processed yet.
        dir_stack = [f for f in dir_stack if os.listdir(pjoin(savepath, f)) == []]

    for f in dir_stack:
        folder = pjoin(filepath, f)
        target_folder = pjoin(savepath, f)

        try:
            mat_processer(folder, target_folder)
        except Exception as e:
            info(
                f"\nError encountered when processing {target_folder}  "
                "Psychophysical information. Rest of pipeline will continue but "
                "you should process this part individually."
            )
            error(e, exc_info=True)

        try:
            data = nev_processer(folder, target_folder)
            for electrode in data.keys():
                savepath2 = pjoin(target_folder, electrode)
                spikes = data[electrode]["Waveforms"]
                spike_processer(spikes, savepath2)
                # Delete before starting each new loop to save memory
                del spikes
            del data
            with open(diagnostic_file, "a") as txt:
                txt.write(f"{f}        {str(datetime.now())}\n")

        except Exception as e:  # Catches any exception
            info(
                f"\nError encountered when processing {target_folder}. This "
                "folder will be skiped and you should process it individually."
            )
            error(e, exc_info=True)
