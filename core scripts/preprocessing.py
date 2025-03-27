# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:47:41 2021

@author: Sebastian
"""
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
from scipy.signal.windows import tukey
import numpy as np


def spike_denoiser(spikes, window_length=5, polyorder=3):
    """
    savgol_filter: Filter with a window length of 5 and a degree 3 polynomial.
        Use the defaults for all other parameters.


        Parameters
        ----------
        spikes : array, shape (n_spikes,n_sample_points)
            Array of all spikes you wish to denoise. Contains one spike per row.

        window_length : int
            The length of the filter window (i.e., the number of coefficients).

        polyorder : int
            The order of the polynomial used to fit the samples.
            `polyorder` must be less than `window_length`.

        Returns
        -------
        spikes_denoised : array, shape (n_spikes,n_sample_points)
            Array denoised all spikes after denoising. Contains one spike per row.


        TO DO: Implement more options for denoising filters.
    """
    spikes_denoised = savgol_filter(spikes, 5, 3)
    return spikes_denoised


def spike_aligner(
    spikes,
    upsample_rate=8,
    alignment="tukey",
    window_length=24,
    min_sample=7,
    alpha=0.35,
    min_hs=11,
    max_hs=33,
    classic_center=11,
):
    n_sample_points = np.shape(spikes)[1]
    sample_points = np.arange(n_sample_points)
    dense_sample_points = np.arange(0, n_sample_points, 1 / upsample_rate)

    interpolator = PchipInterpolator(sample_points, spikes, axis=1)
    spikes_dense = interpolator(dense_sample_points)

    if alignment == "tukey":
        min_index = np.argmin(spikes_dense, axis=1)

        window = tukey(n_sample_points * upsample_rate, alpha=alpha)
        spikes_tukeyed = spikes_dense * window
        center = 12 * upsample_rate  # make this optional later

        spikes_aligned_dense = np.zeros(np.shape(spikes_tukeyed))

        # We apply circular shift to the spikes so that they are all aligned
        # to their respective minimums at the center point
        for count, row in enumerate(spikes_tukeyed):
            spikes_aligned_dense[count] = np.roll(row, -min_index[count] + center)
        # Note: It is very important that the downsampling is somehow
        #       Aligned to the minimum of each spike.
        downsample_points = np.arange(0, n_sample_points * upsample_rate, upsample_rate)

        spikes_aligned = spikes_aligned_dense[:, downsample_points]

    elif alignment == "classic":
        # WARNING: THIS IMPLEMENTATION IS CURRENTLY ONLY FOR QUIROGA DATA
        #          A Generalized version will come soon.
        # avg_min = int(np.argmin(spikes_dense, axis=1).mean())
        min_hs = 11 * upsample_rate
        max_hs = 33 * upsample_rate
        min_index = np.argmin(spikes_dense[:, min_hs:max_hs], axis=1) + min_hs
        # center = 15 * upsample_rate
        spikes_aligned_dense = np.full(spikes_dense.shape, np.nan)
        for i, row in enumerate(spikes_dense):
            r = row[min_index[i] - classic_center * upsample_rate :]
            spikes_aligned_dense[i, : len(r)] = r
        spikes_aligned_dense = spikes_aligned_dense[:, : window_length * upsample_rate]
        downsample_points = np.arange(0, window_length * upsample_rate, upsample_rate)
        spikes_aligned = spikes_aligned_dense[:, downsample_points]

    return spikes_aligned
