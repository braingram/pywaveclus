#!/usr/bin/env python

import numpy

from ... import dsp


class Detector(object):
    def __init__(self, baseline=None, nthresh=None):
        pass

    def __call__(self, data, **kwargs):
        return find_spikes(data, *self.args, **self.kwargs)


def calculate_threshold(data, n=5):
    """
    Calculate the threshold for spike detection given a subset of the data

    Parameters
    ----------
    data : 1d array
        Data (usually a subset) used to calculate the threshold
    n : int,optional
        Number of estimated standard deviations of the background noise to
        use as the threshold

    Returns
    -------
    threshold : float
        Spike threshold

    Notes
    -----
    threshold = n * median( |data| ) / 0.6745

    Quian Quiroga R, Nadasdy Z, Ben-Shaul Y (2004)
        Unsupervised Spike Detection and Sorting with
        Wavelets and Superparamagnetic Clustering.
        Neural Comp 16:1661-1687
    """
    assert data.ndim == 1, "Calculate threshold requires a 1d array"
    return n * numpy.median(numpy.abs(data)) / 0.6745


def find_spikes(data, threshold, artifact, \
        direction='neg', minwidth=2, ref=44):
    indices = dsp.threshold.find_spikes(data, threshold, direction, \
            minwidth, ref)
    # remove artifacts
    return indices[abs(data[indices]) < artifact]
