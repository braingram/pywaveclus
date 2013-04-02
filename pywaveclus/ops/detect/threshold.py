#!/usr/bin/env python

import logging

import numpy as np

from ... import utils


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
    return n * np.median(np.abs(data)) / 0.6745


def find_spikes(data, threshold, artifact, direction, ref, minwidth, slop):
    crossings = utils.find_crossings(data, threshold, direction)
    if not len(crossings):
        return []
    find_extreme = utils.find_extreme(direction)

    sis = []
    start = crossings[0]
    end = -ref

    for (i, c) in enumerate(crossings[:-1]):
        if (crossings[i + 1] - c) > slop + 1:
            # next crossing is NOT part of this spike
            if ((c - end) >= ref) and ((c - start) >= minwidth):
                pi = find_extreme(data[start:c] + start)
                if abs(data[pi]) < artifact:
                    sis.append(pi)
                end = c
            start = crossings[i + 1]

    if (start > end):
        end = crossings[-1]
        if (end - start) > minwidth:
            pi = find_extreme(data[start:end] + start)
            if abs(data[pi]) < artifact:
                sis.append(pi)
    return sis


def old_find_spikes(data, threshold, artifact, direction='neg', prew=40,
                    postw=88, ref=44, minwidth=2,  slop=0, oversample=10):
    """
    Parameters:
        data :
        threshold :
        artifact : threshold at which spike is considered an artifact
        direction :
        prew :
        postw :
        ref :
        minwidth :
        slop :
        oversample :
    """
    crossings = utils.find_crossings(data, threshold, direction)
    if not len(crossings):
        logging.debug(
            "No threshold crossings found over %f to %f at T:%f" %
            (data.min(), data.max(), threshold))
        return [], []

    find_extreme = utils.find_extreme(direction)

    spikeindices = []

    #TODO artifact detection?
    ndata = len(data)
    start = crossings[0]
    end = -ref

    for (i, c) in enumerate(crossings[:-1]):
        if (crossings[i + 1] - c) > slop + 1:
            # next crosuings is NOT part of this spike
            if ((c - end) >= ref) and ((c - start) >= minwidth):
                peaki = find_extreme(data[start:c]) + start
                fullwave = data[peaki - (prew * 2):peaki + (postw * 2)]
                if len(fullwave) == ((prew + postw) * 2) and \
                        abs(data[peaki]) < artifact:
                    spikeindices.append(peaki)
                    #spikewaveforms.append(dsp.interpolate.interpolate_peak(\
                    #        fullwave, prew, postw, oversample, find_extreme))
                    end = c
            start = crossings[i + 1]

    # get last spike
    if (start > end):
        end = crossings[-1]
        if (end - start) > minwidth:
            # get spike
            # spikebounds.append([start,crossings[-1]])
            peaki = find_extreme(data[start:end]) + start
            fullwave = data[peaki - (prew * 2):peaki + (postw * 2)]
            if len(fullwave) == ((prew + postw) * 2) and \
                    abs(data[peaki]) < artifact:
                spikeindices.append(peaki)
                #spikewaveforms.append(dsp.interpolate.interpolate_peak(\
                #        fullwave, prew, postw, oversample, find_extreme))
            else:
                logging.debug(
                    "Spike ran off end of data at frame %i[%i]" %
                    (peaki, len(fullwave)))

    return spikeindices  # , spikewaveforms
