#!/usr/bin/env python

import numpy

from ... import utils


class Detector(object):
    def __init__(self, baseline=None, nthresh=None, reader=None, filt=None):
        assert baseline is not None, "baseline must have a value"
        assert nthresh is not None, "nthresh must have a value"
        assert reader is not None, "reader must have a value"
        assert filt is not None, "filt must have a value"

        # calculate threshold
        start, end = utils.parse_time_range(baseline, 0, len(reader))
        reader.seek(start)
        self.threshold = calculate_threshold(filt(reader.read(end - start)), \
                nthresh)

    def __call__(self, data, **kwargs):
        return find_spikes(data, self.threshold, *self.args, **self.kwargs)


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
    indices = find_all_spikes(data, threshold, direction, minwidth, ref)
    # remove artifacts
    return indices[abs(data[indices]) < artifact]


def detect_sub(data, mf, sf, minwidth, ref):
    """
    Detection subroutine

    Parameters
    ----------
    data : 1d array
        Where to find the spikes

    mf : function
        Maxima/Minima/Extreme function, used to find the precise
        index where the spike occured

    sf : function
        Function to measure if a point is sub-threshold

    minwidth : int
        See find_spikes

    ref : int
        See find_spikes

    Returns
    -------
    spike_indices : 1d array
        See find_spikes
    """
    sub = numpy.where(sf(data))[0]
    sis = sub[:-1]
    sws = sub[1:] - sub[:-1]

    # remove spikes < minwidth
    cis = numpy.where(sws >= minwidth)[0]
    sis = sis[cis]
    sws = sws[cis]

    # remove spikes < ref samples apart
    di = numpy.empty_like(sis)
    di[0] = ref
    di[1:] = sis[1:] - sis[:-1]
    sis = sis[di >= ref]
    sws = sws[di >= ref]

    s = numpy.empty_like(sis)
    for i in xrange(len(sis)):
        si = sis[i]
        sw = sws[i]
        s[i] = si + mf(data[si:si + sw])  # custom
        #s[i] = si + mf(data[si:si + ref])  # custom

    return s


def detect_pos(data, ht, minwidth, ref):
    return detect_sub(data, lambda d: d.argmax(), lambda d: d < ht, \
            minwidth, ref)


def detect_neg(data, ht, minwidth, ref):
    return detect_sub(data, lambda d: d.argmin(), lambda d: d > -ht, \
            minwidth, ref)


def detect_both(data, ht, minwidth, ref):
    return detect_sub(data, lambda d: numpy.abs(d).argmax(), \
            lambda d: numpy.abs(d) < ht, \
            minwidth, ref)


def find_all_spikes(data, threshold, direction, minwidth, ref):
    """
    Find the indices of peaks/troughs in data

    Parameters
    ----------
    data : 1d array
        Datapoints in which to find spikes

    threshold : float
        Threshold at which a spike detection event is triggered

    direction : string ('pos', 'neg', or 'both')
        Direction of spikes to detect

    minwidth : int
        Minimum super-threshold spike width (in samples)

    ref : int
        Refractory period (in samples) after a spike
        during which no spike should be triggered

    Returns
    -------
    spike_indices : 1d array
        Indices where the spike extremes (maxima/minima) were found
    """
    return {'pos': detect_pos,
            'neg': detect_neg,
            'both': detect_both,
            }[direction](data, threshold, minwidth, ref)
