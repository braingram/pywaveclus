#!/usr/bin/env python

import numpy


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


def find_spikes(data, threshold, direction, minwidth, ref):
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


def find_crossings(data, threshold, direction='neg'):
    """
    Find the sub/super-threshold indices of data

    Parameters
    ----------
    data : 1d array
        Datapoints

    threshold: float
        Threshold (possibly calculated using calculate_threshold)
        Must be positive

    direction: string
        Must be one of the following:
            'pos' : find points more positive than threshold
            'neg' : find points more negative than -threshold
            'both': find points > threshold or < -threshold

    Returns
    -------
    crossings : 1d array
        Indices in data that cross the threshold
    """
    assert direction in ['pos', 'neg', 'both'], \
            "Unknown direction[%s]" % direction
    assert threshold > 0, "Threshold[%d] must be > 0" % threshold
    if type(data) != numpy.ndarray:
        data = numpy.array(data)

    if direction == 'pos':
        return numpy.where(data > threshold)[0]
    if direction == 'neg':
        return numpy.where(data < -threshold)[0]
    if direction == 'both':
        return numpy.union1d(numpy.where(data > threshold)[0], \
                numpy.where(data < -threshold)[0])
