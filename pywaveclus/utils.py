#!/usr/bin/env python

import logging
import os
import sys

import numpy as np
import scipy.stats

import cconfig


def load_config(local=None):
    bfn = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'pywavelus.ini')
    ufn = '~/.pywaveclus'
    local = 'pywaveclus.ini' if local is None else local
    return cconfig.TypedConfig(base=bfn, user=ufn, local=local)


def get_os():
    """
    Attempt to determine the operating system

    Returns
    -------
    systype : string
        Either 'linux', 'win', or 'mac'
    """
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform == 'win32':
        return 'win'
    elif sys.platform == 'darwin':
        return 'mac'
    else:
        # 'cygwin', 'os2', 'os2emx', 'riscos', 'atheos'
        raise ValueError("Unknown operating system: %s" % sys.platform)


def ks(coeffs):
    """
    A thin wrapper around scipy.stats.kstest to set ddof to 1 distribution
    to norm

    Parameters
    ----------
    coeffs : 1d array
        A single wavelet coefficient measured across many spikes

    Returns
    -------
    d : float
        D statistic from Kolmogorov-Smirnov test
    """
    d, _ = scipy.stats.kstest(scipy.stats.zscore(coeffs, ddof=1), 'norm')
    return d


def chunk(n, chunksize, overlap=0):
    """
    Chunk generator
    """
    for i in xrange((n / chunksize) + 1):
        if (i * chunksize) >= n:
            return
        if ((i + 1) * chunksize + overlap) < n:
            yield (i * chunksize, (i + 1) * chunksize + overlap)
        else:
            yield (i * chunksize, n)


def error(string, exception=Exception):
    logging.error(string)
    raise exception(string)


def parse_value(string, default, tmin, tmax, ttype):
    if string.strip() == '':
        return ttype(default)

    if '.' in string:
        return ttype(float(string) * tmax)

    if '+' in string:
        return ttype(string) + tmin

    if '-' in string:
        return tmax + ttype(string)

    return ttype(string)


def parse_time_range(timerange, tmin, tmax, ttype=int):
    """
    possible time ranges:
        '' -> min, max
        '1' -> min, 1
        '1:' -> 1, max
        ':1' -> min, 1
        '0.5:' -> .5 * max, max
        ':0.5' -> min, .5 * max
        ':+1' -> min, min + 1
        '-10:' -> max - 10, max
        ':-10' -> min, max - 10
    """
    # '' -> min, max
    if timerange.strip() == '':
        return ttype(tmin), ttype(tmax)

    # '1' -> min, 1
    if ':' in timerange:
        start, end = timerange.split(':')
    else:
        start, end = '', timerange

    start = parse_value(start, tmin, tmin, tmax, ttype)
    # passing in start rather than tmin allows '+1' to be relative to start
    end = parse_value(end, tmax, start, tmax, ttype)

    start = max(start, tmin)
    end = min(end, tmax)

    if start >= end:
        raise ValueError("time range start >= end: %s [%i:%i]" %
                         (timerange, start, end))
    return start, end


def find_extreme(direction):
    if direction == 'pos':
        return lambda x: x.argmax()
    elif direction == 'neg':
        return lambda x: x.argmin()
    elif direction == 'both':
        return lambda x: np.abs(x).argmax()
    else:
        raise ValueError("Unknown direction [%s] for find_extreme" % direction)


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
    if type(data) != np.ndarray:
        data = np.array(data)

    if direction == 'pos':
        return np.where(data > threshold)[0]
    if direction == 'neg':
        return np.where(data < -threshold)[0]
    if direction == 'both':
        return np.union1d(np.where(data > threshold)[0],
                          np.where(data < -threshold)[0])
