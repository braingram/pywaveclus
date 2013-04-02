#!/usr/bin/env python

import contextlib
import logging
import os
import time
import sys

import numpy as np
import scipy.stats


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


@contextlib.contextmanager
def waiting_lock_dir(lock_dir, delay=1):
    while os.path.exists(lock_dir):
        logging.info("Found lock directory, waiting to recheck in %d..." % \
                delay)
        time.sleep(delay)
    os.makedirs(lock_dir)
    try:
        yield
    finally:
        os.rmdir(lock_dir)


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
        raise ValueError("time range start >= end: %s [%i:%i]" % \
                (timerange, start, end))
    return start, end


def old_parse_time_range(timerange, minVal, maxVal, toVal=int):
    if timerange.strip() == '':
        return minVal, maxVal
    elif ':' in timerange:
        tokens = timerange.split(':')
        assert len(tokens) == 2, "Invalid timerange[%s]" % timerange
        start = tokens[0]
        end = tokens[1]
    else:
        start = str(minVal)
        end = timerange

    try:
        startVal = toVal(start)
    except Exception as E:
        raise ValueError("Count not convert %s to %s: %s" % (start, toVal, E))
    try:
        endVal = toVal(end)
    except Exception as E:
        raise ValueError("Count not convert %s to %s: %s" % (end, toVal, E))

    return max(startVal, minVal), min(endVal, maxVal)


def find_extreme(direction):
    if direction == 'pos':
        return lambda x: x.argmax()
    elif direction == 'neg':
        return lambda x: x.argmin()
    elif direction == 'both':
        return lambda x: np.abs(x).argmax()
    else:
        raise ValueError("Unknown direction [%s] for find_extreme" % direction)
