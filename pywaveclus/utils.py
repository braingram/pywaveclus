#!/usr/bin/env python

import logging
import sys

import numpy


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
        raise ValueError("time range start >= end: %s [%i:%i]" % \
                (timerange, start, end))
    return start, end


def find_extreme(direction):
    return {'pos': lambda x: x.argmax(),
            'neg': lambda x: x.argmin(),
            'both': lambda x: numpy.abs(x).argmax(),
            }[direction]
