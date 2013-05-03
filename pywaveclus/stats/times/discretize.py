#!/usr/bin/env python

import numpy


def nonoverlapping(times, binsize=4410, start=None):
    if start is None:
        start = times[0]
    end = start + binsize
    bins = []
    n = 0
    for t in times:
        while (t > end):
            bins.append(n)
            start = end
            end = start + binsize
            n = 0
        n += 1
    bins.append(n)
    return numpy.array(bins)
