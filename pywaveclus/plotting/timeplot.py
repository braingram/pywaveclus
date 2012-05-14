#!/usr/bin/env python

import pylab


def plot(times, amplitudes, clusters, max_n=20000, cm=None, \
        skwargs=None):
    """
    """
    assert len(times) == len(amplitudes)
    assert len(amplitudes) == len(clusters)

    if len(times) == 0:
        return

    if skwargs is None:
        skwargs = {}
    if cm is None:
        cm = pylab.cm.jet

    nclus = max(clusters) + 1
    for cl in xrange(nclus):
        color = cm(cl / float(nclus))
        cinds = clusters == cl
        st = times[cinds]
        sa = amplitudes[cinds]
        if len(st) > max_n:
            ds = int(len(st) / float(max_n) + 1)
            st = st[::ds]
            sa = sa[::ds]
        pylab.scatter(st, sa, color=color, **skwargs)
