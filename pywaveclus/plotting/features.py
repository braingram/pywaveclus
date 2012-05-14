#!/usr/bin/env python

#import numpy
import pylab


def plot(features, clusters, max_n=20000, cm=None, \
        skwargs=None):
    """
    """
    assert len(features) == len(clusters), \
            "Features and clusters must have same length: %i, %i" \
            % (len(features), len(clusters))
    if len(features) == 0:
        return
    assert features.ndim > 1, "Features must have > 1 dimension: %i" \
            % features.ndim

    if skwargs is None:
        skwargs = {'alpha': 0.5, 's': 1}
    if cm is None:
        cm = pylab.cm.jet

    ndim = features.shape[1]
    nclus = max(clusters) + 1
    for cl in xrange(nclus):
        fs = features[clusters == cl]
        if len(fs) > max_n:
            fs = fs[::int(len(fs) / float(max_n) + 1)]
        color = cm(cl / float(nclus))
        for x in xrange(ndim):
            for y in xrange(ndim):
                pylab.subplot(ndim, ndim, x + y * ndim + 1)
                pylab.scatter(fs[:, x], fs[:, y], color=color, **skwargs)
