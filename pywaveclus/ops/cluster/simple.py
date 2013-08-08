#!/usr/bin/env python
"""

_, p = scipy.stats.normaltest(fvs)
if p < 0.001:
    good feature!

Simple features

1) peak-to-peak: value at wf[pre] - trough
2) width
4) power (signal ** 2)

other
1) refractory period
2) just peak
3) just trough
"""


import numpy

from . import pca


def peak_to_peak(w, p):
    if w[p] > 0:
        return w[p] - w.min()
    return w.max() - w[p]


def width(w, p):
    aw = numpy.abs(numpy.array(w))
    s = numpy.where(aw[:p] < aw[p] / 2.)[0]
    if len(s) == 0:
        return numpy.nan
    s = s[-1]
    e = numpy.where(aw[p:] < aw[p] / 2.)[0]
    if len(e) == 0:
        return numpy.nan
    e = e[0] + p
    return e - s


def power(w, p):
    return numpy.sum(w ** 2)


def peak_to_trough_time(w, p):
    if w[p] > 0:
        return w[p:].argmin()
    return w[p:].argmax()


def peak(w, p):
    return w[p]


def trough(w, p):
    if w[p] > 0:
        return w.argmin()
    return w.argmax()


feature_list = [
    peak_to_peak,
    width,
    power,
    peak_to_trough_time,
    peak,
    trough,
]


def features(wfs, p, n):
    if n > len(feature_list):
        raise ValueError(
            "Only %i simple features are available, %i requested" %
            (len(feature_list), n))
    # reshape to [index, wf]
    wfs = pca.stack_waveforms(wfs)
    fs = numpy.empty((len(wfs), n), dtype='f8')
    for i in xrange(n):
        fs[:, i] = map(lambda w: feature_list[i](w, p), wfs)
    return fs, {'features': [feature_list[i].__name__ for i in xrange(n)]}
