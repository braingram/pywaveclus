#!/usr/bin/env python


import numpy
import pylab


def plot(waveforms, clusters, max_n=20000, cm=None, \
        pkwargs=None, fkwargs=None, akwargs=None):
    """
    """
    assert len(waveforms) == len(clusters), \
            "waveforms and clusters must be same length: %i != %i" \
            % (len(waveforms), len(clusters))
    if len(waveforms) == 0:
        return
    if pkwargs is None:
        pkwargs = {}
    if fkwargs is None:
        fkwargs = {}
    if akwargs is None:
        akwargs = {}
    for a in [pkwargs, fkwargs, akwargs]:
        if a is None:
            a = {}
    if cm is None:
        cm = pylab.cm.jet
    if waveforms.ndim == 2:
        waveforms = waveforms[:, numpy.newaxis, :]
    nwaves, nch, nsamps = waveforms.shape
    nclus = max(clusters) + 1
    for cl in xrange(nclus):
        cinds = numpy.where(clusters == cl)
        ws = waveforms[cinds]
        color = cm(cl / float(nclus))
        if len(ws) == 0:
            continue

        if len(ws) > max_n:
            ws = ws[::int(len(ws) / float(max_n) + 1)]

        for ch in xrange(nch):
            pylab.subplot(nch, nclus, ch * nclus + cl + 1)
            pylab.plot(ws[:, ch, :].T, color=color, alpha=0.33, **pkwargs)
            av = numpy.average(ws[:, ch, :], 0)
            sd = numpy.std(ws[:, ch, :], 0, ddof=1)
            se = sd / numpy.sqrt(len(ws))
            pylab.fill_between(range(len(av)), av + se, av - se, \
                    color=color, alpha=0.75, **fkwargs)
            pylab.plot(av, color=color, **akwargs)
