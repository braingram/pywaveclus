#!/usr/bin/env python

import numpy


def peak_to_trough(waves, pre=28, sign=None, constrain=True, full=False):
    """
    Inspired by
        Ravassard etal 2013 Science May 2 2013

    Use the peak_to_trough time (and firing rate) to
    split pyramidal from inhibitory cells
    """
    # TODO deal with + and - spikes
    # try to guess if all spikes are + or -
    if sign is None:
        if numpy.all(waves[:, pre] > 0):
            sign = 1
        elif numpy.all(waves[:, pre] < 0):
            sign = -1
        else:
            raise ValueError(
                "Unable to guess sign, some waveforms +, some -: %s, %s"
                % (numpy.sum(waves[:, pre] > 0), numpy.sum(waves[:, pre] < 0)))
    if sign == 1:
        pf = numpy.argmax
        tf = numpy.argmin
    elif sign == -1:
        pf = numpy.argmin
        tf = numpy.argmax
    else:
        raise ValueError("Invalid sign, must be 1 or -1: %s" % sign)

    ps = pf(waves, 1)
    if constrain:
        ts = tf(waves[:, pre:], 1) + pre
    else:
        ts = tf(waves, 1)

    if full:
        return ts - ps, ps, ts

    return ts - ps
