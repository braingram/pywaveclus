#!/usr/bin/env python

import numpy


def instantaneous(spike_times, sf=44100., pad=False):
    """
    pad (boolean):
        bad beginning of return array with a duplicate of the first value
    """
    isi = spike_times[1:] - spike_times[:-1]
    if pad:
        isi = numpy.hstack((isi[0], isi))
    return 1. / (isi / float(sf))


def average_rate(spike_times, sf=44100.):
    return numpy.mean(instantaneous(spike_times, sf, pad=False))
