#!/usr/bin/env python

#from .. import utils

import numpy


def simple(data, indices, pre, post):
    waves = numpy.array([]).reshape((0, pre + post))
    for i in indices:
        #wave = data[i - pre: i + post]
        #if len(wave) == pre + post:
        #    waves = numpy.vstack((waves, wave[numpy.newaxis, :]))
        waves = numpy.vstack((waves, data[numpy.newaxis, i - pre:i + post]))
        #waves.append(data[i - pre:i + post])
    return waves
