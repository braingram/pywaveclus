#!/usr/bin/env python

#from .. import utils


def simple(data, indices, pre, post):
    waves = []
    for i in indices:
        waves.append(data[i - pre:i + post])
    return waves
