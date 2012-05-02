#!/usr/bin/env python

import channelmapping


probes = {'nna': channelmapping.schemes}


def lookup_converter_function(probetype, from_scheme, to_scheme):
    return probes[probetype][from_scheme][to_scheme]


def list_probes():
    return probes.keys()


def iter_schemes(probetype):
    for (k, v) in probes[probetype].iteritems():
        yield k, v.keys()


def list_schemes(probetype):
    return list(iter_schemes(probetype))
