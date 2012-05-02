#!/usr/bin/env python

import tables


def make_spike_table_description(pre, post, nfeatures):
    #TODO add features
    class SpikeTableDescription(tables.IsDescription):
        time = tables.Int64Col()
        wave = tables.Float64Col(shape=(pre + post))
        clu = tables.UInt8Col()
        #features = tables.Float64Col(shape=(nfeatures))
    return SpikeTableDescription


def read_spike_times(hfile, ):
    pass


def write_spike_times(hfile, times, clusters, cfg):
    pass


def read_spike_waveforms(hfile, ):
    pass


def write_spike_waveforms(hfile, ):
    pass


def read_clustering_info(hfile, ):
    pass


def write_clustering_info(hfile, ):
    pass
