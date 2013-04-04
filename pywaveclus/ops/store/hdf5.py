#!/usr/bin/env python

import tables


class SpikeStorage(object):
    def __init__(self, file):
        # table
        # info
        # cluster info (per channel)
        pass

    def save_timerange(self, start, end):
        # table attrs
        pass

    def create_spikes(self, chi, sis, sws):
        # table
        pass

    def load_waves(self, chi):
        # table
        pass

    def update_spikes(self, chi, clusters):
        # table
        pass

    def save_cluster_info(self, chi, info):
        # pickle dict
        pass

    def save_filename(self, chi, fn):
        # table attrs?
        pass

    def save_info(self, info):
        # pickle dict
        pass
