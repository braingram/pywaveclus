#!/usr/bin/env python


import klustakwik
import pca

__all__ = ['klustakwik', 'pca']


class KlustakwikCluster(object):
    def __init__(self, nfeatures, minclusters, maxclusters, minspikes, \
            separate, pre):
        self.nfeatures = nfeatures
        self.minclusters = minclusters
        self.maxclusters = maxclusters
        self.minspikes = minspikes
        self.separate = separate
        self.pre = pre

    def __call__(self, waveforms):
        return klustakwik.cluster(waveforms, self.nfeatures, \
                self.minclusters, self.maxclusters, self.separate, \
                self.pre, self.minspikes)


def get_cluster(cfg, section='cluster'):
    nfeatures = cfg.getint(section, 'nfeatures')
    minclusters = cfg.getint(section, 'minclusters')
    maxclusters = cfg.getint(section, 'maxclusters')
    minspikes = cfg.getint(section, 'minspikes')
    separate = cfg.get(section, 'separate')
    pre = cfg.getint(section, 'pre')

    return KlustakwikCluster(nfeatures, minclusters, maxclusters, minspikes, \
            separate, pre)
