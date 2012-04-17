#!/usr/bin/env python

import klustakwik
import skl
import spc

from .. import dsp

__all__ = ['klustakwik', 'skl', 'spc']


def cluster_from_config(cfg):
    method = cfg.get('cluster', 'method')

    if method == 'spc':
        raise NotImplemented("SPC cannot handle new 'adjacent files'")
        nfeatures = cfg.getint('cluster', 'nfeatures')
        levels = cfg.getint('cluster', 'levels')
        wtype = cfg.get('cluster', 'wavelet')
        nclusters = cfg.getint('cluster', 'nclusters')
        nnoise = cfg.getint('cluster', 'nnoise')
        minspikes = cfg.getint('cluster', 'minspikes')

        return lambda x: spc.cluster(dsp.wavelet.features(x, nfeatures, \
                levels, wavelet=wtype), nclusters=nclusters, \
                nnoiseclusters=nnoise)
    elif method == 'klustakwik':
        nfeatures = cfg.getint('cluster', 'nfeatures')
        minclusters = cfg.getint('cluster', 'minclusters')
        maxclusters = cfg.getint('cluster', 'maxclusters')
        ftype = cfg.get('cluster', 'featuretype')
        separate = cfg.get('cluster', 'separate')
        pre = cfg.getint('detect', 'pre')
        minspikes = cfg.getint('cluster', 'minspikes')
        if separate == 'peak':
            separate = True
        else:
            separate = False

        return lambda x: klustakwik.cluster(x, nfeatures, ftype, \
                minclusters, maxclusters, separate, pre, minspikes)
    elif method == 'kmeans':
        nfeatures = cfg.getint('cluster', 'nfeatures')
        ftype = cfg.get('cluster', 'featuretype')
        nclusters = cfg.getint('cluster', 'nclusters')
        separate = cfg.get('cluster', 'separate')
        pre = cfg.getint('detect', 'pre')
        minspikes = cfg.getint('cluster', 'minspikes')
        if separate == 'peak':
            separate = True
        else:
            separate = False
        return lambda x: skl.cluster(x, nfeatures, ftype, nclusters, \
                separate, pre, minspikes)
    elif method == 'gmm':
        nfeatures = cfg.getint('cluster', 'nfeatures')
        ftype = cfg.get('cluster', 'featuretype')
        nclusters = cfg.getint('cluster', 'nclusters')
        separate = cfg.get('cluster', 'separate')
        pre = cfg.getint('detect', 'pre')
        minspikes = cfg.getint('cluster', 'minspikes')
        cvtype = cfg.get('cluster', 'cvtype')
        if separate == 'peak':
            separate = True
        else:
            separate = False
        return lambda x: skl.gmm(x, nfeatures, ftype, nclusters, \
                separate, pre, minspikes, cvtype)

    else:
        raise ValueError("Unknown cluster method: %s" % method)
