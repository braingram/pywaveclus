#!/usr/bin/env python


import numpy as np
import scikits.learn.cluster

from .. import dsp


def remove_empty(clusters):
    # skip cluster 0
    nzc = np.unique(clusters)
    nzc = nzc[nzc > 0]
    offset = nzc - np.arange(1, 1 + len(nzc))
    for (c, o) in zip(nzc, offset):
        if o != 0:
            clusters[clusters == c] -= o
    return clusters


def sort_clusters(clusters):
    cis = sorted(np.unique(clusters), \
            key=lambda u: -np.sum(clusters[clusters == u]))
    for (ni, ci) in enumerate(cis):
        clusters[clusters == ci] = ni
    return clusters


def cluster(waveforms, nfeatures, featuretype, nclusters, \
        separate, pre, tmp='/tmp', quiet=True):
    """
    """
    if separate:
        # separate waveforms into + and -, cluster separately
        waveforms = np.array(waveforms)
        if waveforms.ndim == 3:
            signs = np.sign(waveforms[:, 0, pre])
        elif waveforms.ndim == 2:
            signs = np.sign(waveforms[:, pre])
        else:
            raise ValueError("Invalid waveforms dimensions: %s" % \
                    waveforms.shape)

        pinds = np.where(signs == 1)[0]
        ninds = np.where(signs == -1)[0]

        # if no + or no -, just cluster the other
        if (len(pinds) == 0) or (len(ninds) == 0):
            return cluster(waveforms, nfeatures, featuretype, nclusters, \
                False, pre, tmp=tmp, quiet=quiet)

        pwaves = waveforms[pinds]
        nwaves = waveforms[ninds]

        pc, pi = cluster(pwaves, nfeatures, featuretype, nclusters, \
                False, pre, tmp=tmp, quiet=quiet)
        nc, ni = cluster(nwaves, nfeatures, featuretype, nclusters, \
                False, pre, tmp=tmp, quiet=quiet)

        info = dict([('p' + k, v) for k, v in pi.iteritems()])
        info.update(dict([('n' + k, v) for k, v in ni.iteritems()]))

        clusters = np.zeros(len(pc) + len(nc), dtype=pc.dtype)
        # merge cluster 0 from each
        # interleave other clusters
        pc *= 2  # keeps 0 -> 0
        nc = nc * 2 + 1  # makes 0 -> 1
        ti = nc > 0
        clusters[pinds] = pc
        clusters[ninds[ti]] = nc[ti]

        return remove_empty(clusters), info

    if featuretype == 'pca':
        features, pca_info = dsp.pca.features(waveforms, nfeatures)
    elif featuretype == 'ica':
        features = dsp.ica.features(waveforms, nfeatures)
    else:
        raise ValueError("Unknown feature type[%s]" % featuretype)

    kmeans = scikits.learn.cluster.KMeans(k=nclusters)
    np.save('waves_%i' % features.shape[0], waveforms)
    np.save('features_%i' % features.shape[0], features)
    kmeans.fit(features)

    clusters = kmeans.labels_

    clusters = sort_clusters(clusters)

    return clusters, pca_info
