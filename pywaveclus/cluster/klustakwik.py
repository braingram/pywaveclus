#!/usr/bin/env python

import logging
import os
import shutil
import stat
import subprocess
import tempfile

import numpy as np

from .. import dsp
from .. import utils


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
    # # reorganize clusters
    # what I want is a function that maps old to new cluster index
    #   where f(1) is always 0, but f(>1) returns a number based on
    # the # of spikes
    min_cluster_i = clusters.min()  # should be 1
    if min_cluster_i == 0:
        # Klustakwik should NOT return any spike with a cluster of 0
        # the code is not setup for this case so throw an exception
        raise ValueError("Klustakwik returned a cluster with index 0")

    max_cluster_i = clusters.max()
    if min_cluster_i == max_cluster_i:
        # only 1 cluster was found, just return the data without resorting
        if min_cluster_i == 1:
            return np.zeros_like(clusters)  # all noise
        else:
            return np.ones_like(clusters)  # all 1 signal

    # count non-noise (!= 1) clusters
    cluster_counts = {}
    for i in xrange(min_cluster_i, max_cluster_i + 1):
        if i != 1:
            cluster_counts[i] = len(clusters[clusters == i])

    # sort by count (and add 1)
    sorted_cluster_counts = [1] + sorted(cluster_counts, \
            key=lambda k: cluster_counts[k])[::-1]
    old_to_new_map = dict(zip(sorted_cluster_counts, \
            map(sorted_cluster_counts.index, sorted_cluster_counts)))

    return np.array(map(lambda c: old_to_new_map[c], clusters))


def cluster(waveforms, nfeatures, featuretype, minclusters, maxclusters, \
        separate, pre, minspikes, tmp='/tmp', quiet=True):
    """
    method: klustakwik
    nfeatures: 3
    minclusters: 3
    maxclusters: 5
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
            return cluster(waveforms, nfeatures, featuretype, minclusters, \
                maxclusters, False, pre, tmp=tmp, quiet=quiet)

        pwaves = waveforms[pinds]
        nwaves = waveforms[ninds]

        pc, pi = cluster(pwaves, nfeatures, featuretype, minclusters, \
                maxclusters, False, pre, minspikes, tmp=tmp, quiet=quiet)
        nc, ni = cluster(nwaves, nfeatures, featuretype, minclusters, \
                maxclusters, False, pre, minspikes, tmp=tmp, quiet=quiet)

        info = dict([('p' + k, v) for k, v in pi.iteritems()])
        info.update(dict([('n' + k, v) for k, v in ni.iteritems()]))

        clusters = np.zeros(len(pc) + len(nc), dtype=pc.dtype)
        # merge cluster 0 from each
        # interleave other clusters
        pc *= 2  # keeps 0 -> 0
        nc = nc * 2 - 1  # makes 0 -> -1
        ti = nc > 0
        clusters[pinds] = pc
        clusters[ninds[ti]] = nc[ti]

        return remove_empty(clusters), {}

    if len(waveforms) < minspikes:
        return np.zeros(len(waveforms)), {}
    tempdir = tempfile.mkdtemp(dir=tmp, suffix='_pywaveclus')

    datafile = "/".join((tempdir, "k_input.fet.1"))
    outfile = "/".join((tempdir, "k_input.clu.1"))

    if featuretype == 'pca':
        features, pca_info = dsp.pca.features(waveforms, nfeatures)
    elif featuretype == 'ica':
        features = dsp.ica.features(waveforms, nfeatures)
    else:
        raise ValueError("Unknown feature type[%s]" % featuretype)

    with open(datafile, 'w') as df:
        df.write('%i\n' % nfeatures)
        np.savetxt(df, features)

    # copy correct executable
    exefile = os.path.dirname(os.path.abspath(__file__)) + \
            '/../bin/klustakwik_' + utils.get_os()
    logging.debug("Found klustakwik executable: %s" % exefile)
    shutil.copy2(exefile, tempdir)
    exefile = 'klustakwik_' + utils.get_os()

    # run executable
    olddir = os.getcwd()
    os.chdir(tempdir)
    # make file executable
    os.chmod(exefile, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | \
            stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    cmd = './klustakwik_%s k_input 1 -UseFeatures %s -MinClusters %i ' \
            '-MaxClusters %i -Screen 0 -Log 0' %\
            (utils.get_os(), '1' * nfeatures, minclusters, maxclusters)
    logging.debug("Running klustakwik: %s" % cmd)
    if quiet:
        with open(os.devnull, 'w') as fp:
            retcode = subprocess.call(cmd.split(), stdout=fp)
    else:
        retcode = subprocess.call(cmd.split())
    os.chdir(olddir)
    logging.debug("Clustering returned: %i" % retcode)

    clusters = np.loadtxt(outfile, dtype=np.int32, skiprows=1)
    #with open(outfile, 'r') as outfile:
    #    nclusters = int(outfile.readline())

    #np.save("features_%i" % len(features), features)

    shutil.rmtree(tempdir)

    clusters = sort_clusters(clusters)

    return clusters, pca_info
