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

def sort_clusters(clusters):
    # # reorganize clusters
    # what I want is a function that maps old to new cluster index
    #   where f(1) is always 0, but f(>1) returns a number based on the # of spikes
    min_cluster_i = clusters.min() # should be 1
    if min_cluster_i == 0:
        # Klustakwik should NOT return any spike with a cluster of 0
        # the code is not setup for this case so throw an exception
        raise ValueError("Klustakwik returned a cluster with index 0")
    
    max_cluster_i = clusters.max()
    if min_cluster_i == max_cluster_i:
        # only 1 cluster was found, just return the data without resorting
        if min_cluster_i == 1:
            return np.zeros_like(clusters) # all noise
        else:
            return np.ones_like(clusters) # all 1 signal
    
    # count non-noise (!= 1) clusters
    cluster_counts = {}
    for i in xrange(min_cluster_i, max_cluster_i + 1):
        if i != 1:
            cluster_counts[i] = len(clusters[clusters == i])

    # sort by count (and add 1)
    sorted_cluster_counts = [1] + sorted(cluster_counts, key = lambda k: cluster_counts[k])[::-1]
    old_to_new_map = dict( zip( sorted_cluster_counts, \
            map(sorted_cluster_counts.index, sorted_cluster_counts)))

    return np.array(map(lambda c: old_to_new_map[c], clusters))

def cluster(waveforms, nfeatures, featuretype, minclusters, maxclusters, tmp = '/tmp', quiet = True):
    """
    method: klustakwik
    nfeatures: 3
    minclusters: 3
    maxclusters: 5
    """
    
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
    exefile = os.path.dirname(os.path.abspath(__file__)) + '/../bin/klustakwik_' + utils.get_os()
    logging.debug("Found klustakwik executable: %s" % exefile)
    shutil.copy2(exefile, tempdir)
    exefile = 'klustakwik_' + utils.get_os()

    # run executable
    olddir = os.getcwd()
    os.chdir(tempdir)
    # make file executable
    os.chmod(exefile, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    cmd = './klustakwik_%s k_input 1 -UseFeatures %s -MinClusters %i -MaxClusters %i -Screen 0' %\
            (utils.get_os(), '1'*nfeatures, minclusters, maxclusters)
    logging.debug("Running klustakwik: %s" % cmd)
    if quiet:
        with open(os.devnull, 'w') as fp:
            retcode = subprocess.call(cmd.split(), stdout=fp)
    else:
        retcode = subprocess.call(cmd.split())
    os.chdir(olddir)
    logging.debug("Clustering returned: %i" % retcode)
    
    clusters = np.loadtxt(outfile, dtype=np.int32, skiprows=1)
    with open(outfile,'r') as outfile:
        nclusters = int(outfile.readline())
    
    shutil.rmtree(tempdir)
    
    clusters = sort_clusters(clusters)
    
    return clusters, pca_info
    
