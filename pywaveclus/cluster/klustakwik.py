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

def cluster(waveforms, nfeatures, minclusters, maxclusters, tmp = '/tmp', quiet = True):
    """
    method: klustakwik
    nfeatures: 3
    minclusters: 3
    maxclusters: 5
    """
    
    tempdir = tempfile.mkdtemp(dir=tmp, suffix='_pywaveclus')
    
    datafile = "/".join((tempdir, "k_input.fet.1"))
    outfile = "/".join((tempdir, "k_input.clu.1"))
    
    features = dsp.pca.features(waveforms, nfeatures)
    #features = dsp.ica.features(waveforms, nfeatures)
    
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
    
    # # reorganize clusters
    # logging.debug("Klustakwik found %i clusters" % nclusters)
    # clusterList = []
    # # cluster 1 is noise
    # for i in xrange(1,nclusters+1): # klustakwik uses 1-based numbering
    #     clusterList.append(np.where(clusters == i)[0])
    
    # np.savetxt('features.txt',features)
    
    return clusters, None
    
