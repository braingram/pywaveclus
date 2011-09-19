#!/usr/bin/env python

import logging, os, shutil, stat, subprocess, sys, tempfile

import numpy as np

from .. import utils

def find_temperature_using_minimum_cluster(tree, nclusters, minclus):
    # find good 'temperature' for clustering
    # dt = np.diff(tree,axis=0)[:,4:4+nclusters] # only consider n clusters, this is based on WaveClus
    logging.debug("Cluster temperature threshold: %i" % minclus)
    # temp1 = np.where(np.any(ct1[:,4:4+nclusters+1] > thresh,1))[0][-1] # find max temp with 1 clus > thresh
    goodtemps = np.where(np.any(tree[:,4:4+nclusters] > minclus,1))[0]
    if len(goodtemps) == 0:
        temp = 1
    else:
        temp = goodtemps[-1]
    if temp == 0: temp = 1 # based on WaveClus, to overcome first temperature being all 1 spin
    # temp = len(np.where(np.max(dt,1) > minclus)[0])
    # if temp == 0 and tree[0,nclusters+1] < minclus:
        # temp += 1 # based on WaveClus... all seems arbitrary :-/
    logging.debug("Cluster temperature: %i" % temp)
    return temp

def find_temperature_using_noise_clusters(tree, nclusters, nnoiseclusters):
    """
    Find temperature without using a minimum cluster size
    """
    ctree = tree[:,4:4+nclusters] # tree of [:,nclusters]
    dtree = np.diff(ctree, axis = 0)
    votingtemps = dtree[:,nnoiseclusters:].argmax(0)
    votes = np.sum(dtree[votingtemps,1:] > 0, 1) #TODO maybe >=?
    temp = votingtemps[votes.argmax()] + 1
    logging.debug("Cluster temperature: %i" % temp)
    return temp

def cluster(features, tmp = '/tmp', mintemp = 0, maxtemp = 0.201, tempstep = 0.01,
        swcycles = 100, knn = 11, nclusters = 5, nnoiseclusters = 1, quiet = True):
    """
    Super-paramagnetic clustering
    
    Parameters
    ----------
    features : 2d array
        Wavelet coefficients ordered [spike, coeff]
    tmp : string
        Temporary directory in which scratch files will be created and deleted
    mintemp : float
        Minimum temperature for spc algorithm
    maxtemp : float
        Maximum temperature for spc algorithm
    tempstep : float
        Size of temperature change between spc iterations
    swcycles : int
        Number of montecarlo iterations to use in spc
    knn : int
        Number of nearest neighbors to use in spc
    nclusters : int
        Number of clusters to return
    nnoiseclusters : int
        Number of expected 'noise' clusters, used for temperature calculation
    quiet : bool
        Suppress the spc executable stdout
    
    Returns
    -------
    clusters : list of 1d arrays
        List containing indices of each cluster. Length = nclusters + 1
        clusters[0] contains indices for unmatched spikes
    cdata : 2d array
        Spc cluster data for each temperature
    tree : 2d array
        Spc status data for each temperature
    """
    tempdir = tempfile.mkdtemp(dir=tmp, suffix='_pywaveclus')
    
    datafile = "/".join((tempdir, "features"))
    outfile = "/".join((tempdir, "clusters"))
    
    runfile = """
            NumberOfPoints: %i
            DataFile: %s
            Dimentions: %i
            MinTemp: %f
            MaxTemp: %f
            TempStep: %f
            OutFile: %s
            SWCycles: %i
            KNearestNeighbours: %i
            MSTree|
            DirectedGrowth|
            SaveSuscept|
            WriteLables|
            WriteCorFile~
    """ % (features.shape[0], datafile, features.shape[1], mintemp, maxtemp, tempstep,
            outfile, swcycles, knn)
    
    # write runfile
    runfilename = 'spc.run'
    rf = open("/".join((tempdir, runfilename)),"w")
    rf.write(runfile)
    rf.close()
    
    # write datafile
    np.savetxt(datafile, features)
    
    # copy correct executable
    exefile = os.path.dirname(os.path.abspath(__file__)) + '/../bin/spc_' + utils.get_os()
    logging.debug("Found spc executable: %s" % exefile)
    shutil.copy2(exefile, tempdir)
    exefile = 'spc_' + utils.get_os()
    
    # run executable
    olddir = os.getcwd()
    os.chdir(tempdir)
    # make file executable
    os.chmod(exefile, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    if quiet:
        with open(os.devnull, 'w') as fp:
            retcode = subprocess.call(['./spc_' + utils.get_os(), runfilename], stdout=fp)
    else:
        retcode = subprocess.call(['./spc_' + utils.get_os(), runfilename])
    os.chdir(olddir)
    logging.debug("Clustering returned: %i" % retcode)
    
    # read output
    cdata = np.loadtxt(outfile+'.dg_01.lab')
    tree = np.loadtxt(outfile+'.dg_01')
    
    # clean up
    shutil.rmtree(tempdir)
    
    # find good 'temperature' for clustering
    # temp = spc_find_temperature(tree, nclusters, minclus)
    temp = find_temperature_using_noise_clusters(tree, nclusters, nnoiseclusters)
    # temp = spc_find_temperature_2(tree, nclusters)
    
    clusters = []
    for i in xrange(nclusters):
        clusters.append(np.where(cdata[temp,2:] == i)[0])
    # unmatched = np.setdiff1d(range(len(features)), [c for cluster in clusters for c in cluster])
    unmatched = np.setdiff1d(range(len(features)), np.hstack(clusters))
    clusters = [unmatched,] + clusters
    
    return clusters_to_indices(clusters), (cdata, tree)

def spc_recluster(nspikes, cdata, tree, temp, nclusters = 5):
    """
    Use existing spc data but recluster for a different temperature
    
    Parameters
    ----------
    nspikes : int
        Number of spikes. Needed to determine if any are unmatched
    cdata : 2d array
        Spc cluster data for each temperature
    tree : 2d array
        Spc status data for each temperature
    temp : int
        New temperature at which to cluster
    nclusters : int
        Number of clusters to return
    
    Returns
    -------
    clusters : list of 1d arrays
        List containing indices of each cluster. Length = nclusters + 1
        clusters[0] contains indices for unmatched spikes
    cdata : 2d array
        see parameters
    tree : 2d array
        see parameters
    """
    # TODO: can I just use cdata.shape to get the number of spikes?
    clusters = []
    for i in xrange(nclusters):
        clusters.append(np.where(cdata[temp,2:]==i)[0])
    # TODO is it more efficient to do an hstack rather than a double list comprehension
    # unmatched = np.setdiff1d(range(nspikes), [c for cluster in clusters for c in cluster])
    unmatched = np.setdiff1d(range(nspikes), np.hstack(clusters))
    clusters = [unmatched,] + clusters
    
    return clusters, cdata, tree

def clusters_to_indices(clusters):
    """
    
    Parameters
    ----------
    clusters : list of 1d arrays
        List containing indices of each cluster. Length = nclusters + 1
        clusters[0] contains indices for unmatched spikes
    
    Returns
    -------
    indices : 1d array of ints
        Array containing the cluster index for each spike
    """
    nspks = sum([len(c) for c in clusters])
    indices = np.zeros(nspks, dtype=int) - 1
    for i in xrange(nspks):
        for ci in xrange(len(clusters)):
            if i in clusters[ci]: indices[i] = ci
        if indices[i] == -1: raise IndexError("Spike index [%i] was not found in any cluster" % i)
    return indices