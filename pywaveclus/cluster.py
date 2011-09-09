#!/usr/bin/env python

import logging, os, shutil, stat, subprocess, sys, tempfile

import numpy as np

def get_os():
    """
    Attempt to determine the operating system
    
    Returns
    -------
    systype : string
        Either 'linux', 'win', or 'mac'
    """
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform == 'win32':
        return 'win'
    elif sys.platform == 'darwin':
        return 'mac'
    else:
        # 'cygwin', 'os2', 'os2emx', 'riscos', 'atheos'
        raise ValueError("Unknown operating system: %s" % sys.platform)

def spc_find_temperature(tree, nclusters, minclus):
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

def spc_find_temperature_2(tree, nclusters):
    """
    Find temperature without using a minimum cluster size
    """
    ctree = tree[:,4:4+nclusters] # tree of [:,nclusters]
    dtree = np.diff(ctree, axis = 0)
    votingtemps = dtree[:,1:].argmax(0)
    votes = np.sum(dtree[votingtemps,1:] > 0, 1) #TODO maybe >=?
    temp = votingtemps[votes.argmax()] + 1
    logging.debug("Cluster temperature: %i" % temp)
    return temp
    
    goodtemps = np.where(dtree[:,1:] > ctree[0,0] * 0.1)[0] # the % here is dependent on the SNR of the signal
    if len(goodtemps) == 0:
        temp = 1
    else:
        temp = goodtemps.max() + 1
    logging.debug("Cluster temperature: %i" % temp)
    return temp

def spc(features, tmp = '/tmp', mintemp = 0, maxtemp = 0.201, tempstep = 0.01,
        swcycles = 100, knn = 11, minclus = None, minperc = 0.5, nclusters = 5,
        quiet = True):
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
    minclus : int
        Minimum cluster size, for determining optimal temperature
        Should be ~session length in seconds
    minperc : float
        Percentage of total spikes used to determine minimum cluster size.
        Only used when minclus is None
    nclusters : int
        Number of clusters to return
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
    # mintemp = kwargs.get('mintemp', 0) 
    # maxtemp = kwargs.get('maxtemp', 0.201)
    # tempstep = kwargs.get('tempstep', 0.01)
    outfile = "/".join((tempdir, "clusters"))
    # swcycles = kwargs.get('swcycles', 100)
    # knn = kwargs.get('knn', 11)
    if minclus is None:
        minclus = int(len(features) * minperc)
    # minclus = kwargs.get('minclus', int(len(features) * 0.25))
    
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
    exefile = os.path.dirname(os.path.abspath(__file__)) + '/bin/spc_' + get_os()
    logging.debug("Found spc executable: %s" % exefile)
    shutil.copy2(exefile, tempdir)
    exefile = 'spc_' + get_os()
    
    # run executable
    olddir = os.getcwd()
    os.chdir(tempdir)
    # make file executable
    os.chmod(exefile, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    if quiet:
        with open(os.devnull, 'w') as fp:
            retcode = subprocess.call(['./spc_' + get_os(), runfilename], stdout=fp)
    else:
        retcode = subprocess.call(['./spc_' + get_os(), runfilename])
    os.chdir(olddir)
    logging.debug("Clustering returned: %i" % retcode)
    
    # read output
    cdata = np.loadtxt(outfile+'.dg_01.lab')
    tree = np.loadtxt(outfile+'.dg_01')
    
    # clean up
    shutil.rmtree(tempdir)
    
    # find good 'temperature' for clustering
    # temp = spc_find_temperature(tree, nclusters, minclus)
    temp = spc_find_temperature_2(tree, nclusters)
    # # find good 'temperature' for clustering
    # # dt = np.diff(tree,axis=0)[:,4:4+nclusters] # only consider n clusters, this is based on WaveClus
    # logging.debug("Cluster temperature threshold: %i" % minclus)
    # # temp1 = np.where(np.any(ct1[:,4:4+nclusters+1] > thresh,1))[0][-1] # find max temp with 1 clus > thresh
    # goodtemps = np.where(np.any(tree[:,4:4+nclusters] > minclus,1))[0]
    # if len(goodtemps) == 0:
    #     temp = 1
    # else:
    #     temp = goodtemps[-1]
    # if temp == 0: temp = 1 # based on WaveClus, to overcome first temperature being all 1 spin
    # # temp = len(np.where(np.max(dt,1) > minclus)[0])
    # # if temp == 0 and tree[0,nclusters+1] < minclus:
    #     # temp += 1 # based on WaveClus... all seems arbitrary :-/
    # logging.debug("Cluster temperature: %i" % temp)
    
    clusters = []
    for i in xrange(nclusters):
        clusters.append(np.where(cdata[temp,2:] == i)[0])
    # unmatched = np.setdiff1d(range(len(features)), [c for cluster in clusters for c in cluster])
    unmatched = np.setdiff1d(range(len(features)), np.hstack(clusters))
    clusters = [unmatched,] + clusters
    
    return clusters, cdata, tree

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

def test_clusters_to_indices():
    ids = clusters_to_indices([[1,2,3],[4,5,0,6]])
    assert all(ids == np.array([1, 0, 0, 0, 1, 1, 1])), "clusters_to_indicies failed"
    
    try:
        clusters_to_indices([[1,2,3],[4,5,6]]) # missing index 0
        raise Exception("clusters_to_indices failed to raise IndexError on missing index")
    except IndexError:
        pass # this is what it's supposed to do
    except Exception as e:
        raise e

def klustakwik(features, tmp = '/tmp'):
    """
    NotImplemented
    """
    raise NotImplemented

def test_spc(plot=False):
    fakefeatures = np.array([\
    [9.121892737849592869e-01,1.702694147769417965e-01,1.976813384432012377e+00],
    [1.505255486133956655e+00,-2.132262451773800915e-01,1.139754580088775171e+00],
    [1.796021056608481992e+00,5.482791889945581865e-02,1.199430447971401303e+00],
    [1.766677651808613581e+00,-2.355300788060969985e-01,1.486193183085017466e+00],
    [1.600278741356271794e+00,-1.775991502788796250e-01,1.527281967784109185e+00],
    [9.670918210696861639e-01,2.486423132958039961e-01,1.961735764050466013e+00],
    [1.734223899292208548e+00,-2.512813368026616256e-01,1.296319560299583173e+00],
    [1.094575833590590630e+00,1.267858197785596275e-01,1.779466395716537086e+00],
    [1.059276514121765711e+00,1.305797159406407726e-01,2.009464406413342452e+00],
    [1.674525742681307161e+00,-2.156485144335169579e-01,1.303970246832113933e+00],
    [9.501799423598481509e-01,2.315823895577806546e-01,1.768562216091043693e+00],
    [1.041072987670736083e+00,-4.213460119602729925e-02,2.121463958816212081e+00],
    [1.836259659903269004e+00,-2.598869543998592047e-01,1.459478778381221531e+00],
    [1.714749555013693305e+00,-7.457626088312108159e-02,1.289433351235938607e+00],
    [9.027047382404921327e-01,5.136644878710594497e-02,1.936821952740319208e+00],
    [1.074105537844705749e+00,6.360771785660634947e-02,1.838655582554846735e+00],
    [1.643212005392947983e+00,-2.745992386999307477e-01,1.297965407475970112e+00],
    [9.489769736806544786e-01,2.426835164045688664e-01,1.894859490065952112e+00],
    [1.734098755661226399e+00,-3.926875607205684293e-01,1.317626529276785918e+00],
    [9.882055268780466362e-01,1.558599301169458773e-01,1.934077210710049455e+00],
    [1.808304977297616301e+00,-2.198704001991904633e-01,1.501091799349095757e+00],
    [1.764705842536313352e+00,-3.250740305534627339e-01,1.516096375539988550e+00],
    [1.646630712538638353e+00,-2.826626520191597436e-01,1.249581548493040595e+00],
    [1.683388848533310611e+00,-2.002866528469764429e-01,1.346262890851235516e+00],
    [1.756721094225609958e+00,-2.730115827316190735e-01,1.112886885501471523e+00],
    [1.669121401396321591e+00,-2.186152753041630259e-01,1.435177756094649260e+00],
    [1.056231117027902400e+00,1.871922006208025513e-01,1.890754987792580444e+00],
    [1.629917946089443292e+00,-1.795630738890776623e-01,1.218610980156035417e+00],
    [1.603581822150865976e+00,-2.819810598942671387e-01,1.460395310442928807e+00],
    [1.661431343339078603e+00,-1.971581615096433815e-01,1.393247354066568855e+00]])
    
    
    logging.basicConfig(level=logging.DEBUG)
    
    features = fakefeatures
    # features = realfeatures
    
    clusters, cdata, tree = spc(features)#, minclus = 15)
    logging.info("Clusters: %s" % str([len(c) for c in clusters]))
    # print len(unmatched)
    
    if plot:
        import pylab as pl
        from mpl_toolkits.mplot3d import Axes3D
        
        ax = Axes3D(pl.figure())
        colors = ['k','b','g','r','m','y']
        for (i,c) in enumerate(clusters):
            d = features[c]
            if len(d) == 0: continue
            ax.scatter(d[:,0],d[:,1],d[:,2],c=colors[i])
        
        # d = features[unmatched]
        # ax.scatter(d[:,0],d[:,1],d[:,2],c='k')
        
        pl.figure()
        
        nf = features.shape[1]
        for x in xrange(nf):
            for y in xrange(nf):
                if x > y:
                    pl.subplot(nf,nf,x+y*nf+1)
                    for (i, c) in enumerate(clusters):
                        d = features[c]
                        if len(d) == 0: continue
                        pl.scatter(d[:,x], d[:,y], c=colors[i], s=10, alpha=0.5, edgecolors=None)
                        # pl.scatter(features[:,x], features[:,y], s=1)
                    # pl.gca().set_axis_off()
                    pl.gca().set_xticks([])
                    pl.gca().set_yticks([])
                    b = 1/8.
                    xr = features[:,x].max() - features[:,x].min()
                    pl.xlim(features[:,x].min()-xr*b, features[:,x].max()+xr*b)
                    yr = features[:,y].max() - features[:,y].min()
                    pl.ylim(features[:,y].min()-yr*b, features[:,y].max()+yr*b)
        
        pl.figure()
        pl.imshow(tree, interpolation='nearest')
        pl.figure()
        pl.imshow(cdata, interpolation='nearest')
        pl.show()
    return

if __name__ == '__main__':
    test_spc(True)