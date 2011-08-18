#!/usr/bin/env python

import logging

import numpy as np

def match(traces, clusters, method = 'center', **kwargs):
    """
    Parameters
    ----------
    traces : 2d array
        Can be raw spikes or features, used to create the templates
    clusters: list of 1d array of int
        Clusters, use 1: to make templates, fit 0 (unmatched) to templates
    method : string
        Method to find closest fit for unmatched traces
        One of: nn center ml mahal
    kwargs
        Arguments that will be passed on to the matching method
    
    Returns
    -------
    clusters : list of 1d array of int
        Clusters with rematched cluster 0
    """
    if method == 'nn':
        matches =  nn(traces, clusters, **kwargs)
    elif method == 'center':
        matches = center(traces, clusters, **kwargs)
    elif method == 'ml':
        matches = ml(traces, clusters)
    elif method == 'mahal':
        matches = mahal(traces, clusters)
    else:
        raise ValueError("Unknown template match method %s" % method)
    
    # construct new cluster from matches
    unmatchedindices = list(clusters[0])[:]
    clusters[0] = []
    clusters = [list(c) for c in clusters] # convert to lists for easy appending
    for i, m in zip(unmatchedindices, matches):
        clusters[m].append(i)
    clusters = [np.array(c,dtype=int) for c in clusters] # convert back to 2d array
    return clusters

def test_match():
    c1 = np.random.randn(10,3) * .1 + 1.0
    c2 = np.random.randn(10,3) * .05 + 2.0
    traces = np.vstack((c1,c2))
    clusters = [[19,], range(10), range(10,19)]
    
    clusters = match(traces, clusters, method='nn', kmin = 5)
    r = [[], np.arange(10), np.arange(10,20)]
    c = clusters
    assert all([list(c[i]) == list(r[i]) for i in xrange(len(c))]), "nn:%s" % str(clusters)
    
    clusters = match(traces, clusters, method='center')
    r = [[], np.arange(10), np.arange(10,20)]
    c = clusters
    assert all([list(c[i]) == list(r[i]) for i in xrange(len(c))]), "center:%s" % str(clusters)

def nearestneighbor(x, vectors, maxdist, k = 1):
    """
    Parameters
    ----------
    x : 1d array
        Point for which to find neighbors
    vectors : 2d array
        Locations of other points
    maxdist : float or 1d array
        Maximum euclidean distance of neighbors
    k : int
        Return up to k nearest neighbors
    
    Returns
    -------
    nearest : 1d array of int
        Up to k nearest neighbors
        May return empty array, or <k array
    
    Notes
    -----
    """
    assert maxdist >= 0., "maxdist [%s] must be >= 0" % maxdist
    # vectors = np.atleast_2d(vectors)
    assert vectors.shape[1] == len(x), "vectors.shape[1] = %i must == len(x) [%i]" % (vectors.shape[1], len(x))
    assert k > 0, "k[%i] must be > 0" % k
    dists = np.sum((vectors - x )**2,1)**0.5
    conforming = np.where(dists < maxdist)[0]
    # in WaveClus: pointdist and pointlimit are always inf, and have no effect, so skipping...
    if len(conforming) == 0:
        return np.array([],dtype=int)
    return conforming[np.argsort(dists[conforming])][:k]

def test_nearestneighbor():
    vs = np.mgrid[:5,:5][0] / 4. # 5x5, row values 0. -> 1.
    p = np.ones(5) * 0.5
    indices = nearestneighbor(p, vs, 0.1)
    assert len(indices) == 1, "len(indices)[%i] should == 1" % len(indices)
    assert indices[0] == 2, "indices[0] = %i should == 2" % indices[0]
    
    indices = nearestneighbor(p, vs, 0.6)
    assert len(indices) == 1, "len(indices)[%i] should == 1" % len(indices)
    assert indices[0] == 2, "indices[0] = %i should == 2" % indices[0]
    
    indices = nearestneighbor(p, vs, 0.6, 3)
    assert len(indices) == 3, "len(indices)[%i] should == 3" % len(indices)
    assert indices[0] == 2, "indices[0] = %i should == 2" % indices[0]
    assert indices[1] in [1,3], "indices[1] = %i should == 1 or 3" % indices[1]
    assert indices[2] in [1,3], "indices[2] = %i should == 1 or 3" % indices[2]
    
    indices = nearestneighbor(p, vs, 0.6, 5)
    assert len(indices) == 3, "len(indices)[%i] should == 3" % len(indices)
    assert indices[0] == 2, "indices[0] = %i should == 2" % indices[0]
    assert indices[1] in [1,3], "indices[1] = %i should == 1 or 3" % indices[1]
    assert indices[2] in [1,3], "indices[2] = %i should == 1 or 3" % indices[2]

def nn(traces, clusters, nsd = 3, k = 10, kmin = 10):
    """
    Parameters
    ----------
    traces : 2d array
        see match
    clusters : list of 1d array of int
        see match
    nsd : float
        Number of std-dev used to calculate max neighbor distance
    k : int
        Number of nearest neighbors to consider
    kmin : int
        Minimum neighbors required for match
    
    Returns
    -------
    clusters : list of 1d array of int
        see match
    """
    sd = np.sqrt(np.sum(np.var(traces,1,ddof=1)))*nsd
    goodspikes = np.hstack(clusters[1:])
    votes = np.hstack([np.ones(len(c),dtype=int)*(i+1) for (i,c) in enumerate(clusters[1:])])
    matches = []
    for i in clusters[0]:
        # print traces[i], goodspikes, traces[goodspikes], sd, k
        nn = nearestneighbor(traces[i],traces[goodspikes],sd,k)
        # print nn, traces[i], traces[goodspikes], sd, k
        if len(nn) == 0:
            matches.append(0)
            continue
        # vote
        cvotes = [sum(votes[nn] == i) for i in xrange(1,len(clusters))] # how many points, from what clusters
        # print cvotes, kmin
        if max(cvotes) < kmin:
            matches.append(0)
            continue
        matches.append(cvotes.index(max(cvotes))+1) # cluster w/most neighbors
    return matches

def test_nn():
    c1 = np.random.randn(10,3) * .1 + 1.0
    c2 = np.random.randn(10,3) * .05 + 2.0
    traces = np.vstack((c1,c2))
    clusters = [[19,], range(10), range(10,19)]
    matches = nn(traces, clusters, kmin = 5)
    assert len(matches) == 1, "len(matches)[%i] should be == 1" % len(matches)
    assert matches[0] == 2, "matches[%i] should be == 2" % matches[0]

def center(traces, clusters, nsd = 3):
    """
    Parameters
    ----------
    traces : 2d array
        see match
    clusters : list of 1d array of int
        see match
    nsd : float
        Number of std-dev used to calculate max neighbor distance
    
    Returns
    -------
    clusters : list of 1d array of int
        see match
    """
    sds = []
    centers = []
    for i in xrange(1,len(clusters)):
        ctraces = traces[clusters[i]]
        centers.append(np.mean(ctraces,0))
        sds.append(np.sqrt(np.sum(np.var(ctraces,1))) * nsd) # should be over N not N-1 as per WaveClus
    centers = np.array(centers)
    matches = []
    for i in clusters[0]:
        n = nearestneighbor(traces[i], centers, sds)
        if len(n) == 0:
            matches.append(0)
        else:
            matches.append(n[0]+1)
    return matches

def test_center():
    c1 = np.random.randn(10,3) * .1 + 1.0
    c2 = np.random.randn(10,3) * .05 + 2.0
    traces = np.vstack((c1,c2))
    clusters = [[19,], range(10), range(10,19)]
    matches = center(traces, clusters)
    assert len(matches) == 1, "len(matches)[%i] should be == 1" % len(matches)
    assert matches[0] == 2, "matches[%i] should be == 2" % matches[0]
# 
# def ml(traces, clusters):
#     """
#     Parameters
#     ----------
#     traces : 2d array
#         see match
#     clusters : list of 1d array of int
#         see match
#     
#     Returns
#     -------
#     clusters : list of 1d array of int
#         see match
#     """
#     function index = ML_gaussian(x,mu,sigma)
#     % function index = ML_gaussian(x,mu,sigma)
#     % x is a vector drawn from some multivariate gaussian
#     % mu(i,:) is the mean of the ith Gaussian
#     % sigma(:,:,i) is the covariance of the ith Gaussian
#     % 
#     % Returns the index of the Gaussian with the highest value of p(x).
# 
#     N = size(mu,1);  % number of Gaussians
# 
#     if( N == 0 )
#         index = 0;
#     else
#         for i=1:N,
#             % leave out factor of 1/(2*pi)^(N/2) since it doesn't affect argmax
#             p(i) = 1/sqrt(det(sigma(:,:,i)))*exp(-0.5*(x-mu(i,:))*inv(sigma(:,:,i))*(x-mu(i,:))');
#         end
#         [m index] = max(p);
#     end
#     pass
# 
# def mahal(traces, clusters):
#     """
#     Parameters
#     ----------
#     traces : 2d array
#         see match
#     clusters : list of 1d array of int
#         see match
#     
#     Returns
#     -------
#     clusters : list of 1d array of int
#         see match
#     """
#     function index = nearest_mahal(x,mu,sigma)
#     % function index = nearest_mahal(x,mu,sigma)
#     % x is a vector
#     % mu(i,:) is the mean of the ith Gaussian
#     % sigma(:,:,i) is the covariance of the ith Gaussian
#     % 
#     % Returns the index of the Gaussian closest (by the Mahalanobis distance)
#     % to x.
# 
#     N = size(mu,1);  % number of Gaussians
#     d = [];
#     if( N == 0 )
#         index = 0;
#     else
#         for i=1:N,
#             d(i) = (x-mu(i,:))*inv(sigma(:,:,i))*(x-mu(i,:))';
#         end
#         [m index] = min(d);
#     end
#     pass

if __name__ == '__main__':
    test_nearestneighbor()
    test_nn()
    test_match()
    test_center()
    # test_ml()
    # test_mahal()