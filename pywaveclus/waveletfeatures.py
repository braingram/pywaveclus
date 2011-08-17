#!/usr/bin/env python

import logging

import numpy as np
import scipy.stats

import pywt

def test_ks(coeffs):
    """
    A thin wrapper around scipy.stats.kstest to set ddof to 1 distribution to norm
    
    Parameters
    ----------
    coeffs : 1d array
        A single wavelet coefficient measured across many spikes
    
    Returns
    -------
    d : float
        D statistic from Kolmogorov-Smirnov test
    """
    d, _ = scipy.stats.kstest(scipy.stats.zscore(coeffs,ddof=1),'norm')
    return d

def wavelet_features(waveforms, nfeatures = 10, levels = 4, wavelet = 'haar'):
    """
    Given an array of spike waveforms, determine the best wavlet coefficients for clustering
    by using the Kolmogorov-Smirnov test.
        Was wave_features in WaveClus
    
    Parameters
    ----------
    waveforms : 2d array
        Spike waveforms, where waveforms[0] is the waveform for the first spike
    nfeatures : int
        Number of resulting measured features. Was inputs in WaveClus
    levels : int
        Number of wavelet levels for wavedec. Was scales in WaveClus
    
    Returns
    -------
    features : 2d array
        Resulting spike features, where features[0] are the features for the first spike.
        shape = (len(waveforms), nfeatures)
    """
    assert(nfeatures > 0)
    assert(levels > 0)
    if type(waveforms) != np.ndarray:
        waveforms = np.array(waveforms)
    assert(waveforms.ndim == 2)
    nwaveforms = len(waveforms)
    
    # test for size of coefficient vector
    get_coeffs = lambda wf: np.array([i for sl in pywt.wavedec(wf, wavelet, level=levels) for i in sl][:len(wf)])
    tr = get_coeffs(waveforms[0])
    ncoeffs = len(tr)
    
    coeffs = np.zeros((nwaveforms, ncoeffs))
    coeffs[0][:] = tr # store calculated coefficients for waveform 1
    for i in xrange(1,len(waveforms)): # get coefficient for other waveforms
        coeffs[i][:] = get_coeffs(waveforms[i])
    
    # KS test for coefficient selection
    coefffitness = np.zeros(ncoeffs)
    for i in xrange(ncoeffs):
        thrdist = np.std(coeffs[:,i],ddof=1) * 3
        thrdistmin = np.mean(coeffs[:,i]) - thrdist
        thrdistmax = np.mean(coeffs[:,i]) + thrdist
        # test for how many points lie within 3 std dev of mean
        culledcoeffs = coeffs[(coeffs[:,i] > thrdistmin) & (coeffs[:,i] < thrdistmax),i]
        if len(culledcoeffs) > 10:
            coefffitness[i] = test_ks(culledcoeffs)
        # else 0 (see coefffitness definition)
    # print coefffitness
    # store the indices of the 'good' coefficients
    ind = np.argsort(coefffitness)
    goodcoeff = ind[::-1][:nfeatures]
    # print goodcoeff
    # print ind
    
    # this returns features
    return coeffs[:,goodcoeff]

def test_wavelet_features(plot=False):
    import pylab as pl
    
    swf1 = [0., 0.2, 0.4, 0.8, 1.6, 1.4, 0.8, 0.3, -0.6, -0.8, -0.5, -0.3, 0.]
    # swf2 = [0., 0.1, 0.5, 0.9, 1.6, 1.4, 0.4, 0.1, -0.3, -0.2, -0.1,   0., 0.]
    swf2 = [0., 0., 0.1, 0.3, 1.8, 2.1, 0.4, 0.1, -0.3, -0.2, -0.1,   0., 0.]
    assert(len(swf1) == len(swf2))
    noiselvl = 1./10.
    n1 = 10
    n2 = 20
    nfeatures = 3
    n = n1 + n2
    wfs = pl.randn(n,len(swf1)) * noiselvl
    for i in xrange(n):
        if pl.rand() * n < n1:
            wfs[i] += swf1
        else:
            wfs[i] += swf2
    
    features = wavelet_features(wfs, nfeatures=nfeatures)
    # print n, nfeatures, features.shape
    assert(features.shape == (n,nfeatures))
    
    # np.savetxt('features', features, delimiter=',', newline='],\n')
    
    if plot:
        from mpl_toolkits.mplot3d import Axes3D
        pl.figure()
        ax = pl.gcf().add_subplot(2, 2, 1, projection='3d')
        ax.scatter(features[:,0],features[:,1],features[:,2])
        pl.subplot(222)
        pl.plot(np.transpose(wfs))
        pl.subplot(223)
        pl.plot(swf1)
        pl.plot(swf2)
        pl.show()

if __name__ == '__main__':
    test_wavelet_features(True)