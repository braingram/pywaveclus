#!/usr/bin/env python

# Copyright (c) 2009, Alex Wiltschko
# With modifications by Christoph Weidemann (03/2009)
# With modifications by Brett Graham (08/2011)
# This code is released under the terms of the BSD License

import logging

import numpy as np
import pywt

from .. import utils

def filt(data, maxlevel = 6, wavelet = 'db20', mode = 'sym', minlevel = 1):
    """
    Filter a multi-channel signal using wavelet filtering.
        Named wavefilter in WaveClus

    Parameters
    ----------
    data : array
        Data array with a row for each channel and a column for each
        sample.
    maxlevel : int,optional
        Level of decomposition to perform on the data. This implicitly
        defines the cutoff frequency of the filter (cutoff frequency =
        samplingrate/(2^(maxlevel+1))
    minlevel : int,optional
        Lowpass the data (cutoff frequency = samplingrate/(2^(minlevel)))
    wavelet : {str, pywt.Wavelet},optional
        Wavelet to use. If string, name of the wavelet (see
        pywt.wavelist() for acceptable values). Alternatively a
        pywt.Wavelet object can be specified.
    mode : str,optional
        Signal extension mode. See the docstring for pywt.MODES for
        details.

    Returns
    -------
    filtered_data : array

    Notes
    -----
    See the following paper for more information:
    Wiltschko A. B., Gage G. J., & Berke, J. D. (2008). Wavelet filtering
        before spike detection preserves waveform shape and enhances
        single-unit discrimination. Journal of Neuroscience Methods, 173,
        34-40.
    doi:10.1016/j.jneumeth.2008.05.016
    http://www.ncbi.nlm.nih.gov/pubmed/18597853
    """

    data = np.atleast_2d(data)
    # print data.shape
    numchannels, datalength = data.shape

    # Initialize the container for the filtered data
    fdata = np.empty((numchannels, datalength))

    for i in range(numchannels):
        # Decompose the signal
        coeffs = pywt.wavedec(data[i,:], wavelet, mode=mode, level=maxlevel)
        # Destroy the approximation coefficients
        coeffs[0][:] = 0
        lencoeffs = len(coeffs)
        # Highpass
        for lvl in np.arange(1,minlevel):
            # print 'zeroing level: %i at index %i of %i coeffs' % (lvl, maxlevel-lvl, len(coeffs))
            coeffs[lencoeffs-lvl][:] = 0
        # Reconstruct the signal and save it. If len(data[i,:]) is odd,
        # the array returned by pywt.waverec will have one extra value
        # at the end, so we need to make sure to trim the returned
        # array to the length of data[i,:]:
        fdata[i,:] = pywt.waverec(coeffs, wavelet, mode=mode)[:len(data[i,:])]
    
    if fdata.shape[0] == 1:
        return fdata.ravel() # If the signal is 1D, return a 1D array
    else:
        return fdata # Otherwise, give back the 2D array

def calculate_cutoffs(samplingrate, maxlevel=None):
    """
    Calculate the cutoff frequences for each decomposition level
    
    Parameters
    ----------
    samplingrate : int
        Number of data samples per second
    maxlevel : int, optional
        Maximum wavlet decomposition level at which to calculate the cutoffs
            default = int(ceil(log2(samplingrate)-1))
    
    Returns
    -------
    cutoffs : 1d array
        Cutoff frequencies for wavlet decomposition level boundries
        len(cutoffs) = maxlevel + 1
    
    Notes
    -----
    cutoffs 0 & 1 are for level 1, 1 & 2 for level 2, etc...
    
    """
    if maxlevel is None:
        # sf / (2 ** (lvl+1)) = 1Hz
        # sf = 2 ** (lvl+1)
        # log2(sf) = lvl+1
        # log2(sf) - 1 = lvl
        maxlevel = int(np.ceil(np.log2(samplingrate) - 1))
    return samplingrate / (2 ** (np.arange(1,maxlevel+2)))

def level_to_cutoffs(samplingrate, level):
    """
    Calculate the cutoff frequencies for a single wavelet decomposition level
    
    Parameters
    ----------
    samplingrate : int
        Number of data samples per second
    level : int
        Wavelet decomposition level at which to calculate cutoff frequencies
    
    Returns
    -------
    lowcutoff : float
        Low frequency cutoff point
    highcutoff : float
        High frequency cutoff point
    """
    return (samplingrate / 2 ** (level+1), samplingrate / 2 ** level)

def features(waveforms, nfeatures = 10, levels = 4, wavelet = 'haar'):
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
    assert nfeatures > 0, "nfeatures[%i] must be > 0" % nfeatures
    assert levels > 0, "levels[%i] must be > 0" % levels
    if type(waveforms) != np.ndarray:
        waveforms = np.array(waveforms)
    assert waveforms.ndim == 2, "waveforms.ndim[%i] must be == 2" % waveforms.ndim
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
            coefffitness[i] = utils.ks(culledcoeffs)
        # else 0 (see coefffitness definition)
    # print coefffitness
    # store the indices of the 'good' coefficients
    ind = np.argsort(coefffitness)
    goodcoeff = ind[::-1][:nfeatures]
    # print goodcoeff
    # print ind
    
    # this returns features
    return coeffs[:,goodcoeff]