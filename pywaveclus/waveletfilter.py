#!/usr/bin/env python

# Copyright (c) 2009, Alex Wiltschko
# With modifications by Christoph Weidemann (03/2009)
# With modifications by Brett Graham (08/2011)
# This code is released under the terms of the BSD License

import numpy as np
import pywt

def waveletfilter(data, maxlevel = 6, wavelet = 'db20', mode = 'sym', minlevel = 1):
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
        maxlevel = len(coeffs)
        # Highpass
        for lvl in np.arange(1,minlevel):
            # print 'zeroing level: %i at index %i of %i coeffs' % (lvl, maxlevel-lvl, len(coeffs))
            coeffs[maxlevel-lvl][:] = 0
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

def test_waveletfilter():
    import pylab as pl
    
    # make test signal
    Fs = 44100
    freqs = [100,1000,2000,4000,5000,6000,7000,8000,9000,10000]
    t = np.arange(Fs,dtype=np.float64) / float(Fs) # 1 second
    x = np.zeros(len(t))
    for f in freqs:
        x += np.sin(t * f * 2. * np.pi)
    x /= len(freqs)
    # print t,x
    
    filtered = waveletfilter(x, minlevel=3, maxlevel=6)
    
    pl.subplot(221)
    pl.plot(t,x)
    pl.subplot(222)
    pl.psd(x,Fs=Fs)
    pl.subplot(223)
    pl.plot(t,filtered)
    pl.subplot(224)
    pl.psd(filtered,Fs=Fs)
    
    pl.show()

if __name__ == '__main__':
    test_waveletfilter()