#!/usr/bin/env python

import numpy as np

import scipy.signal

def interpolate_peak(data, pre = 40, post = 88, oversample = 10., find_extreme = lambda x: x.argmax()):
    """
    Use cubic bspline interpolation to estimate the actual peak given a sequence of samples
    
    """
    # data len should be (pre + post) * 2
    assert len(data) == (pre+post)*2, "Data length(%i) must be (pre+post)*2[%i]" % (len(data), (pre+post)*2)
    
    # find current peak
    peakI = pre*2#find_extreme(data)
    # pull out wave
    wave = data[peakI-pre:peakI+post]
    # fit bsplines
    coeffs = scipy.signal.cspline1d(wave)
    # generate oversampled wave
    oversampled = scipy.signal.cspline1d_eval(coeffs, np.linspace(0,pre+post-1,(pre+post)*oversample))
    # find peak of oversampled wave
    oversampledT = np.linspace(0,len(wave)-1,len(oversampled))
    opeakI = oversampledT[find_extreme(oversampled)]
    # downsample oversampled wave to original sampling rate
    retTs = np.linspace(opeakI-pre, opeakI+post-1, pre+post)
    # retTs = oversampledT[opeakI-pre:opeakI+post]
    ret = scipy.signal.cspline1d_eval(coeffs, retTs)
    return ret


def cubic(data, pre = 40, post = 88, oversample = 10., find_extreme = lambda x: x.argmax(), ts = None):
    """
    Use cubic bspline interpolation to estimate the actual peak given a sequence of samples
    if ts are defined, resample the cubic spline at these new points
    """
    # data len should be (pre + post) * 2
    assert len(data) == (pre+post)*2, "Data length(%i) must be (pre+post)*2[%i]" % (len(data), (pre+post)*2)
    
    # find current peak
    peakI = pre*2#find_extreme(data)
    # pull out wave
    wave = data[peakI-pre:peakI+post]
    # fit bsplines
    coeffs = scipy.signal.cspline1d(wave)
    # generate oversampled wave
    oversampled = scipy.signal.cspline1d_eval(coeffs, np.linspace(0,pre+post-1,(pre+post)*oversample))
    # find peak of oversampled wave
    oversampledT = np.linspace(0,len(wave)-1,len(oversampled))
    opeakI = oversampledT[find_extreme(oversampled)]
    # downsample oversampled wave to original sampling rate
    if ts is None:
        ts = np.linspace(opeakI-pre, opeakI+post-1, pre+post)
    # retTs = oversampledT[opeakI-pre:opeakI+post]
    ret = scipy.signal.cspline1d_eval(coeffs, ts)
    return ts, ret
