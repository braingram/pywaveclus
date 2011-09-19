#!/usr/bin/env python

import numpy as np

import pywaveclus

def test_butter():
    # generate fake data
    freq = 0.5
    t = np.linspace(0, 1, 44100, endpoint = False)
    r = np.sin(t * 2. * np.pi * freq)
    
    f = pywaveclus.dsp.butter.filt(r, 300, 3000, 44100, 3)
    
    assert len(f) == len(r)
    assert np.max(f) - np.min(f) < 0.1, np.max(f) - np.min(f)
    
    freq = 100
    r = np.sin(t * 2. * np.pi * freq)
    f = pywaveclus.dsp.butter.filt(r, 300, 3000, 44100, 3)
    assert np.max(f) - np.min(f) < 0.1, np.max(f) - np.min(f)
    
    freq = 1000
    r = np.sin(t * 2. * np.pi * freq)
    f = pywaveclus.dsp.butter.filt(r, 300, 3000, 44100, 3)
    assert np.max(f) - np.min(f) > 0.9, np.max(f) - np.min(f)
    
    freq = 10000
    r = np.sin(t * 2. * np.pi * freq)
    f = pywaveclus.dsp.butter.filt(r, 300, 3000, 44100, 3)
    assert np.max(f[100:-100]) - np.min(f[100:-100]) < 0.1, np.max(f[100:-100]) - np.min(f[100:-100])
    