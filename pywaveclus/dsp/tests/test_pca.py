#!/usr/bin/env python

import numpy as np

import pywaveclus

def test_stack_waveforms():
    # make 2d waveforms
    x = 13
    y = 7
    z = 5

    w2d = np.arange(x*y).reshape((x,y))
    w3d = np.arange(x*y*z).reshape((x,y,z))

    f = pywaveclus.dsp.pca.stack_waveforms # i'm lazy
    
    # check 2d
    assert(w2d.shape == f(w2d).shape) # check shapes
    assert(np.all(f(w2d) == w2d)) # check values

    # check 3d
    t3d = np.arange(x*y*z).reshape((x,y*z))
    assert(t3d.shape == f(w3d).shape) # check shapes
    assert(np.all(t3d == f(w3d))) # check values
