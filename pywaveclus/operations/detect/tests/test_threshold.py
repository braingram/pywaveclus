#!/usr/bin/env python

from nose.tools import eq_

import numpy as np

import pywaveclus


#def find_crossings(data, threshold, direction = 'neg'):

def test_threshold():
    d = np.array([0., 1., -1., 0.])
    
    c = pywaveclus.dsp.threshold.find_crossings(d, 0.5, 'neg')
    eq_(list(c),[2,])
    
    c = pywaveclus.dsp.threshold.find_crossings(d, 0.5, 'pos')
    eq_(list(c),[1,])
    
    c = pywaveclus.dsp.threshold.find_crossings(d, 0.5, 'both')
    eq_(list(c),[1,2])
    
    d = [0, 1, -1, 0] # test list
    c = pywaveclus.dsp.threshold.find_crossings(d, 0.5, 'both')
    eq_(list(c),[1,2])