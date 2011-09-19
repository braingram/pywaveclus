#!/usr/bin/env python

from nose.tools import eq_

import numpy as np

import pywaveclus

def test_calculate_threshold():
    d = np.array([0,0,0])
    T = pywaveclus.detect.threshold.calculate_threshold(d)
    eq_(T, 0.)
    
    d = np.array([1,1,1])
    T = pywaveclus.detect.threshold.calculate_threshold(d, 5.)
    eq_(T, 7.4128984432913274)

def test_find_spikes(plot=False):
    # generate fake data
    Fs = 44100
    t = np.arange(Fs, dtype=np.float64)/ float(Fs)
    swf = [0., 0.2, 0.4, 0.8, 1.6, 1.4, 0.8, 0.3, -0.6, -0.8, -0.5, -0.3, 0.]
    peakoffset = 4
    sts = [4410, 8820]
    x = np.random.randn(len(t)) / 10.
    for st in sts:
        x[st-peakoffset:st-peakoffset+len(swf)] += swf
    
    x[-4] = 2.
    
    threshold = pywaveclus.detect.threshold.calculate_threshold(x)
    positivespikes = pywaveclus.detect.threshold.find_spikes(x, threshold, 'pos')
    negativespikes = pywaveclus.detect.threshold.find_spikes(-x, threshold, 'neg')
    bothspikes = pywaveclus.detect.threshold.find_spikes(x, threshold, 'both')
    invbothspikes = pywaveclus.detect.threshold.find_spikes(-x, threshold, 'both')
    
    assert sum(np.array(bothspikes[0]) - np.array(invbothspikes[0])) == 0,\
        "Found different # of spikes for both and both (on inverted data) detection"
    assert sum(np.array(positivespikes[0]) - np.array(negativespikes[0])) == 0,\
        "Found different # of spikes for pos and neg (on inverted data) detection"
    
    prew = 5
    postw = 5
    spiketimes, spikewaveforms = pywaveclus.detect.threshold.find_spikes(x, threshold, 'both', prew=5, postw=5)
    print np.array(spikewaveforms).shape
    print spikewaveforms
    print spiketimes
    assert np.array(spikewaveforms).shape == (len(sts),prew+postw), \
        "Invalid waveform shape[%s], should be %s" % (np.array(spikewaveforms).shape, (len(sts),prew+postw))
    
    if plot:
        import pylab as pl
        pl.figure()
        pl.plot(t,x)
        pl.axhline(threshold,c='r')
        sis = positivespikes[0]
        pl.scatter(t[sis], x[sis], c='r')
        
        pl.figure()
        for swf in positivespikes[1]:
            pl.plot(swf)
        pl.show()