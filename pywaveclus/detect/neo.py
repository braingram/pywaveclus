#!/usr/bin/env python

import numpy as np

def neo(data, N = 529):#265):
    # 
    # From: Neural Spike Sorting Under Nearly 0-dB Signal-to-Noise Ratio
    #       Using Nonlinear Energy Operator and Artificial Neural-Network Classifier
    #
    #   neo = (dx/dt)**2 - x(t)(d**2x/dt**2)
    #
    # for discrete-time sequence data
    #   neo = x(n)**2 - x(n+1)x(n-1)
    #
    # all of this followed by bartlet window convolution
    #   window length = 6-12 samples@10k [0.6ms -> 1.2ms] 265-529 samples@44.1k
    neo = np.empty_like(data)
    neo[0] = 0.
    neo[-1] = 0.
    neo[1:-1] = data[1:-1]**2 - data[2:]*data[:-2]
    
    # bartlett filter
    b = np.bartlett(N)
    return np.convolve(neo, b, mode='same')

def calculate_threshold(data, n = 5):
    assert data.ndim == 1
    nd = neo(data)
    return np.mean(nd) + np.std(nd) * n

def find_spikes(data, threshold, prew = 40, postw = 88, ref = 66):
    nd = neo(data)
    crossings = find_threshold_crossings(nd, threshold, 'pos')

    find_extreme = lambda x: np.abs(x).argmax()
    # find_extreme = lambda x: x.argmax()

    spikeindices = []
    spikewaveforms = []

    ndata = len(data)
    last = 0
    for i in xrange(len(crossings)):
        if crossings[i] >= last + ref:
            extremeindex = find_extreme(data[crossings[i]:crossings[i]+postw+1])
            spikeindex = extremeindex + crossings[i]
            if spikeindex+postw > ndata:
                logging.debug("Spike ran off end of data at frame %i" % spikeindex)
                continue
            spikewaveforms.append(data[spikeindex-prew:spikeindex+postw])
            spikeindices.append(spikeindex)
            last = spikeindex

    return spikeindices, spikewaveforms