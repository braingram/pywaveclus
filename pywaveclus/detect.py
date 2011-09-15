#!/usr/bin/env python

import logging

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

def neo_calculate_threshold(data, n = 5):
    assert data.ndim == 1
    nd = neo(data)
    return np.mean(nd) + np.std(nd) * n

def calculate_threshold(data, n = 5):
    """
    Calculate the threshold for spike detection given a subset of the data
    
    Parameters
    ----------
    data : 1d array
        Data (usually a subset) used to calculate the threshold
    n : int,optional
        Number of estimated standard deviations of the background noise to
        use as the threshold

    Returns
    -------
    threshold : float
        Spike threshold

    Notes
    -----
    threshold = n * median( |data| ) / 0.6745
    
    Quian Quiroga R, Nadasdy Z, Ben-Shaul Y (2004)
        Unsupervised Spike Detection and Sorting with
        Wavelets and Superparamagnetic Clustering.
        Neural Comp 16:1661-1687
    """
    assert data.ndim == 1, "Calculate threshold requires a 1d array"
    return n * np.median(np.abs(data)) / 0.6745

def find_threshold_crossings(data, threshold, direction = 'neg'):
    """
    Find the sub/super-threshold indices of data
    
    Parameters
    ----------
    data : 1d array
        Datapoints
    
    threshold: float
        Threshold (possibly calculated using calculate_threshold)
        Must be positive
    
    direction: string
        Must be one of the following:
            'pos' : find points more positive than threshold
            'neg' : find points more negative than -threshold
            'both': find points > threshold or < -threshold
    
    Returns
    -------
    crossings : 1d array
        Indices in data that cross the threshold
    """
    assert direction in ['pos','neg','both'], "Unknown direction[%s]" % direction
    assert threshold > 0, "Threshold[%d] must be > 0" % threshold
    
    if direction == 'pos':
        return np.where(data > threshold)[0]
    if direction == 'neg':
        return np.where(data < -threshold)[0]
    if direction == 'both':
        return np.union1d(np.where(data > threshold)[0], np.where(data < -threshold)[0])

def find_spikes2(data, threshold, direction = 'neg', prew = 40, postw = 88, ref = 44, minwidth = 2,  slop = 0, oversample = 10):
    crossings = find_threshold_crossings(data, threshold, direction)
    if not len(crossings):
        logging.debug("No spikes found")
        return [], []
    
    if direction == 'pos':
        find_extreme = lambda x: x.argmax()
    elif direction == 'neg':
        find_extreme = lambda x: x.argmin()
    elif direction == 'both':
        find_extreme = lambda x: np.abs(x).argmax()
    else:
        raise ValueError, "direction[%s] must be neg, pos, or both" % direction
    
    spikeindices = []
    spikewaveforms = []
    
    #TODO artifact detection?
    ndata = len(data)
    start = crossings[0]
    end = -ref
    for i in xrange(len(crossings)-1):
        c = crossings[i]
        c2 = crossings[i+1]
        if (c2 - c) > slop + 1:
            # next crossings is NOT part of this spike
            if (c - end) >= ref:
                # print "Width:", (c-start)
                if (c - start) >= minwidth:
                    end = c
                    # get spike
                    # spikebounds.append([start,end])
                    
                    peaki = find_extreme(data[start:end]) + start
                    fullwave = data[peaki-(prew*2):peaki+(postw*2)]
                    if len(fullwave) == ((prew+postw)*2):
                        # assert fullwave[prew*2] == data[peaki]
                        spikeindices.append(peaki)
                        spikewaveforms.append(interpolate_peak(fullwave, prew, postw, oversample, find_extreme))
                    else:
                        logging.debug("Spike ran off end of data at frame %i[%i]" % (peaki, len(fullwave)))
                    start = c2
                else:
                    logging.debug("Skipping thin spike[%i] at %i" % ((c-start), c))
                    start = c2
            else:
                # print "spike found in refractory time, skipping:", c
                logging.debug("Skipping spike in refractory time at %i" % c)
                start = c2
    
    if (start > end):
        end = crossings[-1]
        if (end - start) > minwidth:
            # get spike
            # spikebounds.append([start,crossings[-1]])
            peaki = find_extreme(data[start:end]) + start
            fullwave = data[peaki-(prew*2):peaki+(postw*2)]
            if len(fullwave) == ((prew+postw)*2):
                spikeindices.append(peaki)
                spikewaveforms.append(interpolate_peak(fullwave, prew, postw, oversample, find_extreme))
            else:
                logging.debug("Spike ran off end of data at frame %i[%i]" % (peaki, len(fullwave)))
    
    return spikeindices, spikewaveforms

def find_spikes(data, threshold, direction = 'neg', prew = 40, postw = 88, ref = 66):
    """
    Find spikes within a single recording
    
    Parameters
    ----------
    data : 1d array
        Datapoints
    
    threshold: float
        see find_threshold_crossings
    
    direction: string
        see find_threshold_crossings
    
    prew: int
        Number of samples to save prior to the peak/trough as part of the spike waveform
    
    postw: int
        Number of samples to save after to the peak/trough as part of the spike waveform
    
    ref: int
        Refractory period of spike detection. Number of samples after a peak/trough is
        detected during which a second spike will not be detected.
    
    Returns
    -------
    spikeindices: list of ints
        Indices of spike peaks/troughs in data
    
    spikewaveforms: list of 1d arrays
        Data samples around a detected peak/trough.
        Len of each array is prew + postw with peak/trough at array[prew]
    """
    assert type(prew) is int, "prew[%s] must be an int" % str(prew)
    assert type(postw) is int, "postw[%s] must be an int" % str(postw) 
    assert prew > 0, "prew[%i] must be > 0" % prew
    assert postw > 0, "postw[%i] must be > 0" % postw
    
    crossings = find_threshold_crossings(data, threshold, direction)
    
    if direction == 'pos':
        find_extreme = lambda x: x.argmax()
    elif direction == 'neg':
        find_extreme = lambda x: x.argmin()
    elif direction == 'both':
        find_extreme = lambda x: np.abs(x).argmax()
    else:
        raise ValueError, "direction[%s] must be neg, pos, or both" % direction
    
    spikeindices = []
    spikewaveforms = []
    
    #TODO artifact detection?
    ndata = len(data)
    last = 0
    for i in xrange(len(crossings)):
        # if crossings[i] + postw + 1 > ndata:
        #     # this spike waveform runs off the end of the data
        #     break
        if crossings[i] >= last + ref:
            extremeindex = find_extreme(data[crossings[i]:crossings[i]+postw+1])
            spikeindex = extremeindex + crossings[i]
            # peak of spike is at index prew in waveform
            if spikeindex+postw > ndata:
                logging.debug("Spike ran off end of data at frame %i" % spikeindex)
                continue # spike ran off end of data
            spikewaveforms.append(data[spikeindex-prew:spikeindex+postw])
            spikeindices.append(spikeindex)
            last = spikeindex
    
    #TODO waveform interpolation?
    return spikeindices, spikewaveforms

def neo_find_spikes(data, threshold, prew = 40, postw = 88, ref = 66):
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

def test_find_spikes(plot=False):
    logging.basicConfig(level=logging.DEBUG)
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
    
    threshold = calculate_threshold(x)
    positivespikes = find_spikes(x, threshold, 'pos')
    negativespikes = find_spikes(-x, threshold, 'neg')
    bothspikes = find_spikes(x, threshold, 'both')
    invbothspikes = find_spikes(-x, threshold, 'both')
    
    assert sum(np.array(bothspikes[0]) - np.array(invbothspikes[0])) == 0,\
        "Found different # of spikes for both and both (on inverted data) detection"
    assert sum(np.array(positivespikes[0]) - np.array(negativespikes[0])) == 0,\
        "Found different # of spikes for pos and neg (on inverted data) detection"
    
    prew = 5
    postw = 5
    spiketimes, spikewaveforms = find_spikes(x, threshold, 'both', prew=5, postw=5)
    print np.array(spikewaveforms).shape
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

if __name__ == '__main__':
    test_find_spikes(True)