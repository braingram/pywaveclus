#!/usr/bin/env python

import logging

import numpy as np

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
    assert(data.ndim == 1)
    return np.median(np.abs(data))/0.6745 * n

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
    assert(direction in ['pos','neg','both'])
    assert(threshold > 0)
    
    if direction == 'pos':
        return np.where(data > threshold)[0]
    if direction == 'neg':
        return np.where(data < -threshold)[0]
    if direction == 'both':
        return np.union1d(np.where(data > threshold)[0], np.where(data < -threshold)[0])

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
        Len of each array is prew + postw + 1 with peak/trough at array[prew]
    """
    assert(prew > 0)
    assert(postw > 0)
    assert(type(prew) is int)
    assert(type(postw) is int)
    
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
    
    assert(sum(np.array(bothspikes[0]) - np.array(invbothspikes[0])) == 0)
    assert(sum(np.array(positivespikes[0]) - np.array(negativespikes[0])) == 0)
    
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