#!/usr/bin/env python

import logging

import numpy as np

from .. import dsp

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

def find_spikes(data, threshold, direction = 'neg', prew = 40, postw = 88, ref = 44, minwidth = 2,  slop = 0, oversample = 10):
    crossings = dsp.threshold.find_crossings(data, threshold, direction)
    if not len(crossings):
        logging.debug("No threshold crossings found over %f to %f at T:%f" % (data.min(), data.max(), threshold))
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
    
    # # combine spikes with small offsets (hysteresis)
    #     dcrossings = np.diff(crossings)
    #     gcrossings = np.where(dcrossings > slop + 1)[0] # this may or may not miss 0
    #     if gcrossings[0] != 0: gcrossings = np.hstack(([0],gcrossings))
    #     hcrossings = crossings[gcrossings]
    #     
    #     starts = hcrossings[::2]
    #     ends = hcrossings[1::2]
    #     sws = ends - starts
    #     refs = starts[1:] - ends[:-1]
    #     
    #     gss = np.where(sws >= minwidth)[0]
    #     grs = np.hstack(([0],np.where(refs >= ref)[0] + 1))
    #     gis = np.union1d(gss,grs)
    #     # find only > prew*s and < postw*2
    #     
    #     for (s,e) in zip(starts[gis],ends[gis]):
    #         peaki = find_extreme(data[s:e])# + start
    #         fullwave = data[peaki-(prew*2):peaki+(postw*2)]
    #         if len(fullwave) == ((prew+postw)*2):
    #             # assert fullwave[prew*2] == data[peaki]
    #             spikeindices.append(peaki)
    #             spikewaveforms.append(interpolate_peak(fullwave, prew, postw, oversample, find_extreme))
    #     
    #     return spikeindices, spikewaveforms
    #     
    #     # for i in xrange(len(crossings)-1):
    #     for c in crossings[:-1]:
    #         if ((c - end) >= ref) and ((c-start) >= minwidth):
    #             end = c
    #             peaki = find_extreme(data[start:end]) + start
    #             fullwave = data[peaki-(prew*2):peaki+(postw*2)]
    #             if len(fullwave) == ((prew+postw)*2):
    #                 # assert fullwave[prew*2] == data[peaki]
    #                 spikeindices.append(peaki)
    #                 spikewaveforms.append(interpolate_peak(fullwave, prew, postw, oversample, find_extreme))
    #         start = c2
    
    for (i,c) in enumerate(crossings[:-1]):
        if (crossings[i+1] - c) > slop + 1:
            # next crossings is NOT part of this spike
            if ((c - end) >= ref) and ((c - start) >= minwidth):
                peaki = find_extreme(data[start:c]) + start
                fullwave = data[peaki-(prew*2):peaki+(postw*2)]
                if len(fullwave) == ((prew+postw)*2):
                    spikeindices.append(peaki)
                    spikewaveforms.append(dsp.interpolate.interpolate_peak(fullwave, prew, postw, oversample, find_extreme))
                    end = c
            start = crossings[i+1]
    
    # for i in xrange(len(crossings)-1):
    #     c = crossings[i]
    #     c2 = crossings[i+1]
    #     if (c2 - c) > slop + 1:
    #         # next crossings is NOT part of this spike
    #         if (c - end) >= ref:
    #             # print "Width:", (c-start)
    #             if (c - start) >= minwidth:
    #                 end = c
    #                 # get spike
    #                 # spikebounds.append([start,end])
    #                 
    #                 peaki = find_extreme(data[start:end]) + start
    #                 fullwave = data[peaki-(prew*2):peaki+(postw*2)]
    #                 if len(fullwave) == ((prew+postw)*2):
    #                     # assert fullwave[prew*2] == data[peaki]
    #                     spikeindices.append(peaki)
    #                     spikewaveforms.append(interpolate_peak(fullwave, prew, postw, oversample, find_extreme))
    #                 else:
    #                     logging.debug("Spike ran off end of data at frame %i[%i]" % (peaki, len(fullwave)))
    #                 start = c2
    #             else:
    #                 logging.debug("Skipping thin spike[%i] at %i" % ((c-start), c))
    #                 start = c2
    #         else:
    #             # print "spike found in refractory time, skipping:", c
    #             logging.debug("Skipping spike in refractory time at %i" % c)
    #             start = c2
    
    if (start > end):
        end = crossings[-1]
        if (end - start) > minwidth:
            # get spike
            # spikebounds.append([start,crossings[-1]])
            peaki = find_extreme(data[start:end]) + start
            fullwave = data[peaki-(prew*2):peaki+(postw*2)]
            if len(fullwave) == ((prew+postw)*2):
                spikeindices.append(peaki)
                spikewaveforms.append(dsp.interpolate.interpolate_peak(fullwave, prew, postw, oversample, find_extreme))
            else:
                logging.debug("Spike ran off end of data at frame %i[%i]" % (peaki, len(fullwave)))
    
    return spikeindices, spikewaveforms

def _old_find_spikes(data, threshold, direction = 'neg', prew = 40, postw = 88, ref = 66):
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

if __name__ == '__main__':
    test_find_spikes(True)