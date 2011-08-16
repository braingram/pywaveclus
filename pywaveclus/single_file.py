#!/usr/bin/env python
"""
Load and cluster a single file
"""

import logging, sys
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pylab as pl
import scikits.audiolab as al
from scipy.signal import resample

import waveletfilter, detect, waveletfeatures

filename = '/Users/graham/Repositories/coxlab/physiology_analysis/data/clip.wav'
# filename = '/Users/graham/Repositories/coxlab/physiology_analysis/data/long.wav'
# filename = '/Users/graham/Repositories/coxlab/physiology_analysis/data/input_14#01.wav'
# filename = '/Users/graham/Repositories/coxlab/physiology_analysis/data/input_3#01.wav'

if len(sys.argv) > 1:
    filename = sys.argv[1]

baselineTime = 44100 # initial samples to use to calculate threshold
plotRawWave = False
framesPerChunk = 44100
chunkOverlap = 4100
detectionDirection = 'both'
prew = 31 # for spike waveform capture
nfeatures = 10
filterMin = 3
filterMax = 6

def chunk(n, chunksize, overlap=0):
    # n = len(data)
    for i in xrange((n/chunksize)+1):
        if (i * chunksize) >= n:
            return
        if ((i+1) * chunksize + overlap) < n:
            # yield data[i*chunksize:(i+1)*chunksize + overlap]
            yield (i*chunksize, (i+1)*chunksize + overlap)
        else:
            # yield data[i*chunksize:n]
            yield (i*chunksize, n)

af = al.Sndfile(filename)

# find threshold
af.seek(0)
d = af.read_frames(baselineTime) # use first half-second
f = waveletfilter.waveletfilter(d, minlevel=filterMin, maxlevel=filterMax)
threshold = detect.calculate_threshold(f)

spikeindices = None
spikewaveforms = None
nFrames = af.nframes

for (s, e) in chunk(nFrames, framesPerChunk, chunkOverlap):
    af.seek(s)
    d = af.read_frames(e-s)
    if len(d) == 0:
        logging.warning("Read 0 frames from file??")
        continue
    
    # filter
    f = waveletfilter.waveletfilter(d, minlevel=filterMin, maxlevel=filterMax)
    
    # detect
    si, sw = detect.find_spikes(f, threshold, direction = detectionDirection, prew = prew)
    si = np.array(si)
    sw = np.array(sw)
    
    # throw out any spikes that began in overlap
    # TODO test that this doesn't miss some spikes if they are right on the boundry
    goodspikes = np.where(si < (framesPerChunk + prew))[0]
    
    if len(goodspikes) == 0:
        continue
    
    if spikewaveforms is None:
        spikeindices = si[goodspikes] + s
        spikewaveforms = sw[goodspikes]
    else:
        spikeindices = np.hstack((spikeindices, si[goodspikes] + s))
        # print spikewaveforms.shape, sw.shape, goodspikes, si[goodspikes].shape
        spikewaveforms = np.vstack((spikewaveforms, sw[goodspikes]))
        print "%i%% done: %i spikes" % (int((e * 100.) / nFrames), len(spikeindices))

logging.debug("Found %i spikes" % len(spikeindices))
# measure
spikefeatures = waveletfeatures.wavelet_features(spikewaveforms, nfeatures=nfeatures)

# cluster

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# --------------------------------   Plotting   -----------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

from mpl_toolkits.mplot3d import Axes3D
pl.figure()
# plot spikes
pl.subplot(221)
af.seek(0)
if plotRawWave:
    pl.plot(waveletfilter.waveletfilter(af.read_frames(af.nframes), minlevel=3, maxlevel=6))
se = np.abs(np.max(spikewaveforms,1)) # get spike 'extremes'
pl.scatter(np.array(spikeindices),se)

# plot waveforms
pl.subplot(222)
pl.plot(np.transpose(spikewaveforms))

# plot features
ax = pl.gcf().add_subplot(2, 2, 4, projection='3d')
ax.scatter(spikefeatures[:,0],spikefeatures[:,1],spikefeatures[:,2])

pl.figure()
for x in xrange(nfeatures):
    for y in xrange(nfeatures):
        if x > y:
            pl.subplot(nfeatures,nfeatures,x+y*nfeatures+1)
            pl.scatter(spikefeatures[:,x], spikefeatures[:,y], s=1)
            pl.gca().set_axis_off()
            xr = spikefeatures[:,x].max() - spikefeatures[:,x].min()
            b = 1/8.
            pl.xlim(spikefeatures[:,x].min()-xr*b, spikefeatures[:,x].max()+xr*b)
            yr = spikefeatures[:,y].max() - spikefeatures[:,y].min()
            pl.ylim(spikefeatures[:,y].min()-yr*b, spikefeatures[:,y].max()+yr*b)

pl.show()