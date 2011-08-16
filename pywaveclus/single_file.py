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

import waveletfilter, detect, waveletfeatures, cluster

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
prew = 40 # for spike waveform capture
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

# np.savetxt('features', spikefeatures, delimiter=',', newline='],\n')

# cluster
clusters, tree, cdata = cluster.cluster(spikefeatures)

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# --------------------------------   Plotting   -----------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

print "Found Clusters:"
print [len(c) for c in clusters]

from mpl_toolkits.mplot3d import Axes3D
colors = ['k','b','g','r','m','y']

# figures:
#  1 : spike times
#  2 : spike waveforms
#  3 : features
#  4 : cluster information

spikewaveforms = np.array(spikewaveforms)
spikeindices = np.array(spikeindices)

if plotRawWave:
    pl.figure(1)
    af.seek(0)
    pl.plot(waveletfilter.waveletfilter(af.read_frames(af.nframes), minlevel=3, maxlevel=6))

# setup 3d subplot
pl.figure(1)
pl.figure(2)
pl.figure(3)
ax = pl.gcf().add_subplot(2, 2, 3, projection='3d')

for (color, cluster) in zip(colors,clusters):
    if len(cluster) == 0: continue
    sw = spikewaveforms[cluster]
    si = spikeindices[cluster]
    # plot times
    pl.figure(1)
    se = sw[:,prew]
    # if detectionDirection == 'neg':
    #     se = np.min(sw,1)
    # elif detectionDirection == 'pos':
    #     se = np.max(sw,1)
    # else: # 'both'
    #     
    #     peaks = np.max(sw,1)
    #     valleys = np.min(sw,1)
    #     peaks = peaks[np.where(peaks > threshold)[0]]
    #     vallyes = valleys[np.where(valleys < -threshold)[0]]
    #     se = np.union1d(peaks,valleys)
    pl.scatter(si, se, c=color, edgecolors=None)
    
    # waveforms
    pl.figure(2)
    pl.plot(np.transpose(sw), c=color)
    
    # features
    features = spikefeatures[cluster]
    ax.scatter(features[:,0],features[:,1],features[:,2],c=color)
    
    pl.figure(3)
    for x in xrange(nfeatures):
        for y in xrange(nfeatures):
            if x > y:
                pl.subplot(nfeatures,nfeatures,x+y*nfeatures+1)
                pl.scatter(features[:,x], features[:,y], s=10, alpha=0.5, c=color, edgecolors=None)

# set axes for features figure (3)
pl.figure(3)
for x in xrange(nfeatures):
    for y in xrange(nfeatures):
        if x > y:
            pl.subplot(nfeatures,nfeatures,x+y*nfeatures+1)
            pl.gca().set_xticks([])
            pl.gca().set_yticks([])
            b = 1/8.
            xr = spikefeatures[:,x].max() - spikefeatures[:,x].min()
            pl.xlim(spikefeatures[:,x].min()-xr*b, spikefeatures[:,x].max()+xr*b)
            yr = spikefeatures[:,y].max() - spikefeatures[:,y].min()
            pl.ylim(spikefeatures[:,y].min()-yr*b, spikefeatures[:,y].max()+yr*b)

pl.figure(4)
pl.imshow(tree, interpolation='nearest')

pl.figure(5)
isi = np.diff(spikeindices) / 44100.
pl.hist(isi)

# pl.figure()
# # plot spikes
# pl.subplot(221)
# af.seek(0)
# if plotRawWave:
#     pl.plot(waveletfilter.waveletfilter(af.read_frames(af.nframes), minlevel=3, maxlevel=6))
# se = np.abs(np.max(spikewaveforms,1)) # get spike 'extremes'
# pl.scatter(np.array(spikeindices),se)


# # plot waveforms
# pl.subplot(222)
# pl.plot(np.transpose(spikewaveforms))

# plot features
# ax = Axes3D(pl.figure())
# 
# for (i,c) in enumerate(clusters):
#     d = features[c]
#     ax.scatter(d[:,0],d[:,1],d[:,2],c=colors[i])
# 
# pl.figure()
# 
# nf = features.shape[1]
# for x in xrange(nf):
#     for y in xrange(nf):
#         if x > y:
#             pl.subplot(nf,nf,x+y*nf+1)
#             for (i, c) in enumerate(clusters):
#                 d = features[c]
#                 pl.scatter(d[:,x], d[:,y], c=colors[i], s=10, alpha=0.5, edgecolors=None)
#                 # pl.scatter(features[:,x], features[:,y], s=1)
#             # pl.gca().set_axis_off()
#             pl.gca().set_xticks([])
#             pl.gca().set_yticks([])
#             b = 1/8.
#             xr = features[:,x].max() - features[:,x].min()
#             pl.xlim(features[:,x].min()-xr*b, features[:,x].max()+xr*b)
#             yr = features[:,y].max() - features[:,y].min()
#             pl.ylim(features[:,y].min()-yr*b, features[:,y].max()+yr*b)
# 
# pl.figure()
# pl.imshow(tree, interpolation='nearest')
# pl.figure()
# pl.imshow(cdata, interpolation='nearest')
# pl.show()

pl.show()