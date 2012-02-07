#!/usr/bin/env python

import logging, os, sys

import numpy as np
import pylab as pl
import matplotlib
import tables

import pywaveclus

logging.basicConfig(level=logging.DEBUG)

cfg, times, waves, clusters, info = pywaveclus.process.process.process_file()

logging.debug("%i spikes before saving" % len(times))

# ------------------------------------- move this into process_file at some point

# get output directory (assign default if missing)
outdir = cfg.get('main','outputdir').strip()
filename = cfg.get('main','filename')
if outdir == '': outdir = os.path.dirname(os.path.abspath(filename)) + '/pyc_' + os.path.basename(filename)
# 
# if not os.path.exists(outdir): os.makedirs(outdir)
# logging.root.addHandler(logging.FileHandler('%s/pyc.log' % outdir, mode='w'))

outfile = '/'.join((outdir, os.path.splitext(os.path.basename(filename))[0])) + '.h5'
logging.debug("Saving results to %s" % outfile)

pre = cfg.getint('detect','pre')
post = cfg.getint('detect','post')
nadj = len(cfg.get('main','adjacentfiles').split())

class description(tables.IsDescription):
    time = tables.Int64Col()
    wave = tables.Float64Col(shape=(pre + post))
    clu = tables.UInt8Col()

hdfFile = tables.openFile(outfile,"w")
spiketable = hdfFile.createTable("/","SpikeTable",description)

minref = 22 # 0.0005 s at 44100 Hz

logging.debug("writing to hdf5 file")
for i in xrange(len(times)-1):
    nspike = np.where(clusters[i+1:] == clusters[i])[0]
    if (len(nspike) > 0) and (times[nspike[0]+i+1] - times[i] < minref):
        # caton skips spikes that are the same cluster and
        # occur close together in time
        logging.debug("Throwing away spike at %i" % times[i])
    else:
        spiketable.row['wave'] = waves[i][0]
        spiketable.row['time'] = times[i]
        spiketable.row['clu'] = clusters[i]
        spiketable.row.append()

clustering_info = hdfFile.createGroup('/', 'Clustering', 'Clustering info')
for k, v in info.iteritems():
    hdfFile.createArray(clustering_info, k, v)
# spcgroup = hdfFile.createGroup('/', 'SPC', 'SPC results')
# hdfFile.createArray(spcgroup,'cdata',cdata)
# hdfFile.createArray(spcgroup,'ctree',ctree)

logging.debug("closing hdf5 file")
spiketable.flush()
hdfFile.close()


if len(times) == 0:
    logging.debug("No spikes found")
    sys.exit(0)
# generate plots
logging.debug("Plotting")

nspikes = len(times)
if nspikes > 20000:
    ds = nspikes / 20000
    times = np.array(times[::ds])
    waves = np.array(waves[::ds])
    clusters = np.array(clusters[::ds])
else:
    times = np.array(times)
    waves = np.array(waves)
    clusters = np.array(clusters)

nclusters = np.max(clusters) + 1
colors = matplotlib.cm.get_cmap('jet',nclusters)


pl.figure(1)
pl.suptitle('Spike Times')
pl.xlabel('Time(samples)')
pl.ylabel('Amplitude')

n_features = cfg.getint('cluster','nfeatures')

pl.figure(2, figsize=(nclusters*3,n_features))
pl.suptitle("Spike Waveforms")
pl.ylabel("Amplitude")

pl.figure(3, figsize=(n_features,n_features))
pl.suptitle("Spike Features")

logging.debug("Waveform array shape: %s" % str(waves.shape))

# recompute features
features = pywaveclus.dsp.pca.features_from_info(waves, info)
logging.debug("Features array shape: %s" % str(features.shape))

for cl in xrange(nclusters):
    gi = np.where(clusters == cl)
    if (len(gi) == 0): continue
    gi = gi[0]
    if (len(gi) == 0): continue
    st = times[gi]
    sw = waves[gi,0,:] # only plot first wave (from main channel)
    se = sw[:,pre]
    c = colors(cl)
    # plot times
    pl.figure(1)
    pl.scatter(st,se,label='%i' % cl, color=c)
    
    for i in xrange(nadj+1):
        sw = waves[gi,i,:]
        # plot waves
        pl.figure(2)
        pl.subplot((nadj+1)*2,nclusters,cl+1+nclusters*(i*2))
        pl.plot(sw.T, color=c)
        pl.title("N=%i" % len(sw))
        pl.subplot((nadj+1)*2,nclusters,cl+nclusters+1+nclusters*(i*2))
        av = np.average(sw,0)
        sd = np.std(sw,0,ddof=1)
        se = sd / np.sqrt(len(sw))
        pl.fill_between(range(len(av)), av+se*1.96, av-se*1.96, color=c, alpha=0.5)
        pl.plot(av, color=c)
    
    # plot features
    pl.figure(3)
    sf = features[gi,:]
    ndims = int(sf.shape[1])
    logging.debug("Feature Dims: %i, %s, %s" % (ndims, str(sf.shape), str(type(features))))
    for x in xrange(ndims):
        for y in xrange(ndims):
            if y < x:
                #spt = (ndims, ndims, x + y * ndims + 1)
                pl.subplot(ndims, ndims, x + y * ndims + 1)
                #logging.debug("Subplot tuple: %s" % str(spt))
                #sys.stdout.flush()
                #sys.stderr.flush()
                #pl.subplot((ndims, ndims, x + y * ndims + 1))
                #pl.subplot(*spt)
                pl.scatter(sf[:,x], sf[:,y], s = 1, label='%i', color=c, alpha = 0.5)
                pl.xticks([])
                pl.yticks([])

logging.debug("Saving plots")
for (i,name) in enumerate(['times','waves','features']):
    pl.figure(i+1)
    pl.savefig('/'.join((outdir, name + '.png')))
