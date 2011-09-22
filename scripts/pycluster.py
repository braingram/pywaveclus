#!/usr/bin/env python

import logging, os

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

class description(tables.IsDescription):
    time = tables.Int32Col()
    wave = tables.Float64Col(shape=(pre + post,))
    clu = tables.Int8Col()

hdfFile = tables.openFile(outfile,"w")
spiketable = hdfFile.createTable("/","SpikeTable",description)

minref = 22 # 0.0005 s at 44100 Hz

logging.debug("writing to hdf5 file")
for i in xrange(len(times)-1):
    if (times[i+1] - times[i] < minref) and (clusters[i+1] == clusters[i]):
        # caton skips spikes that are the same cluster and
        # occur close together in time
        logging.debug("Throwing away spike at %i" % times[i])
    else:
        spiketable.row['wave'] = waves[i]
        spiketable.row['time'] = times[i]
        spiketable.row['clu'] = clusters[i]
        spiketable.row.append()

# spcgroup = hdfFile.createGroup('/', 'SPC', 'SPC results')
# hdfFile.createArray(spcgroup,'cdata',cdata)
# hdfFile.createArray(spcgroup,'ctree',ctree)

logging.debug("closing hdf5 file")
spiketable.flush()
hdfFile.close()


# generate plots
logging.debug("Plotting")

times = np.array(times)
waves = np.array(waves)
clusters = np.array(clusters)
nclusters = np.max(clusters) + 1
colors = matplotlib.cm.get_cmap('jet',nclusters)


pl.figure(1)
pl.suptitle('Spike Times')
pl.xlabel('Time(samples)')
pl.ylabel('Amplitude')

pl.figure(2, figsize=(nclusters*3,6))
pl.suptitle("Spike Waveforms")
pl.ylabel("Amplitude")

pl.figure(3)
pl.suptitle("Spike Features")

for cl in xrange(nclusters):
    gi = np.where(clusters == cl)[0]
    if (len(gi) == 0): continue
    st = times[gi]
    sw = waves[gi]
    se = sw[:,pre]
    c = colors(cl)
    # plot times
    pl.figure(1)
    pl.scatter(st,se,label='%i' % cl, color=c)
    
    # plot waves
    pl.figure(2)
    pl.subplot(2,nclusters,cl+1)
    pl.plot(sw.T, color=c)
    pl.title("N=%i" % len(sw))
    pl.subplot(2,nclusters,cl+nclusters+1)
    av = np.average(sw,0)
    sd = np.std(sw,0,ddof=1)
    se = sd / np.sqrt(len(sw))
    pl.fill_between(range(len(av)), av+se*1.96, av-se*1.96, color=c, alpha=0.5)
    pl.plot(av, color=c)
    
    # plot features

logging.debug("Saving plots")
for (i,name) in enumerate(['times','waves','features']):
    pl.figure(i+1)
    pl.savefig('/'.join((outdir, name + '.png')))
