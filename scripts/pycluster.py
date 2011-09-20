#!/usr/bin/env python

import logging, os

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

logging.debug("writing to hdf5 file")
for i in xrange(len(times)):
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
