#!/usr/bin/env python

import config
import logging

import numpy as np

from .. import data
from .. import dsp
from .. import detect
from .. import cluster
from .. import utils

def get_operations(customCfg = None, options = None):
    """
    fd = filt(data)
    sis, sws = detect(fd)
    # remove chunk overlap spikes
    clusters, ... = cluster(sws)
    """
    cfg = config.load(customCfg)
    cfg.read_commandline(options)
    
    # reader
    reader = data.reader_from_config(cfg)
    
    # filt
    ffunc = dsp.filt_from_config(cfg)
    
    # detect
    dfunc = detect.detect_from_config(reader, ffunc, cfg)
    # TODO remove spikes in chunk overlap
    
    # extract ? part of detect ?
    
    # cluster
    cfunc = cluster.cluster_from_config(cfg)
    
    return cfg, reader, ffunc, dfunc, cfunc

def process_file(customCfg = None, options = None):
    cfg, reader, ffunc, dfunc, cfunc = get_operations(customCfg, options)
    
    outdir = cfg.get('main','outputdir').strip()
    filename = cfg.get('main','filename')
    if outdir == '': outdir = os.path.dirname(os.path.abspath(filename)) + '/pyc_' + os.path.basename(filename)
    
    if not os.path.exists(outdir): os.makedirs(outdir)
    logging.root.addHandler(logging.FileHandler('%s/pyc.log' % outdir, mode='w'))
    
    cfg.pretty_print(logging.debug)
    
    csize = cfg.getint('reader','chunksize')
    coverlap = cfg.getint('reader','chunkoverlap')
    
    start, end = utils.parse_time_range(cfg.get('main','timerange'), 0, reader.nframes, int)
    
    pre = cfg.getint('detect','pre')
    assert pre*2 < coverlap, "chunk overlap[%i] must be more than 2*pre[%i]" % (coverlap, pre*2)
    
    waveforms = None
    indices = None
    for chunk, start, end in reader.chunk(start, end, csize, coverlap):
        sis, sws = dfunc(ffunc(chunk))
        logging.debug("Found %i spikes between %i and %i" % (len(sis), start, end))
        
        sis = np.array(sis)
        sws = np.array(sws)
        
        goodIs = np.where(sis < (csize + pre*2.))[0]
        
        if len(goodIs) == 0: continue
        
        sis += start
        
        if waveforms is None:
            indices = sis[goodIs]
            waveforms = sws[goodIs]
        else:
            indices = np.hstack((indices, sis[goodIs]))
            waveforms =  np.vstack((waveforms, sws[goodIs]))
    
    if waveforms is None:
        return cfg, [], [], [], []
    else:
        clusters, info = cfunc(waveforms)
        return cfg, indices, waveforms, clusters, info
    