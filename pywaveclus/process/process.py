#!/usr/bin/env python

import config
import logging
import os

import numpy as np

from .. import data
from .. import dsp
from .. import detect
from .. import cluster
from .. import extract
from .. import utils


def get_operations(customCfg=None, options=None):
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
    efunc = extract.extract_from_config(cfg)

    # cluster
    cfunc = cluster.cluster_from_config(cfg)

    return cfg, reader, ffunc, dfunc, efunc, cfunc


def process_file(customCfg=None, options=None):
    cfg, reader, ffunc, dfunc, efunc, cfunc = \
            get_operations(customCfg, options)

    outdir = cfg.get('main', 'outputdir').strip()
    filename = cfg.get('main', 'filename')
    if outdir == '':
        outdir = os.path.dirname(os.path.abspath(filename)) + \
                '/pyc_' + os.path.basename(filename)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logging.root.addHandler( \
            logging.FileHandler('%s/pyc.log' % outdir, mode='w'))

    cfg.pretty_print(logging.debug)

    csize = cfg.getint('reader', 'chunksize')
    coverlap = cfg.getint('reader', 'chunkoverlap')

    start, end = utils.parse_time_range( \
            cfg.get('main', 'timerange'), 0, reader.nframes, int)
    logging.debug("Timerange: %i to %i samples" % (start, end))
    logging.debug("Chunk size %i, overlap %i" % (csize, coverlap))

    pre = cfg.getint('detect', 'pre')
    assert pre * 2 < coverlap, \
            "chunk overlap[%i] must be more than 2*pre[%i]" % \
            (coverlap, pre * 2)

    #waveforms = None
    indices = None
    nspikes = 0
    for chunk, chunkstart, chunkend in \
            reader.chunk(start, end, csize, coverlap):
        sis = dfunc(ffunc(chunk))
        logging.debug("Found %i spikes between %i and %i" % \
                (len(sis), chunkstart, chunkend))

        sis = np.array(sis)
        #sws = np.array(sws)

        goodIs = np.where(sis < (csize + pre * 2.))[0]

        if len(goodIs) == 0:
            continue

        # make times releative to first sample (audio time 0)
        sis += chunkstart

        nspikes += len(goodIs)

        if indices is None:
            indices = sis[goodIs]
            #waveforms = sws[goodIs]
        else:
            indices = np.hstack((indices, sis[goodIs]))
            #waveforms =  np.vstack((waveforms, sws[goodIs]))

    # get waveforms from 'adjacent' channels
    adjacentFiles = cfg.get('main', 'adjacentfiles')
    logging.debug("adjacent files: %s" % adjacentFiles)
    readers = [reader, ]
    if adjacentFiles.strip() != '':
        adjacentFiles = adjacentFiles.split()
        readers = [reader, ]
        for adjacentFile in adjacentFiles:
            dtype = np.dtype(cfg.get('reader', 'dtype'))
            lockdir = cfg.get('reader', 'lockdir')
            if lockdir.strip() == '':
                lockdir = None
            ref = cfg.get('main', 'reference')
            if ref.strip() != '':
                readers.append(data.audio.ReferencedReader( \
                        adjacentFile, ref, dtype, lockdir))
            else:
                readers.append(data.audio.Reader(adjacentFile, dtype, lockdir))
            #readers.append(adj)
            #for si in indices:
            #    pass
    logging.debug("%i spikes (by nspikes)" % nspikes)
    if indices is None:
        return cfg, [], [], [], {}
    waveforms = np.array(efunc(readers, indices, ffunc))

    logging.debug("%i spikes (by nspikes)" % nspikes)
    if not (indices is None):
        logging.debug("%i spikes (by indices)" % len(indices))
    else:
        logging.debug("0 spikes (by indices)")

    if waveforms is None:
        return cfg, [], [], [], {}
    elif len(waveforms) < cfg.getint('cluster', 'minspikes'):
        logging.warning("%i spikes less than minspikes [%i], " \
                "assigning all to 1 cluster" % \
                (len(waveforms), cfg.getint('cluster', 'minspikes')))
        clusters = np.array([1] * len(waveforms))
        return cfg, indices, waveforms, clusters, {}
    else:
        logging.debug("%i spikes before clustering" % len(indices))
        clusters, info = cfunc(waveforms)
        return cfg, indices, waveforms, clusters, info
