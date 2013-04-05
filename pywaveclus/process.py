#!/usr/bin/env python

import logging
import os

import numpy as np

from . import ops
from . import utils


def get_operations(fns, ica=None, cfg=None):
    cfg = utils.load_config() if cfg is None else cfg
    info = {}

    # reader
    if ica is None:
        reader, info['reader'] = ops.reader.from_config(fns, cfg)
    else:
        reader, info['reader'] = ops.reader.from_config(fns, cfg, ica)

    # filt
    ff, info['filter'] = ops.filter.from_config(cfg)

    # get baseline
    bf, info['baseline'] = ops.baseline.from_config(cfg)
    baseline = ff(bf(reader))
    reader.seek(0)

    # detect
    df, info['detect'] = ops.detect.from_config(baseline, cfg)

    # extract
    ef, info['extract'] = ops.extract.from_config(cfg)

    # cluster
    cf, info['cluster'] = ops.cluster.from_config(cfg)
    return info, cfg, reader, ff, df, ef, cf


def process_file(cfg, reader, ff, df, ef, cf, store):
    start, end = utils.parse_time_range(
        cfg.get('main',  'timerange'), 0, len(reader))
    store.save_timerange(start, end)
    sd = dict([(i, []) for i in xrange(reader.nchan)])
    overlap = reader._chunkoverlap
    csize = reader._chunksize
    pre = cfg.get('extract', 'pre')
    # this may take up too much memory
    # if so, write inds & waves to disk as they are found
    for chunk, cs, ce in reader.chunk(start, end):
        # chunk is a 2D array, TODO parallel here?
        for (chi, ch) in enumerate(chunk):
            fd = ff(ch)
            # get potential spikes
            psis = df(chi, fd)
            sis = [i for i in psis if (i - pre) < (csize - overlap)]
            sws = ef(fd, sis)  # get waveforms
            chsd = [(si + cs, sw) for (si, sw) in
                    zip(sis, sws) if (sw is not None)]
            sd[chi] = chsd
            del sws
            #sd[chi] += [(si + cs, sw) for (si, sw) in
            #            zip(sis, sws) if (sw is not None)]
        # write to file and cleanup
        for chi in sd:
            if len(sd[chi]):
                sis, sws = zip(*sd[chi])
                store.create_spikes(chi, sis, sws)
                del sws
                sd[chi] = []
    for chi in sd:
        sws = store.load_waves(chi)
        clusters, info = cf(sws)
        store.update_spikes(chi, clusters)
        store.save_cluster_info(chi, info)
        store.save_filename(chi, reader.filenames[chi])

        #d = {}
        #d['filename'] = reader.filenames[chi]
        #d['index'] = chi
        #d['indices'] = sis
        #d['waveforms'] = sws
        #d['clusters'] = clusters
        #d['cluster_info'] = info
        #cd[chi] = d
    return store


def old_process_file(customCfg=None, options=None):
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
        clusters = np.array([0] * len(waveforms))
        return cfg, indices, waveforms, clusters, {}
    else:
        logging.debug("%i spikes before clustering" % len(indices))
        clusters, info = cfunc(waveforms)
        return cfg, indices, waveforms, clusters, info
