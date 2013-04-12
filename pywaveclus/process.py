#!/usr/bin/env python

import glob
import logging
import cPickle as pickle
import os
import re

import joblib
import numpy
#import numpy as np
import tables

from . import ops
from . import utils


def process_session(sdir, ofn='output.h5', full=False):
    afdir = sdir + '/Audio Files'
    assert os.path.exists(afdir)
    fns = sorted([os.path.realpath(fn) for fn in glob.glob(afdir + '/input*')])
    print "filenames:", fns

    ica = None
    ica_fn = sdir + '/ica.p'
    if os.path.exists(ica_fn):
        ica = pickle.load(open(ica_fn))
        print "found ica"
    else:
        print "no ica found"
    #ica = None
    #ica_mm = sdir + '/mixingmatrix'
    #ica_um = sdir + '/unmixingmatrix'
    #if (os.path.exists(ica_mm) and os.path.exists(ica_um)):
    #    mm = numpy.matrix(numpy.loadtxt(ica_mm))
    #    um = numpy.matrix(numpy.loadtxt(ica_um))
    #    cm = mm * um
    #    ica = dict(cm=cm, fns=fns)
    #    print "found ica"
    #else:
    #    print "no ica found"

    store = ops.store.hdf5.SpikeStorage(
        tables.openFile(ofn, 'w'))

    cfg = utils.load_config()
    # make channel re-mapping function
    indexre = cfg.get('main', 'indexre')
    order = cfg.get('main', 'order')

    def reorder(index):
        fn = fns[index]
        ms = re.findall(indexre, fn)
        assert len(ms) == 1
        i = int(ms[0])
        return order[i]

    store.convert_channel_index = reorder

    info, cfg, reader, ff, df, ef, cf = get_operations(fns, ica=ica, cfg=cfg)

    process_file(info, cfg, reader, ff, df, ef, cf, store)
    #store.save_info(info)  # do this in process_file
    #store.close()
    if full:
        return store, dict(cfg=cfg, reader=reader, fns=fns, ff=ff,
                           df=df, ef=ef, cf=cf, info=info, ica=ica)
    return store


def get_operations(fns, ica=None, cfg=None):
    cfg = utils.load_config() if cfg is None else cfg
    info = dict(fns=fns, cfg=cfg.as_dict())

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


def process_chunk(chi, ch, ff, df, pre, csize, overlap, ef, cs):
    fd = ff(ch)  # time: 38%
    # get potential spikes
    #sis = []
    #for i in df(fd):
    #    if (i - pre) < csize:
    #        sis.append(i)
    #    else:
    #        print i, pre, csize, overlap
    #psis = df(fd)
    #sis = [i for i in psis if (i - pre) < (csize - overlap)]
    sis = [i for i in df(fd) if (i - pre) < csize]
    sws = ef(fd, sis)  # get waveforms
    return [(si + cs, sw) for (si, sw) in
            zip(sis, sws) if (sw is not None)]


def process_file(info, cfg, reader, ff, df, ef, cf, store):
    start, end = utils.parse_time_range(
        cfg.get('main',  'timerange'), 0, len(reader))
    #store.save_timerange(start, end)
    #sd = dict([(i, []) for i in xrange(reader.nchan)])
    overlap = reader._chunkoverlap
    csize = reader._chunksize
    pre = cfg.get('extract', 'pre')
    # this may take up too much memory
    # if so, write inds & waves to disk as they are found

    # the biggest problem with making this parallel is the inability to
    # pickle functions, the current library structure builds a set of
    # functions from a configuration (ff, df, ef...) all of which are
    # not pickleable
    # to make this parallel I need to make these functions pickleable
    # possibly I could define them as (callable) objects that get
    # recreated in each process. This is straightforward for everything
    # except detection which has channel specific thresholds
    # A work-around for this would be to make detection follow the form of:
    #   df(i)(d)
    # instead of
    #   df(i, d)
    # to to parallelize this, I could deal out each channels detect function
    # from the main process
    njobs = cfg.get('main', 'njobs')
    assert njobs != 0, "njobs cannot == 0"
    for chunk, cs, ce in reader.chunk(start, end):  # 45%
        sd = joblib.Parallel(njobs)(joblib.delayed(process_chunk)(
            chi, ch, ff, df[chi], pre, csize, overlap, ef, cs) for
            (chi, ch) in enumerate(chunk))  # 20%
        for (i, d) in enumerate(sd):
            if len(d):
                sis, sws = zip(*d)
                store.create_spikes(i, sis, sws)
                del sws
        sd = None
    # these channel indices won't necessarily be the same as before
    # if channels where reordered (see cfg['main']['order'] and indexre)
    info['clustering'] = {}
    for chi in xrange(reader.nchan):
        sws = store.load_waves(chi)
        clusters, cinfo = cf(sws)  # time: 24%
        store.update_spikes(chi, clusters)
        info['clustering'][chi] = cinfo
    store.info = info
    store.time_range = (start, end)
    return store
