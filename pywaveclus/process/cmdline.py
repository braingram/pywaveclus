#!/usr/bin/env python

import numpy

import config
from .. import operations

import joblib


def parse(args):
    """
    Parameters
    ----------
    args : list of strings
        similar to sys.argv[1:]

    Returns
    -------
    files : list of strings
        Input filenames
    cfg : cconfig.CConfig
        Configuration file object loaded with config.load
    """
    if ':' in args:
        i = args.index(':')
        files = args[:i]
        args = args[i + 1:]
    else:
        files = args
        args = []
    cfg = config.load(args=args)
    return files, cfg


def process(files, cfg, section='main'):
    """docstring for process"""
    # find operations (based on cfg)
    reader = operations.get_reader(files, cfg)
    filt = operations.get_filt(cfg)
    detect = operations.get_detect(cfg, reader, filt)
    extract = operations.get_extract(cfg)
    cluster = operations.get_cluster(cfg)
    writer = operations.get_writer(cfg)

    # get main options
    njobs = cfg.get(section, 'njobs')
    nchan = reader.nchan

    # run
    indices = numpy.array([]).reshape((nchan, 0))
    waveforms = numpy.array([]).reshape((nchan, 0, extract.pre + extract.post))
    for chunk, start, end in reader.chunks(full=True):
        # parallel filter and detect
        fdata = joblib.Parallel(njobs=njobs)(joblib.delayed(filt)(ch) \
                for ch in chunk)
        sis = joblib.Parallel(njobs=njobs)(joblib.delayed(detect)(ch) \
                for ch in chunk)\
        # remove spikes that start > chunksize
        sis = sis[sis <= reader.chunksize]
        if len(sis) == 0:
            continue
        waves = joblib.Parallel(njos=njobs)(joblib.delayed(extract)(ch, fd) \
                for ch, fd in zip(fdata, sis))
        # offset spike indices to absolute position
        sis += start
        numpy.hstack((indices, sis))
        numpy.hstack((waveforms, waves))

    # indices = spike indices (locations)
    # waveforms = spike waveforms (only single channel now)
    # write data to file
    writer.write_indices(indices)
    writer.write_waveforms(waveforms)

    ## features
    #features = joblib.Parallel(njobs=njobs)(joblib.delayed(feature)(ch) \
    #        for ch in waveforms)
    #writer.write_features(features)

    # cluster: returns clusters and cluster_info
    clusters = joblib.Parallel(njobs=njobs)(joblib.delayed(cluster)(ch) \
            for ch in waveforms)
    cluster_info = [c[1] for c in clusters]
    clusters = [c[0] for c in clusters]

    # write cluster to data
    writer.write_clusters(clusters)
    writer.write_cluster_info(cluster_info)

    # plot????
    #[writer(cluster(extract(detect(filt(chunk))))) \
    #        for chunk in reader.chunks()]

    # return output stuff
