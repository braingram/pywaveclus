#!/usr/bin/env python

import logging

import joblib
import numpy

import config
from .. import operations


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
    logging.root.setLevel(logging.DEBUG)

    # find operations (based on cfg)
    reader = operations.get_reader(files, cfg)
    filt = operations.get_filt(cfg)
    detect = operations.get_detect(cfg, reader, filt)
    extract = operations.get_extract(cfg)
    #feature = operations.get_feature(cfg)
    cluster = operations.get_cluster(cfg)
    writer = operations.get_writer(cfg)

    # get main options
    n_jobs = cfg.getint(section, 'n_jobs')
    nchan = reader.nchan

    # run
    indices = []
    waveforms = []
    for i in xrange(nchan):
        indices.append(numpy.array([]))
        waveforms.append(numpy.array([]).reshape(0, \
                extract.pre + extract.post))
    #indices = numpy.array([]).reshape((nchan, 0))
    #waveforms = numpy.array([]).reshape((nchan, 0, \
    #        extract.pre + extract.post))
    for chunk, start, end in reader.chunk(full=True):
        logging.debug("Processing chunk: %s, %s, %s" \
                % (chunk.shape, start, end))
        #print "Processing chunk: %s, %s, %s" \
        #        % (chunk.shape, start, end)
        # parallel filter and detect
        fdata = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(filt)(ch) \
                for ch in chunk)
        sis = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(detect)(fd, i) \
                for (i, fd) in enumerate(fdata))
        # remove spikes that start > chunksize
        sis = [s[s <= reader.chunksize] for s in sis]
        # remove spikes that start < extract.pre
        sis = [s[s >= extract.pre] for s in sis]
        # reemove spikes that go over the end of the data
        sis = [s[s <= len(fdata[0]) - extract.post] for s in sis]
        waves = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(extract)(fd, \
                inds) for fd, inds in zip(fdata, sis))
        # offset spike indices to absolute position
        sis = [s + start for s in sis]
        for i in xrange(nchan):
            indices[i] = numpy.hstack((indices[i], sis[i]))
            waveforms[i] = numpy.vstack((waveforms[i], waves[i]))

    logging.debug("Indices  :" + "".join([" %i" % len(i) for i in indices]))
    logging.debug("Waveforms:" + "".join([" %i" % len(w) for w in waveforms]))

    # write data to file
    writer.write_indices(indices)
    writer.write_waveforms(waveforms)

    ## features
    #features = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(feature)(ch) \
    #        for ch in waveforms)
    #feature_info = [f[1] for f in features]
    #features = [f[0] for f in features]
    #writer.write_features(features)
    #writer.write_feature_info(feature_info)

    # cluster: returns clusters and cluster_info
    clusters = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(cluster)(ch) \
            for ch in waveforms)
    features_info = [c[2] for c in clusters]
    features = [c[1] for c in clusters]
    #cluster_info = [c[1] for c in clusters]
    clusters = [c[0] for c in clusters]

    # write cluster to data
    writer.write_clusters(clusters)
    writer.write_features(features)
    writer.write_features_info(features_info)
    #writer.write_cluster_info(cluster_info)

    # plot????
    #[writer(cluster(extract(detect(filt(chunk))))) \
    #        for chunk in reader.chunks()]

    # return output stuff
    writer.close()
