#!/usr/bin/env python

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


def process(files, cfg):
    """docstring for process"""
    # find operations (based on cfg)
    reader = operations.get_reader(files, cfg)
    filt = operations.get_filt(cfg)
    detect = operations.get_detect(cfg, reader, filt)
    extract = operations.get_extract(cfg)
    cluster = operations.get_cluster(cfg)
    writer = operations.get_writer(cfg)

    # run
    [writer(cluster(extract(detect(filt(chunk))))) \
            for chunk in reader.chunks()]

    # return output stuff
    pass
