#!/usr/bin/env python

import cPickle as pickle

import audio
import hdf5

__all__ = ['audio', 'hdf5']


def to_number(string):
    if '.' in string:
        return float(string)
    else:
        return int(string)


def get_reader(files, cfg, section='reader'):
    kwargs = {}
    kwargs['filenames'] = files
    kwargs['probetype'] = cfg.get(section, 'probetype')
    kwargs['indexre'] = cfg.get(section, 'indexre')
    kwargs['chunksize'] = cfg.getint(section, 'chunksize')
    kwargs['chunkoverlap'] = cfg.getint(section, 'chunkoverlap')

    if cfg.has_option(section, 'start'):
        kwargs['start'] = to_number(cfg.get(section, 'start'))
    else:
        kwargs['start'] = 0

    if cfg.has_option(section, 'stop'):
        kwargs['stop'] = to_number(cfg.get(section, 'stop'))

    if cfg.has_option(section, 'icafilename'):
        with open(cfg.get(section, 'icafilename'), 'r') as f:
            ica_info = pickle.load(f)
        return audio.ICAReader(ica_info, **kwargs)
    else:
        return audio.Reader(**kwargs)


def get_writer(cfg, section='writer'):
    kwargs = {}
    kwargs['filename'] = cfg.get(section, 'filename')
    for attr in ['pre', 'post', 'nfeatures']:
        kwargs[attr] = cfg.getint(section, attr)
    return hdf5.HDF5Writer(**kwargs)
