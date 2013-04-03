#!/usr/bin/env python

import copy

from . import audio

__all__ = ['audio']


def from_kwargs(filenames, *args, **kwargs):
    if len(args):
        info = copy.deepcopy(kwargs)
        info['ica'] = True
        return audio.ICAReader(filenames, *args, **kwargs), info
    else:
        info = copy.deepcopy(kwargs)
        info['ica'] = False
        return audio.Reader(filenames, **kwargs), info


def from_config(filenames, cfg, *args, **kwargs):
    s = kwargs.get('section', 'reader')
    for k in ('dtype', 'chunksize', 'chunkoverlap'):
        kwargs[k] = cfg.get(s, k)
    return from_kwargs(filenames, *args, **kwargs)
