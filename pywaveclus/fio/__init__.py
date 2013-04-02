#!/usr/bin/env python

import audio

__all__ = ['audio']


def reader_from_kwargs(filenames, *args, **kwargs):
    if len(args):
        return audio.ICAReader(filenames, *args, **kwargs)
    else:
        return audio.Reader(filenames, **kwargs)


def reader_from_config(filenames, cfg, *args, **kwargs):
    s = kwargs.get('section', 'reader')
    for k in ('dtype', 'chunksize', 'chunkoverlap'):
        kwargs[k] = cfg.get(s, k)
    return reader_from_kwargs(filenames, *args, **kwargs)
