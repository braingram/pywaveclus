#!/usr/bin/env python
"""
Extracts samples from a data file reader
"""

from . import simple


def from_kwargs(**kwargs):
    method = kwargs.pop('method')
    if 'method' == 'simple':
        tr = kwargs['range']
        return lambda x: simple.simple(x, tr)
    else:
        raise ValueError('Unknown baseline method: %s' % method)


def from_config(cfg, section='baseline'):
    kwargs = {}
    for k in ('method', ):
        if cfg.has_option(section, k):
            kwargs[k] = cfg.get(section, k)
    return from_kwargs(**kwargs)
