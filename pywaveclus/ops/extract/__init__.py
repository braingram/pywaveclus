#!/usr/bin/env python

import simple
import cubic


def from_kwargs(**kwargs):
    method = kwargs.get('method', 'simple')
    if method == 'simple':
        pre = kwargs['pre']
        post = kwargs['post']

        def f(d, i):
            return simple.simple(d, i, pre, post)
        return f, kwargs
        #return lambda data, indices: simple.simple(data, indices, pre, post), \
        #    kwargs
    elif method == 'cubic':
        raise NotImplementedError
        pre = kwargs['pre']
        post = kwargs['post']
        direction = kwargs['direction']
        oversample = kwargs['oversample']
        return lambda data, indices: cubic.cubic(
            data, indices, pre, post, direction, oversample), kwargs
    else:
        raise ValueError('Unknown extract method: %s' % method)
    pass


def from_config(cfg, section='extract'):
    kwargs = {}
    for k in ('method', 'pre', 'post', 'direction', 'oversample'):
        if cfg.has_option(section, k):
            kwargs[k] = cfg.get(section, k)
    return from_kwargs(**kwargs)
